"""OperatorRuntime — the core DPYC protocol engine for all operators.

Also provides ``register_standard_tools(mcp, slug, runtime, settings_fn)``
which registers all standard DPYC tools (check_balance, purchase_credits,
service_status, Secure Courier, Oracle delegation, pricing, onboarding)
on any FastMCP app.  Operators call this once and only write domain tools.

Encapsulates bootstrap, vault initialization, ledger cache, credit
gating, Secure Courier, and npub resolution. Operators instantiate
this once and delegate all DPYC protocol operations to it.

Usage::

    runtime = OperatorRuntime(
        nsec_env_var="TOLLBOOTH_NOSTR_OPERATOR_NSEC",
        tool_costs={"my_tool": ToolTier.READ, "free_tool": ToolTier.FREE},
        credential_service="my-operator",
        credential_template=CredentialTemplate(...),
    )

    # In a tool function:
    npub = runtime.resolve_npub(npub)
    cache = await runtime.ledger_cache()
    err = await runtime.debit_or_error("my_tool", npub)
    if err:
        return err
    # ... do work ...
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def resolve_npub(npub: str) -> str:
    """Validate and return the npub. No fallback, no session cache.

    Raises ValueError if npub is missing or malformed.
    """
    if not npub or not npub.startswith("npub1") or len(npub) < 60:
        raise ValueError(
            "npub is required. Pass your Nostr public key (npub1...) "
            "to identify yourself."
        )
    return npub


class OperatorRuntime:
    """Core DPYC protocol engine shared by all operators.

    Handles:
    - Bootstrap: nsec → Authority → Neon URL → vault → cache
    - Credit gating: debit_or_error, rollback
    - Secure Courier: persistent credential vault
    - Identity: resolve_npub (no OAuth coupling)
    """

    def __init__(
        self,
        *,
        nsec_env_var: str = "TOLLBOOTH_NOSTR_OPERATOR_NSEC",
        tool_costs: dict[str, int] | None = None,
        credential_service: str = "",
        credential_template: Any | None = None,
        relays: list[str] | None = None,
        constraint_gate: Any | None = None,
    ) -> None:
        self._nsec_env_var = nsec_env_var
        self._tool_costs = tool_costs or {}
        self._credential_service = credential_service
        self._credential_template = credential_template
        self._relays = relays
        self._constraint_gate = constraint_gate

        # Lazy singletons
        self._vault: Any | None = None
        self._ledger_cache: Any | None = None
        self._courier: Any | None = None
        self._operator_npub: str | None = None
        self._nsec: str | None = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def resolve_npub(self, npub: str) -> str:
        """Validate npub. Raises ValueError if invalid."""
        return resolve_npub(npub)

    def operator_npub(self) -> str:
        """Return this operator's npub, derived from nsec."""
        if self._operator_npub is not None:
            return self._operator_npub
        from pynostr.key import PrivateKey  # type: ignore[import-untyped]
        nsec = self._get_nsec()
        if nsec.startswith("nsec1"):
            pk = PrivateKey.from_nsec(nsec)
        else:
            pk = PrivateKey(bytes.fromhex(nsec))
        self._operator_npub = pk.public_key.bech32()
        return self._operator_npub

    def _get_nsec(self) -> str:
        if self._nsec is not None:
            return self._nsec
        self._nsec = os.environ.get(self._nsec_env_var, "")
        if not self._nsec:
            raise RuntimeError(f"{self._nsec_env_var} not set.")
        return self._nsec

    def _get_nsec_hex(self) -> str | None:
        """Return nsec as hex bytes for vault encryption."""
        nsec = self._get_nsec()
        if nsec.startswith("nsec1"):
            from pynostr.key import PrivateKey  # type: ignore[import-untyped]
            pk = PrivateKey.from_nsec(nsec)
            return pk.hex()
        return nsec

    # ------------------------------------------------------------------
    # Bootstrap & Vault
    # ------------------------------------------------------------------

    async def vault(self) -> Any:
        """Return the NeonVault, bootstrapping from Authority if needed."""
        if self._vault is not None:
            return self._vault

        from tollbooth.vaults import NeonVault

        # Try env var first (legacy)
        neon_url = os.environ.get("NEON_DATABASE_URL", "")
        if neon_url:
            nsec_hex = self._get_nsec_hex()
            self._vault = NeonVault(
                database_url=neon_url,
                encryption_nsec_hex=nsec_hex,
            )
            await self._vault.ensure_schema()
            return self._vault

        # Bootstrap from Authority
        from tollbooth.bootstrap import ensure_bootstrapped
        result = await ensure_bootstrapped()
        if not result.success or not result.neon_database_url:
            raise ValueError(
                f"Bootstrap failed: {result.error or 'no Neon URL'}. "
                "Operator may not be registered with an Authority."
            )

        self._vault = NeonVault(
            database_url=result.neon_database_url,
            encryption_nsec_hex=result.encryption_nsec_hex,
        )
        await self._vault.ensure_schema()
        logger.info("Vault bootstrapped from Authority (encrypted)")
        return self._vault

    async def ledger_cache(self) -> Any:
        """Return the LedgerCache, bootstrapping if needed."""
        if self._ledger_cache is not None:
            return self._ledger_cache

        from tollbooth import LedgerCache
        v = await self.vault()
        self._ledger_cache = LedgerCache(v)
        asyncio.ensure_future(self._ledger_cache.start_background_flush())
        return self._ledger_cache

    # ------------------------------------------------------------------
    # Secure Courier
    # ------------------------------------------------------------------

    async def courier(self) -> Any:
        """Return the SecureCourierService with persistent vault."""
        if self._courier is not None:
            return self._courier

        nsec = self._get_nsec()
        if not nsec:
            return None

        try:
            from tollbooth.nostr_diagnostics import probe_relay_liveness
            from tollbooth.secure_courier import SecureCourierService
        except ImportError:
            return None

        relays = self._relays or ["wss://nostr.wine"]
        if not self._relays:
            from tollbooth.nostr_diagnostics import probe_relay_liveness
            results = probe_relay_liveness(relays, timeout=5)
            live = [r["relay"] for r in results if r["connected"]]
            if not live:
                fallback = [
                    "wss://relay.primal.net", "wss://relay.damus.io",
                    "wss://nos.lol", "wss://relay.nostr.band",
                ]
                fallback_results = probe_relay_liveness(fallback, timeout=5)
                live = [r["relay"] for r in fallback_results if r["connected"]]
            if not live:
                live = relays + ["wss://relay.primal.net", "wss://nos.lol"]
            relays = live

        # Build credential vault from bootstrapped Neon
        credential_vault = None
        try:
            v = await self.vault()
            from tollbooth.vaults.neon import NeonCredentialVault
            credential_vault = NeonCredentialVault(neon_vault=v)
            await credential_vault.ensure_schema()
        except Exception as exc:
            logger.warning("No persistent credential vault: %s", exc)

        templates = {}
        if self._credential_service and self._credential_template:
            templates[self._credential_service] = self._credential_template

        self._courier = SecureCourierService(
            operator_nsec=nsec,
            relays=relays,
            credential_vault=credential_vault,
            templates=templates,
        )
        return self._courier

    # ------------------------------------------------------------------
    # Credit Gating
    # ------------------------------------------------------------------

    async def debit_or_error(
        self,
        tool_name: str,
        npub: str,
        *,
        operator_proof: str = "",
    ) -> dict[str, Any] | None:
        """Check balance and debit credits for a paid tool call.

        Returns None on success (proceed). Returns error dict on failure.
        """
        from tollbooth import ToolTier

        cost = self._tool_costs.get(tool_name, 0)

        # RESTRICTED: operator-only
        if cost == ToolTier.RESTRICTED:
            try:
                caller = resolve_npub(npub)
            except ValueError as e:
                return {"success": False, "error": str(e)}
            if caller != self.operator_npub():
                if operator_proof:
                    from tollbooth.operator_proof import verify_operator_proof
                    if verify_operator_proof(operator_proof, self.operator_npub(), tool_name):
                        return None
                return {"success": False, "error": "This tool is restricted to the operator."}
            return None

        if cost == 0:
            return None

        try:
            npub = resolve_npub(npub)
            cache = await self.ledger_cache()
        except ValueError as e:
            return {"success": False, "error": str(e)}

        # Constraint gate
        effective_cost = cost
        if self._constraint_gate and self._constraint_gate.enabled:
            ledger = await cache.get(npub)
            denial, effective_cost = self._constraint_gate.check(
                tool_name=tool_name,
                base_cost=cost,
                ledger=ledger,
                npub=npub,
            )
            if denial:
                return {
                    "success": False,
                    "error": denial.reason,
                    "constraint": denial.constraint_name,
                }

        ledger = await cache.get(npub)
        if ledger.balance_api_sats < effective_cost:
            return {
                "success": False,
                "error": (
                    f"Insufficient balance: {ledger.balance_api_sats} sats "
                    f"available, {effective_cost} required for {tool_name}."
                ),
                "balance_sats": ledger.balance_api_sats,
                "cost_sats": effective_cost,
            }

        ledger.debit(effective_cost, tool_name)
        cache.mark_dirty(npub)
        return None

    async def rollback_debit(self, tool_name: str, npub: str) -> None:
        """Rollback a debit after a tool execution failure."""
        try:
            npub = resolve_npub(npub)
            cache = await self.ledger_cache()
            cost = self._tool_costs.get(tool_name, 0)
            if cost > 0:
                ledger = await cache.get(npub)
                ledger.credit(cost, f"rollback:{tool_name}")
                cache.mark_dirty(npub)
        except Exception:
            pass  # best-effort rollback

    # ------------------------------------------------------------------
    # Onboarding
    # ------------------------------------------------------------------

    async def onboarding_status(self, settings: Any) -> dict[str, Any]:
        """Return onboarding status with vault check."""
        from tollbooth.tools.onboarding import get_onboarding_status_with_vault
        return await get_onboarding_status_with_vault(
            settings=settings,
            courier_service=await self.courier(),
            service=self._credential_service,
            operator_npub=self.operator_npub(),
        )

    async def load_credentials(self, field_names: list[str]) -> dict[str, str]:
        """Load specific credentials from the Secure Courier vault."""
        from tollbooth.tools.onboarding import load_config_from_vault
        return await load_config_from_vault(
            await self.courier(),
            self._credential_service,
            self.operator_npub(),
            field_names,
        )


# ======================================================================
# Standard tool registration
# ======================================================================


def register_standard_tools(
    mcp: Any,
    slug: str,
    rt: OperatorRuntime,
    *,
    settings_fn: Any = None,
    service_name: str = "",
    service_version: str = "",
) -> None:
    """Register all standard DPYC tools on a FastMCP app.

    Call once at module level.  Operators only write domain-specific tools.

    Args:
        mcp: The FastMCP app instance.
        slug: Tool name prefix (e.g., ``"weather"``).
        rt: The OperatorRuntime instance.
        settings_fn: Callable returning the operator's Settings instance
            (for onboarding status introspection).
        service_name: Service name for service_status (e.g., ``"tollbooth-sample"``).
        service_version: Version string for service_status.
    """
    from tollbooth.slug_tools import make_slug_tool
    tool = make_slug_tool(mcp, slug)

    # -- Credit tools --------------------------------------------------

    @tool
    async def check_balance(npub: str = "") -> dict[str, Any]:
        """Check your current credit balance and usage summary.

        Free — no credits required. Pass your npub to identify yourself.
        """
        try:
            npub = resolve_npub(npub)
            cache = await rt.ledger_cache()
        except ValueError as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        return await credits.check_balance_tool(cache, npub)

    @tool
    async def purchase_credits(amount_sats: int = 1000, npub: str = "") -> dict[str, Any]:
        """Buy credits via Bitcoin Lightning.

        Creates a Lightning invoice. Pay it with any Lightning wallet,
        then call check_payment to confirm.

        Free — no credits required to call.
        """
        try:
            npub = resolve_npub(npub)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        try:
            from tollbooth.authority_client import AuthorityCertifier
            from tollbooth.registry import resolve_authority_service
            auth_info = await resolve_authority_service(rt.operator_npub())
            cert_result = await AuthorityCertifier(auth_info["url"]).certify_credits(amount_sats)
            certificate = cert_result.get("certificate", "")
        except Exception as e:
            return {"success": False, "error": f"Authority certification failed: {e}"}

        try:
            cache = await rt.ledger_cache()
            from tollbooth.tools import credits
            return await credits.purchase_credits_tool(
                cache, npub, amount_sats, certificate,
            )
        except ValueError as e:
            return {"success": False, "error": str(e)}

    @tool
    async def check_payment(invoice_id: str, npub: str = "") -> dict[str, Any]:
        """Check the payment status of a Lightning invoice.

        Call after paying the invoice from purchase_credits.
        Free — no credits required.
        """
        try:
            npub = resolve_npub(npub)
            cache = await rt.ledger_cache()
        except ValueError as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        return await credits.check_payment_tool(cache, npub, invoice_id)

    @tool
    async def restore_credits(invoice_id: str, npub: str = "") -> dict[str, Any]:
        """Restore credits from a previously paid invoice. Free."""
        try:
            npub = resolve_npub(npub)
            cache = await rt.ledger_cache()
        except ValueError as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        return await credits.restore_credits_tool(cache, npub, invoice_id)

    @tool
    async def account_statement(npub: str = "", days: int = 30) -> dict[str, Any]:
        """Generate a customer-facing account statement with purchase history and usage.

        Returns account summary, invoice line items, active credit tranches,
        all-time per-tool usage breakdown, and recent daily usage logs.
        Free — no credits consumed.

        Args:
            npub: Your Nostr public key.
            days: Number of days of daily usage history to include (default 30).
        """
        try:
            npub = resolve_npub(npub)
            cache = await rt.ledger_cache()
        except ValueError as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        return await credits.account_statement_tool(cache, npub, days=days)

    @tool
    async def account_statement_infographic(npub: str = "", days: int = 30) -> dict[str, Any]:
        """Generate a visual SVG infographic of your account statement.

        Returns the same data as account_statement, rendered as a dark-themed
        SVG graphic with balance hero, metrics cards, health gauge, tranche
        table, and tool usage breakdown. Costs 1 api_sat per call.

        Args:
            npub: Your Nostr public key.
            days: Number of days of daily usage history to include (default 30).
        """
        err = await rt.debit_or_error("account_statement_infographic", npub)
        if err:
            return err
        try:
            npub = resolve_npub(npub)
            cache = await rt.ledger_cache()
        except ValueError as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        statement = await credits.account_statement_tool(cache, npub, days=days)
        from tollbooth.infographic import render_account_infographic
        svg = render_account_infographic(statement)
        return {
            "svg": svg,
            "generated_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        }

    # -- Service status ------------------------------------------------

    @tool
    async def service_status() -> dict[str, Any]:
        """Check the health and configuration of this service. Free."""
        import os
        vault_ok = rt._vault is not None
        courier_ok = rt._courier is not None and hasattr(rt._courier, '_exchange') and rt._courier._exchange._credential_vault is not None
        return {
            "success": True,
            "service": service_name or slug,
            "version": service_version,
            "vault_configured": vault_ok,
            "courier_has_vault": courier_ok,
            "process_id": os.getpid(),
        }

    # -- Onboarding ----------------------------------------------------

    @tool
    async def get_onboarding_status() -> dict[str, Any]:
        """Report this operator's configuration readiness.

        Shows which settings are configured, which are missing, and how
        to deliver each missing value. Free.
        """
        if settings_fn:
            return await rt.onboarding_status(settings_fn())
        return {"success": False, "error": "No settings introspection configured."}

    # -- Secure Courier ------------------------------------------------

    @tool
    async def session_status() -> dict[str, Any]:
        """Check session state — shows whether credentials are active
        or onboarding is needed. Free."""
        return {
            "success": True,
            "operator_npub": rt.operator_npub(),
            "credential_service": rt._credential_service,
            "courier_configured": rt._courier is not None,
        }

    @tool
    async def request_credential_channel(
        sender_npub: str = "",
        service: str = "",
    ) -> dict[str, Any]:
        """Open a Secure Courier channel for credential delivery.

        Sends a welcome DM with a credential template to the provided npub.
        Free.
        """
        if not sender_npub:
            sender_npub = rt.operator_npub()
        if not service:
            service = rt._credential_service
        courier = await rt.courier()
        if courier is None:
            return {"success": False, "error": "Secure Courier not configured."}
        try:
            return await courier.open_channel(
                service, greeting="", recipient_npub=sender_npub,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    async def receive_credentials(
        sender_npub: str = "",
        service: str = "",
        credential_card: str = "",
    ) -> dict[str, Any]:
        """Pick up credentials from the Secure Courier. Free."""
        if not sender_npub:
            sender_npub = rt.operator_npub()
        if not service:
            service = rt._credential_service
        courier = await rt.courier()
        if courier is None:
            return {"success": False, "error": "Secure Courier not configured."}
        try:
            return await courier.receive(
                sender_npub, service, credential_card=credential_card or None,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    async def forget_credentials(service: str = "") -> dict[str, Any]:
        """Delete vaulted credentials for key rotation. Free."""
        if not service:
            service = rt._credential_service
        npub = rt.operator_npub()
        courier = await rt.courier()
        if courier is None:
            return {"success": False, "error": "Secure Courier not configured."}
        try:
            return await courier.forget(npub, service)
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -- Oracle delegation ---------------------------------------------

    @tool
    async def how_to_join() -> dict[str, Any]:
        """Get DPYC onboarding instructions from the Oracle. Free."""
        return await _call_oracle(rt, "how_to_join")

    @tool
    async def get_tax_rate() -> dict[str, Any]:
        """Get the current DPYC certification tax rate. Free."""
        return await _call_oracle(rt, "get_tax_rate")

    @tool
    async def lookup_member(npub: str) -> dict[str, Any]:
        """Look up a DPYC community member by npub. Free."""
        return await _call_oracle(rt, "lookup_member", {"npub": npub})

    @tool
    async def about() -> dict[str, Any]:
        """Describe the DPYC ecosystem via the Oracle. Free."""
        return await _call_oracle(rt, "about")

    @tool
    async def network_advisory() -> dict[str, Any]:
        """Get active network advisories from the Oracle. Free."""
        return await _call_oracle(rt, "network_advisory")

    # -- Pricing CRUD --------------------------------------------------

    @tool
    async def get_pricing_model() -> dict[str, Any]:
        """Get the active pricing model for this operator. Free."""
        try:
            vault = await rt.vault()
            from tollbooth.pricing_store import PricingModelStore
            store = PricingModelStore(neon_vault=vault)
            from tollbooth.tools.pricing import get_pricing_model_tool
            return await get_pricing_model_tool(store, rt.operator_npub())
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @tool
    async def set_pricing_model(model_json: str) -> dict[str, Any]:
        """Set the active pricing model. RESTRICTED to operator."""
        err = await rt.debit_or_error("set_pricing_model", rt.operator_npub())
        if err:
            return err
        try:
            vault = await rt.vault()
            from tollbooth.pricing_store import PricingModelStore
            store = PricingModelStore(neon_vault=vault)
            from tollbooth.tools.pricing import set_pricing_model_tool
            return await set_pricing_model_tool(
                store, rt.operator_npub(), model_json,
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ======================================================================
# Oracle delegation helper
# ======================================================================


async def _call_oracle(
    rt: OperatorRuntime,
    tool_name: str,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call an Oracle tool via MCP-to-MCP."""
    try:
        from tollbooth.registry import resolve_oracle_service
        oracle_info = await resolve_oracle_service(rt.operator_npub())
        from fastmcp import Client
        async with Client(oracle_info["url"], auth="oauth") as client:
            result = await client.call_tool(tool_name, args or {})
            for block in getattr(result, "content", []):
                if hasattr(block, "text"):
                    import json as _json
                    try:
                        return _json.loads(block.text)
                    except (ValueError, TypeError):
                        return {"success": True, "result": block.text}
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": f"Oracle delegation failed: {e}"}
