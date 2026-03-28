"""OperatorRuntime — the core DPYC protocol engine for all operators.

Also provides ``register_standard_tools(mcp, slug, runtime)``
which registers all standard DPYC tools (check_balance, purchase_credits,
service_status, Secure Courier, Oracle delegation, pricing, onboarding)
on any FastMCP app.  Operators call this once and only write domain tools.

Encapsulates bootstrap, vault initialization, ledger cache, credit
gating, Secure Courier, OTS anchoring, and npub resolution. Operators
instantiate this once and delegate all DPYC protocol operations to it.

Supports dual credential templates:
- ``operator_credential_template``: BTCPay + operator-specific secrets,
  delivered once at setup via Secure Courier
- ``patron_credential_template``: per-user API keys delivered via
  Secure Courier (for API-key-based patrons, not OAuth2)

Usage::

    runtime = OperatorRuntime(
        nsec_env_var="TOLLBOOTH_NOSTR_OPERATOR_NSEC",
        tool_costs={"my_tool": ToolTier.READ, "free_tool": ToolTier.FREE},
        operator_credential_template=CredentialTemplate(
            service="my-operator", ...
        ),
        patron_credential_template=CredentialTemplate(
            service="my-patron", ...
        ),
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
        operator_credential_template: Any | None = None,
        patron_credential_template: Any | None = None,
        operator_credential_greeting: str = "",
        patron_credential_greeting: str = "",
        service_name: str = "",
        relays: list[str] | None = None,
        constraint_gate: Any | None = None,
        ots_enabled: bool = False,
        ots_calendars: list[str] | None = None,
    ) -> None:
        self._nsec_env_var = nsec_env_var
        self._tool_costs = tool_costs or {}
        self._operator_credential_template = operator_credential_template
        self._patron_credential_template = patron_credential_template
        self._operator_credential_greeting = operator_credential_greeting
        self._patron_credential_greeting = patron_credential_greeting
        self._service_name = service_name
        self._relays = relays
        self._constraint_gate = constraint_gate
        self._ots_enabled = ots_enabled
        self._ots_calendars = ots_calendars

        # Lazy singletons
        self._vault: Any | None = None
        self._ledger_cache: Any | None = None
        self._courier: Any | None = None
        self._btcpay: Any | None = None
        self._operator_npub: str | None = None
        self._nsec: str | None = None

    @property
    def operator_credential_service(self) -> str:
        """Return the operator credential service name."""
        if self._operator_credential_template:
            return self._operator_credential_template.service
        return ""

    @property
    def patron_credential_service(self) -> str:
        """Return the patron credential service name."""
        if self._patron_credential_template:
            return self._patron_credential_template.service
        return ""

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
        if self._operator_credential_template:
            templates[self._operator_credential_template.service] = self._operator_credential_template
        if self._patron_credential_template:
            templates[self._patron_credential_template.service] = self._patron_credential_template

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

    async def onboarding_status(self) -> dict[str, Any]:
        """Return onboarding status — template-driven, no Settings introspection.

        Checks:
        1. Identity: is nsec set?
        2. Bootstrap: can we reach the vault (Authority-provisioned Neon URL)?
        3. Operator credentials: are all operator_credential_template fields in vault?
        """
        # 1. Identity check
        identity_ok = False
        try:
            self._get_nsec()
            identity_ok = True
        except RuntimeError:
            pass

        # 2. Bootstrap check — actively try to bring up the vault
        vault_ok = self._vault is not None
        bootstrap_error = ""
        if not vault_ok:
            try:
                await self.vault()
                vault_ok = True
            except Exception as exc:
                bootstrap_error = str(exc)

        # 3. Operator credential check — template fields vs vault contents
        configured: list[dict[str, str]] = []
        missing: list[dict[str, str]] = []
        optional_missing: list[dict[str, str]] = []

        if not identity_ok:
            missing.append({
                "field": self._nsec_env_var,
                "category": "identity",
                "status": "missing",
                "how": f"Set {self._nsec_env_var} in the deployment environment.",
            })

        if not vault_ok:
            missing.append({
                "field": "neon_database_url",
                "category": "authority",
                "status": "missing",
                "how": "Auto-provisioned by Authority during registration. "
                       "Call get_operator_config or restart the operator to fetch.",
            })
        else:
            configured.append({
                "field": "neon_database_url",
                "category": "authority",
                "status": "configured",
            })

        # Check operator credential template fields against vault
        if self._operator_credential_template is not None:
            vault_creds = await self._load_vault_creds(
                self._operator_credential_template.service,
            )
            for name, spec in self._operator_credential_template.fields.items():
                if name in vault_creds:
                    configured.append({
                        "field": name, "category": "secret", "status": "configured",
                    })
                else:
                    entry = {
                        "field": name,
                        "category": "secret",
                        "status": "missing",
                        "how": spec.description if spec.description else "Deliver via Secure Courier.",
                    }
                    if spec.required:
                        missing.append(entry)
                    else:
                        optional_missing.append(entry)

        ready = len(missing) == 0
        if ready:
            summary = "Operator is fully configured and ready to serve."
        else:
            secret_missing = [m for m in missing if m["category"] == "secret"]
            parts = []
            if not identity_ok:
                parts.append(f"Set {self._nsec_env_var} to boot")
            if not vault_ok:
                parts.append("bootstrap pending — restart or call get_operator_config")
            if secret_missing:
                names = ", ".join(m["field"] for m in secret_missing)
                parts.append(f"{len(secret_missing)} secret(s) needed via Secure Courier: {names}")
            summary = "Not ready. " + "; ".join(parts) + "."

        result: dict[str, Any] = {
            "ready": ready,
            "configured": configured,
            "missing": missing,
            "optional_missing": optional_missing,
            "summary": summary,
            "bootstrap_error": bootstrap_error,
            "vault_ok": vault_ok,
            "credential_greeting": self._operator_credential_greeting,
            "credential_service": self.operator_credential_service,
            "operator_name": self._service_name,
        }
        return result

    async def patron_onboarding_status(self, patron_npub: str) -> dict[str, Any]:
        """Return patron onboarding status — checks patron_credential_template against vault.

        Only relevant for API-key-based patrons. OAuth2 patrons don't use this.
        """
        if not self._patron_credential_template:
            return {"ready": True, "summary": "No patron credentials required."}

        vault_creds = await self._load_vault_creds(
            self._patron_credential_template.service,
            npub_override=patron_npub,
        )

        configured = []
        missing = []
        for name, spec in self._patron_credential_template.fields.items():
            if name in vault_creds:
                configured.append({
                    "field": name, "category": "patron_secret", "status": "configured",
                })
            else:
                entry = {
                    "field": name,
                    "category": "patron_secret",
                    "status": "missing",
                    "how": spec.description if spec.description else "Deliver via Secure Courier.",
                }
                if spec.required:
                    missing.append(entry)

        ready = len(missing) == 0
        return {
            "ready": ready,
            "configured": configured,
            "missing": missing,
            "summary": "Patron credentials configured." if ready else f"Missing: {', '.join(m['field'] for m in missing)}",
            "credential_greeting": self._patron_credential_greeting,
            "credential_service": self.patron_credential_service,
        }

    async def _load_vault_creds(
        self,
        service: str,
        npub_override: str | None = None,
    ) -> dict[str, str]:
        """Load credentials from vault for a given service."""
        try:
            from tollbooth.tools.onboarding import load_vault_credentials
            courier = await self.courier()
            npub = npub_override or self.operator_npub()
            return await load_vault_credentials(courier, service, npub) or {}
        except Exception:
            return {}

    async def load_credentials(
        self,
        field_names: list[str],
        *,
        service: str | None = None,
    ) -> dict[str, str]:
        """Load specific credentials from the Secure Courier vault.

        Args:
            field_names: Which fields to load.
            service: Credential service name. Defaults to operator credential service.
        """
        from tollbooth.tools.onboarding import load_config_from_vault
        svc = service or self.operator_credential_service
        return await load_config_from_vault(
            await self.courier(),
            svc,
            self.operator_npub(),
            field_names,
        )

    # ------------------------------------------------------------------
    # BTCPay client (from operator credential vault)
    # ------------------------------------------------------------------

    async def ensure_btcpay(self) -> Any:
        """Return a BTCPayClient constructed from vault credentials.

        Loads btcpay_host, btcpay_api_key, btcpay_store_id from the
        operator credential vault.  Cached after first successful load.
        """
        if self._btcpay is not None:
            return self._btcpay

        from tollbooth.btcpay_client import BTCPayClient

        creds = await self.load_credentials(
            ["btcpay_host", "btcpay_api_key", "btcpay_store_id"],
        )
        host = creds.get("btcpay_host")
        api_key = creds.get("btcpay_api_key")
        store_id = creds.get("btcpay_store_id")

        if not all([host, api_key, store_id]):
            raise ValueError(
                "BTCPay not configured. Deliver btcpay_host, btcpay_api_key, "
                "btcpay_store_id via Secure Courier (request_credential_channel)."
            )

        self._btcpay = BTCPayClient(
            host=host, api_key=api_key, store_id=store_id,
        )
        return self._btcpay

    # ------------------------------------------------------------------
    # Horizon auth helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_current_user_id() -> str | None:
        """Extract FastMCP Cloud user ID from request headers.

        Returns None in STDIO mode or when no auth headers present.
        """
        try:
            from fastmcp.server.dependencies import get_http_headers
            headers = get_http_headers(include_all=True)
            return headers.get("fastmcp-cloud-user")
        except Exception:
            return None

    @staticmethod
    def require_user_id() -> str:
        """Extract FastMCP Cloud user ID or raise ValueError."""
        user_id = OperatorRuntime.get_current_user_id()
        if not user_id:
            raise ValueError(
                "Multi-tenant mode requires Horizon authentication. "
                "Connect via the operator's MCP endpoint URL."
            )
        return user_id

    # ------------------------------------------------------------------
    # Low-balance warning injection
    # ------------------------------------------------------------------

    async def inject_low_balance_warning(
        self,
        result: dict[str, Any],
        npub: str,
        seed_balance_sats: int = 0,
    ) -> dict[str, Any]:
        """Append low_balance_warning to result if balance is running low.

        Call this at the end of every paid tool to proactively warn patrons.
        No-op if npub is invalid or cache is unavailable.
        """
        try:
            from tollbooth.tools.credits import compute_low_balance_warning
            npub = resolve_npub(npub)
            cache = await self.ledger_cache()
            warning = compute_low_balance_warning(
                await cache.get(npub), seed_balance_sats,
            )
            if warning:
                result["low_balance_warning"] = warning
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Demand tracking (for surge pricing constraints)
    # ------------------------------------------------------------------

    def _demand_window_key(self) -> str:
        """Current hourly demand window key."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")

    async def get_global_demand(self, tool_name: str) -> dict[str, int]:
        """Get current hourly demand count for a tool (for surge pricing)."""
        try:
            vault = await self.vault()
            count = await vault.get_demand(tool_name, self._demand_window_key())
            return {tool_name: count}
        except Exception:
            return {}

    def fire_and_forget_demand_increment(self, tool_name: str) -> None:
        """Increment demand counter for a tool (non-blocking)."""
        import asyncio

        async def _increment() -> None:
            try:
                vault = await self.vault()
                await vault.increment_demand(
                    tool_name, self._demand_window_key(),
                )
            except Exception:
                pass

        asyncio.create_task(_increment())


# ======================================================================
# Standard tool registration
# ======================================================================


def register_standard_tools(
    mcp: Any,
    slug: str,
    rt: OperatorRuntime,
    *,
    service_name: str = "",
    service_version: str = "",
) -> None:
    """Register all standard DPYC tools on a FastMCP app.

    Call once at module level.  Operators only write domain-specific tools.

    Args:
        mcp: The FastMCP app instance.
        slug: Tool name prefix (e.g., ``"weather"``).
        rt: The OperatorRuntime instance.
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
        courier_ok = (rt._courier is not None
                      and hasattr(rt._courier, '_exchange')
                      and rt._courier._exchange._credential_vault is not None)

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
        return await rt.onboarding_status()

    # -- Secure Courier ------------------------------------------------

    @tool
    async def session_status() -> dict[str, Any]:
        """Check session state — shows whether credentials are active
        or onboarding is needed. Free."""
        return {
            "success": True,
            "operator_npub": rt.operator_npub(),
            "operator_credential_service": rt.operator_credential_service,
            "patron_credential_service": rt.patron_credential_service,
            "courier_configured": rt._courier is not None,
        }

    @tool
    async def request_credential_channel(
        sender_npub: str = "",
        service: str = "",
    ) -> dict[str, Any]:
        """Open a Secure Courier channel for operator credential delivery.

        Sends a welcome DM with a credential template to the provided npub.
        Free.
        """
        if not sender_npub:
            sender_npub = rt.operator_npub()
        if not service:
            service = rt.operator_credential_service
        courier = await rt.courier()
        if courier is None:
            return {"success": False, "error": "Secure Courier not configured."}
        try:
            return await courier.open_channel(
                service,
                greeting=rt._operator_credential_greeting,
                recipient_npub=sender_npub,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    async def receive_credentials(
        sender_npub: str = "",
        service: str = "",
        credential_card: str = "",
    ) -> dict[str, Any]:
        """Pick up operator credentials from the Secure Courier.

        Checks the vault first (instant), then polls Nostr relays for
        encrypted DMs. If a credential_card (ncred1...) is provided,
        redeems it directly without relay polling. Free.
        """
        if not sender_npub:
            sender_npub = rt.operator_npub()
        if not service:
            service = rt.operator_credential_service
        courier = await rt.courier()
        if courier is None:
            return {"success": False, "error": "Secure Courier not configured."}
        try:
            if credential_card:
                return await courier._exchange.redeem_credential_card(
                    credential_card, service,
                )
            return await courier.receive(sender_npub, service)
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    async def forget_credentials(service: str = "") -> dict[str, Any]:
        """Delete vaulted credentials for key rotation. Free."""
        if not service:
            service = rt.operator_credential_service
        npub = rt.operator_npub()
        courier = await rt.courier()
        if courier is None:
            return {"success": False, "error": "Secure Courier not configured."}
        try:
            return await courier.forget(npub, service)
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -- Patron credential tools (only if patron template is set) ------

    if rt._patron_credential_template is not None:
        @tool
        async def request_patron_credentials(
            sender_npub: str = "",
        ) -> dict[str, Any]:
            """Open a Secure Courier channel for patron credential delivery.

            Sends a welcome DM with a credential template to the patron.
            Free.
            """
            if not sender_npub:
                return {"success": False, "error": "sender_npub is required."}
            courier = await rt.courier()
            if courier is None:
                return {"success": False, "error": "Secure Courier not configured."}
            try:
                return await courier.open_channel(
                    rt.patron_credential_service,
                    greeting=rt._patron_credential_greeting,
                    recipient_npub=sender_npub,
                )
            except Exception as e:
                return {"success": False, "error": str(e)}

        @tool
        async def receive_patron_credentials(
            sender_npub: str = "",
            credential_card: str = "",
        ) -> dict[str, Any]:
            """Pick up patron credentials from the Secure Courier.

            Checks the vault first, then polls Nostr relays.
            Free.
            """
            if not sender_npub:
                return {"success": False, "error": "sender_npub is required."}
            courier = await rt.courier()
            if courier is None:
                return {"success": False, "error": "Secure Courier not configured."}
            try:
                service = rt.patron_credential_service
                if credential_card:
                    return await courier._exchange.redeem_credential_card(
                        credential_card, service,
                    )
                return await courier.receive(sender_npub, service)
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

    # -- Constraint Engine tools ---------------------------------------

    @tool
    async def check_price(tool_name: str, npub: str = "") -> dict[str, Any]:
        """Preview the effective cost of a tool call.

        Shows the base cost and any constraint effects (discounts, free
        trials, surge pricing). Free — no credits required.
        """
        base_cost = rt._tool_costs.get(tool_name)
        if base_cost is None:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}. "
                f"Valid: {list(rt._tool_costs.keys())}",
            }

        result: dict[str, Any] = {
            "success": True,
            "tool_name": tool_name,
            "base_cost_api_sats": int(base_cost),
            "effective_cost_api_sats": int(base_cost),
            "constraints_enabled": False,
            "constraint_effects": [],
        }

        gate = rt._constraint_gate
        if gate and gate.enabled and base_cost > 0:
            result["constraints_enabled"] = True
            try:
                resolved = resolve_npub(npub)
                cache = await rt.ledger_cache()
                ledger = await cache.get(resolved)
                demand = await rt.get_global_demand(tool_name)
                denial, effective = gate.check(
                    tool_name=tool_name,
                    base_cost=int(base_cost),
                    ledger=ledger,
                    npub=resolved,
                    global_demand=demand,
                )
                if demand.get(tool_name, 0) > 0:
                    result["current_demand"] = demand[tool_name]
                if denial:
                    result["effective_cost_api_sats"] = 0
                    result["constraint_effects"].append({
                        "type": "denied",
                        "reason": denial.get("constraint_reason", "blocked"),
                    })
                else:
                    result["effective_cost_api_sats"] = effective
                    if effective != base_cost:
                        result["constraint_effects"].append({
                            "type": "discount",
                            "from": int(base_cost),
                            "to": effective,
                        })
            except ValueError:
                result["constraint_effects"].append({
                    "type": "info",
                    "message": "npub required for constraint evaluation.",
                })

        return result

    @tool
    async def list_constraint_types() -> dict[str, Any]:
        """List all available constraint types and their parameter schemas.

        Returns the type, category, description, and parameter specs for
        every constraint that can be used in a pricing pipeline.
        Free — no credits required.
        """
        from tollbooth.tools.pricing import (
            list_constraint_types as _list,
        )
        return {"status": "ok", "constraint_types": _list()}


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
