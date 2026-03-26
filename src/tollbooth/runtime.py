"""OperatorRuntime — the core DPYC protocol engine for all operators.

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
