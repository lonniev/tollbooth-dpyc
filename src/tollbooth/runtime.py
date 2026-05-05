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
        tool_registry={"my_tool": ToolIdentity(capability="my_tool", category="read", intent="...")},
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
    result = await runtime.debit_or_deny(my_tool_uuid, npub, proof=proof)
    if isinstance(result, dict):
        return result  # denial
    cost = result  # int — the computed cost
    # ... do work ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import signal
from typing import Any, Callable

logger = logging.getLogger(__name__)


def resolve_npub(npub: str) -> str:
    """Validate and return the npub. No fallback, no session cache.

    Validates prefix, length, and bech32 decode. Raises ValueError
    if npub is missing or malformed.
    """
    if not npub or not npub.startswith("npub1") or len(npub) < 60:
        raise ValueError(
            "npub is required. Pass your Nostr public key (npub1...) "
            "to identify yourself."
        )
    try:
        from pynostr.key import PublicKey
        PublicKey.from_npub(npub)
    except Exception:
        raise ValueError(f"Invalid npub: bech32 decode failed for {npub[:20]}...")
    return npub


class OperatorRuntime:
    """Core DPYC protocol engine shared by all operators.

    Handles:
    - Bootstrap: nsec → Authority → Neon URL → vault → cache
    - Credit gating: debit_or_deny, rollback
    - Secure Courier: persistent credential vault
    - Identity: resolve_npub (no OAuth coupling)
    """

    def __init__(
        self,
        *,
        nsec_env_var: str = "TOLLBOOTH_NOSTR_OPERATOR_NSEC",
        tool_registry: dict[str, "ToolIdentity"] | None = None,
        operator_credential_template: Any | None = None,
        patron_credential_template: Any | None = None,
        operator_credential_greeting: str = "",
        patron_credential_greeting: str = "",
        service_name: str = "",
        relays: list[str] | None = None,
        constraint_gate: Any | None = None,
        ots_enabled: bool = True,
        ots_calendars: list[str] | None = None,
        on_forget: Any | None = None,
        operator_settings: dict[str, Any] | None = None,
        purchase_mode: str = "certified",
        credential_validator: Any | None = None,
        proven_npub_ttl_seconds: int = 3600,
        npub_proof_field: str = "confirm",
        npub_proof_greeting: str = "",
        on_npub_proven: Any | None = None,
        oauth_provider: Any | None = None,
    ) -> None:
        self._nsec_env_var = nsec_env_var
        self._credential_validator = credential_validator
        from tollbooth.tool_identity import ToolIdentity  # noqa: F811
        # Registry keyed by UUID — the sole economic key.
        # Exclude OTS tools when notarization is disabled so they
        # don't appear in the pricing model as stale entries.
        _OTS_CAPABILITIES = {"notarize_ledger", "get_notarization_proof", "list_notarizations"}
        registry = tool_registry or {}
        if not ots_enabled:
            registry = {k: v for k, v in registry.items() if v.capability not in _OTS_CAPABILITIES}
        self._tool_registry: dict[str, ToolIdentity] = registry
        self._slug: str = ""  # set by register_standard_tools
        self._tool_func_names: dict[str, str] = {}  # UUID → Python function name, populated by paid_tool
        self._mcp_name_cache: dict[str, str] = {}  # UUID → resolved MCP name, built lazily
        self._pricing_resolver: Any | None = None  # lazily created after vault
        self._operator_credential_template = operator_credential_template
        self._patron_credential_template = patron_credential_template
        self._operator_credential_greeting = operator_credential_greeting
        self._patron_credential_greeting = patron_credential_greeting
        self._service_name = service_name
        self._relays = relays
        self._constraint_gate = constraint_gate
        self._ots_enabled = ots_enabled
        self._ots_calendars = ots_calendars
        self._on_forget = on_forget  # callback(service, npub) on credential forget
        self._operator_settings: dict[str, Any] = operator_settings or {}
        self._purchase_mode = purchase_mode  # "certified" or "direct"

        # Lazy singletons
        self._vault: Any | None = None
        self._vault_ready_at: float = 0.0  # time.monotonic() when vault first became ready
        self._ledger_cache: Any | None = None
        self._courier: Any | None = None
        self._cashier: Any | None = None
        self._operator_npub: str | None = None
        self._nsec: str | None = None
        self._reconciled_npubs: set[str] = set()  # dedup auto-reconciliation
        self._proven_npub_ttl = proven_npub_ttl_seconds
        self._proven_npub_cache: Any | None = None  # lazy ProvenNpubCache
        self._npub_proof_field = npub_proof_field
        self._npub_proof_greeting = npub_proof_greeting
        self._on_npub_proven = on_npub_proven  # async callback(npub, payload)
        self._oauth_provider = oauth_provider  # OAuthProviderConfig or None
        # Shutdown state
        self._shutdown_triggered: bool = False
        self._shutdown_handlers_registered: bool = False
        self._cleanup_callbacks: list[Callable[[], Any]] = []

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

    @property
    def operator_settings(self) -> dict[str, Any]:
        """Operator-supplied settings dict.

        Operators pass whatever config they need at OperatorRuntime init
        time and access it here — no global singleton required.
        """
        return self._operator_settings

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
    # Shutdown
    # ------------------------------------------------------------------

    def add_cleanup_callback(self, callback: Callable[[], Any]) -> None:
        """Register an async or sync cleanup callback for graceful shutdown.

        Callbacks run in order during ``graceful_shutdown()`` after the
        ledger cache is flushed and the vault is closed.  Each callback
        is called at most once per shutdown.
        """
        self._cleanup_callbacks.append(callback)

    def register_shutdown_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers that trigger graceful shutdown.

        Idempotent — safe to call multiple times; only the first call
        installs signal handlers.
        """
        if self._shutdown_handlers_registered:
            return
        self._shutdown_handlers_registered = True
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.ensure_future(self.graceful_shutdown()),
                )
        except (RuntimeError, NotImplementedError):
            pass

    async def graceful_shutdown(self) -> None:
        """Flush ledger cache, close vault, run cleanup callbacks.

        Idempotent — only the first invocation performs work.
        """
        if self._shutdown_triggered:
            return
        self._shutdown_triggered = True

        # 1. Flush and stop ledger cache
        if self._ledger_cache is not None:
            dirty = self._ledger_cache.dirty_count
            logger.info("Graceful shutdown: flushing %d dirty entries...", dirty)
            try:
                await asyncio.wait_for(
                    self._shutdown_flush_ledger(), timeout=8.0,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Graceful shutdown timed out after 8s — "
                    "some entries may be lost."
                )
            self._ledger_cache = None

        # 2. Close vault
        if self._vault is not None:
            closer = getattr(self._vault, "close", None)
            if closer is not None:
                await closer()
            self._vault = None

        # 3. Run operator-registered cleanup callbacks
        for cb in self._cleanup_callbacks:
            try:
                result = cb()
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result
            except Exception as exc:
                logger.warning("Shutdown cleanup callback failed: %s", exc)
        self._cleanup_callbacks.clear()

    async def _shutdown_flush_ledger(self) -> None:
        """Flush and stop the ledger cache (extracted for wait_for wrapping)."""
        assert self._ledger_cache is not None
        flushed = await self._ledger_cache.flush_all()
        await self._ledger_cache.stop()
        logger.info("Shutdown: flushed %d entries.", flushed)

    # ------------------------------------------------------------------
    # Bootstrap & Vault
    # ------------------------------------------------------------------

    async def vault(self) -> Any:
        """Return the NeonVault, bootstrapping from Authority if needed.

        Trust-root operators (purchase_mode="direct") read NEON_DATABASE_URL
        from the environment — they have no upstream Authority to bootstrap
        from.  All other operators discover their Neon URL from a Nostr
        relay DM signed by their Authority.
        """
        if self._vault is not None:
            return self._vault

        import os
        import time as _time
        from tollbooth.vaults import NeonVault

        if self._purchase_mode == "direct":
            # Trust root: read Neon URL from env (set by deploy platform).
            neon_url = os.environ.get("NEON_DATABASE_URL", "")
            if not neon_url:
                raise ValueError(
                    "NEON_DATABASE_URL is required for trust-root operators "
                    "(purchase_mode='direct')."
                )
            self._vault = NeonVault(database_url=neon_url)
            await self._vault.ensure_schema()
            self._vault_ready_at = _time.monotonic()
            logger.info("Vault initialized from NEON_DATABASE_URL (trust root)")
            return self._vault

        # Certified operators: bootstrap from Authority relay DM.
        from tollbooth.bootstrap import ensure_bootstrapped
        result = await ensure_bootstrapped(relays=self._relays)
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
        self._vault_ready_at = _time.monotonic()
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
            # Retry vault attachment if courier was created without one
            if (hasattr(self._courier, '_exchange')
                    and self._courier._exchange._credential_vault is None
                    and self._vault is not None):
                try:
                    from tollbooth.vaults.neon import NeonCredentialVault
                    cv = NeonCredentialVault(neon_vault=self._vault)
                    await cv.ensure_schema()
                    self._courier._exchange._credential_vault = cv
                    logger.info("Attached credential vault to courier (late init)")
                except Exception as exc:
                    logger.warning("Credential vault late-attach failed: %s", exc)
            elif (hasattr(self._courier, '_exchange')
                    and self._courier._exchange._credential_vault is None):
                logger.debug(
                    "Courier has no credential vault; vault ready=%s",
                    self._vault is not None,
                )
            return self._courier

        nsec = self._get_nsec()
        if not nsec:
            return None

        try:
            from tollbooth.nostr_diagnostics import resolve_relays
            from tollbooth.secure_courier import SecureCourierService
        except ImportError:
            return None

        relays = resolve_relays(self._relays)

        # Build credential vault from bootstrapped Neon
        credential_vault = None
        try:
            v = await self.vault()
            from tollbooth.vaults.neon import NeonCredentialVault
            credential_vault = NeonCredentialVault(neon_vault=v)
            await credential_vault.ensure_schema()
            logger.info("Credential vault initialized (NeonCredentialVault)")
        except Exception as exc:
            logger.warning("No persistent credential vault (%s): %s", type(exc).__name__, exc)

        templates = {}
        if self._operator_credential_template:
            templates[self._operator_credential_template.service] = self._operator_credential_template
        if self._patron_credential_template:
            templates[self._patron_credential_template.service] = self._patron_credential_template

        # Built-in template for npub ownership proof exchange.
        # The DM itself proves npub ownership (signed by the patron's nsec).
        # The payload field is operator-configurable (e.g., "passphrase" for
        # taxsort, "confirm" for schwab) — the value is immaterial.
        from tollbooth.credential_templates import CredentialTemplate, FieldSpec
        _proof_field = self._npub_proof_field
        templates["npub_ownership"] = CredentialTemplate(
            service="npub_ownership",
            version=1,
            fields={
                _proof_field: FieldSpec(
                    required=True, sensitive=False,
                    description="Any text — your signed DM is the proof",
                ),
                "cache_duration": FieldSpec(
                    required=False, sensitive=False, default="2h",
                    description="How long to cache this proof (e.g. 2h, 1d, 3 days, unlimited)",
                ),
            },
            description="Npub ownership — the signed DM itself proves you own this npub",
        )

        self._courier = SecureCourierService(
            operator_nsec=nsec,
            relays=relays,
            credential_vault=credential_vault,
            templates=templates,
        )
        return self._courier

    def mcp_name_for(self, tool_id: str) -> str:
        """Resolve the full MCP tool name for a tool UUID.

        Uses _tool_func_names (recorded by paid_tool at decoration time)
        to compute {slug}_{function_name}. Falls back to {slug}_{capability}
        for standard tools where function names match capabilities.

        Results are cached after first resolution.
        """
        if tool_id in self._mcp_name_cache:
            return self._mcp_name_cache[tool_id]

        identity = self._tool_registry.get(tool_id)
        if identity is None:
            return tool_id

        # Explicit mcp_name on the identity takes precedence
        if identity.mcp_name:
            self._mcp_name_cache[tool_id] = identity.mcp_name
            return identity.mcp_name

        # Resolve from function name recording (populated by paid_tool)
        func_name = self._tool_func_names.get(tool_id)
        if func_name and self._slug:
            name = f"{self._slug}_{func_name}"
        elif self._slug:
            name = f"{self._slug}_{identity.capability}"
        else:
            name = identity.capability

        self._mcp_name_cache[tool_id] = name
        return name

    # ------------------------------------------------------------------
    # Proven npub ownership cache
    # ------------------------------------------------------------------

    async def proven_npub_cache(self) -> Any:
        """Lazy accessor for the ProvenNpubCache (vault-backed)."""
        if self._proven_npub_cache is None:
            from tollbooth.proven_npub import ProvenNpubCache
            v = await self.vault()
            self._proven_npub_cache = ProvenNpubCache(
                ttl_seconds=self._proven_npub_ttl,
                vault=v,
            )
        return self._proven_npub_cache

    @staticmethod
    def _get_session_id() -> str:
        """Get the MCP transport session ID from FastMCP context."""
        from fastmcp.server.dependencies import get_context
        return get_context().session_id

    # ------------------------------------------------------------------
    # Credit Gating
    # ------------------------------------------------------------------

    async def pricing_resolver(self) -> Any:
        """Lazy accessor for the PricingResolver (requires vault)."""
        if self._pricing_resolver is None:
            vault = await self.vault()
            from tollbooth.pricing_store import PricingModelStore
            from tollbooth.pricing_resolver import PricingResolver
            store = PricingModelStore(neon_vault=vault)
            self._pricing_resolver = PricingResolver(
                store=store,
                operator=self.operator_npub(),
            )
        return self._pricing_resolver

    async def debit_or_deny(
        self,
        tool_id: str,
        npub: str,
        *,
        proof: str = "",
        tool_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | int:
        """Gate a tool call: identity → proof → access → pricing → constraints → billing.

        Returns the computed cost (int, 0 for free) to proceed.
        Returns an error dict to deny.

        Every tool that takes an npub requires a proof (Schnorr-signed
        kind-27235 event proving npub ownership). No proof, no service.
        """
        from tollbooth.constants import ErrorCode
        from tollbooth.identity_proof import verify_proof

        identity = self._tool_registry.get(tool_id)

        # ── Identity ──────────────────────────────────────────
        if identity is None:
            return {
                "success": False,
                "error_code": ErrorCode.TOOL_NOT_REGISTERED,
                "error": f"Unknown tool '{tool_id}' — not registered with this operator.",
            }

        name = self.mcp_name_for(tool_id)        # full MCP name for display/logging
        cap = identity.capability              # short name for proof verification
        category = identity.category           # set in code, never by the pricing model

        # ── Proof verification ────────────────────────────────
        if npub:
            try:
                resolved = resolve_npub(npub)
            except ValueError as e:
                return {
                    "success": False,
                    "error_code": ErrorCode.NPUB_INVALID,
                    "error": str(e),
                }

            if category == "restricted":
                # Restricted tools always require inline proof (operator identity)
                if not proof:
                    return {
                        "success": False,
                        "error_code": ErrorCode.PROOF_REQUIRED,
                        "error": "proof is required. Sign a kind-27235 Nostr event with your nsec.",
                    }
                verify_against = self.operator_npub()
                if not verify_proof(proof, verify_against, cap):
                    return {
                        "success": False,
                        "error_code": ErrorCode.PROOF_INVALID,
                        "error": "Invalid proof — Schnorr signature does not match npub.",
                    }
            else:
                # Non-restricted: check proven npub cache before requiring inline proof.
                # Proof is poison-keyed — the calling application holds the raw
                # poison phrase and supplies it as the ``proof`` parameter.
                # The MCP hashes it and looks up the proof record in Neon.
                import hashlib as _hashlib
                import re as _re
                cache = await self.proven_npub_cache()
                _POISON_RE = _re.compile(r"^[a-z]+-[a-z]+-\d+$")
                if proof and _POISON_RE.match(proof):
                    # Proof looks like a poison phrase — hash and look up
                    poison_hash = _hashlib.sha256(proof.encode()).hexdigest()
                    if await cache.is_proven(poison_hash, resolved):
                        pass  # npub ownership verified via poison-keyed proof
                    else:
                        return {
                            "success": False,
                            "error_code": ErrorCode.PROOF_REFRESH_NEEDED,
                            "error": (
                                "Your npub-proof cache entry is no longer valid. "
                                "This is routine — sign a fresh DM challenge and you're back."
                            ),
                            "next_steps": [
                                "request_npub_proof(patron_npub=<patron_npub>)",
                                "Sign the DM challenge from your Nostr client",
                                "receive_npub_proof(patron_npub=<patron_npub>) to cache the fresh proof_token",
                            ],
                        }
                elif proof:
                    # Proof looks like a Schnorr signature — verify inline
                    if not verify_proof(proof, resolved, cap):
                        return {
                            "success": False,
                            "error_code": ErrorCode.PROOF_INVALID,
                            "error": "Invalid proof — Schnorr signature does not match npub.",
                        }
                else:
                    return {
                        "success": False,
                        "error_code": ErrorCode.PROOF_REQUIRED,
                        "error": (
                            "proof is required. Establish npub ownership before calling paid tools."
                        ),
                        "next_steps": [
                            "request_npub_proof(patron_npub=<patron_npub>)",
                            "Sign the DM challenge from your Nostr client",
                            "receive_npub_proof(patron_npub=<patron_npub>) — remember the returned proof_token",
                            "Pass proof_token as the proof parameter on every subsequent paid tool call",
                        ],
                    }

        # ── Access: operator-restricted ───────────────────────
        # Proof already verified above. Just check the npub is the operator.
        if category == "restricted":
            try:
                caller = resolve_npub(npub)
            except ValueError as e:
                return {
                    "success": False,
                    "error_code": ErrorCode.NPUB_INVALID,
                    "error": str(e),
                }
            if caller == self.operator_npub():
                return 0
            return {
                "success": False,
                "error_code": ErrorCode.RESTRICTED,
                "error": "This tool is restricted to the operator.",
            }

        # ── Pricing ───────────────────────────────────────────
        # Code says "free" → cost is 0, Neon not consulted.
        # Everything else requires Neon, a model, and an explicit price.
        cost = 0
        if category != "free":
            resolver = await self.pricing_resolver()
            await resolver._ensure_fresh()
            if not resolver.neon_available:
                return {
                    "success": False,
                    "error_code": ErrorCode.WARMING_UP,
                    "error": (
                        "Service is warming up — the pricing model could not be "
                        "loaded from the database yet. This typically resolves shortly "
                        "after a cold start. Retry your request. "
                        "Free tools (check_balance, check_payment, proof exchange) "
                        "remain available during warm-up."
                    ),
                    "next_steps": ["Retry the same call shortly"],
                }

            if not await resolver.has_tool(tool_id):
                return {
                    "success": False,
                    "error_code": ErrorCode.TOOL_NOT_PRICED,
                    "error": (
                        f"Tool '{name}' is not yet in the pricing model. "
                        f"Add it to the pricing model before use."
                    ),
                }
            if not await resolver.is_priced(tool_id):
                return {
                    "success": False,
                    "error_code": ErrorCode.TOOL_NOT_PRICED,
                    "error": (
                        f"Tool '{name}' has not been priced yet (TBD). "
                        f"Set a price in the pricing model before use."
                    ),
                }
            pricing = await resolver.get_tool_pricing(tool_id)
            cost = pricing.compute(**(tool_kwargs or {}))

        # ── Resolve caller ────────────────────────────────────
        # Paid tools always require an npub.  Free tools work
        # without one — but constraints won't evaluate.
        try:
            npub = resolve_npub(npub)
        except ValueError:
            if category != "free":
                return {
                    "success": False,
                    "error_code": ErrorCode.NPUB_INVALID,
                    "error": (
                        "npub is required. Pass your Nostr public key (npub1...) "
                        "to identify yourself."
                    ),
                }
            npub = ""

        # ── Constraints ───────────────────────────────────────
        # The pipeline applies to every non-restricted tool.
        # Constraints can modify the effective cost (discounts,
        # free, surge) or deny access (temporal windows, rate
        # limits). Without an npub, constraints cannot evaluate.
        effective_cost = cost
        if npub and category != "free":
            try:
                resolver = await self.pricing_resolver()
                engine = resolver._cached_engine
                if engine is not None:
                    from tollbooth.constraints.base import (
                        ConstraintContext, LedgerSnapshot, PatronIdentity, EnvironmentSnapshot,
                    )
                    cache = await self.ledger_cache()
                    ledger = await cache.get(npub)
                    demand = await self.get_global_demand(name)
                    context = ConstraintContext(
                        ledger=LedgerSnapshot(
                            balance_api_sats=ledger.balance_api_sats,
                            total_deposited_api_sats=ledger.total_deposited_api_sats,
                            total_consumed_api_sats=ledger.total_consumed_api_sats,
                            total_expired_api_sats=ledger.total_expired_api_sats,
                        ),
                        patron=PatronIdentity(npub=npub),
                        env=EnvironmentSnapshot(
                            utc_now=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                            tool_name=name,
                            global_demand=tuple(demand.items()),
                        ),
                        proof=proof,
                    )
                    result = engine.evaluate(name, context)
                    if not result.allowed:
                        error: dict[str, Any] = {
                            "success": False,
                            "error_code": ErrorCode.CONSTRAINT_DENIED,
                            "error": result.message or f"Constraint denied: {result.reason}",
                            "constraint_reason": result.reason,
                        }
                        if result.retry_after:
                            error["retry_after"] = result.retry_after.isoformat()
                        return error
                    if result.price_modifier:
                        effective_cost = result.price_modifier.apply_to(cost)
            except Exception as exc:
                logger.warning("Constraint evaluation failed for %s: %s", name, exc)
                # Fall through with base cost — don't block the call

        # ── No charge ─────────────────────────────────────────
        if effective_cost == 0:
            if npub:
                cache = await self.ledger_cache()
                ledger = await cache.get(npub)
                ledger.debit(name, 0)
                cache.mark_dirty(npub)
            return 0

        # ── Billing ───────────────────────────────────────────
        cache = await self.ledger_cache()
        ledger = await cache.get(npub)

        # Apply tranche lifetime retroactively to any unstamped tranches
        ttl = await self.resolve_tranche_lifetime()
        ledger.apply_tranche_lifetime(ttl)

        if npub not in self._reconciled_npubs and ledger.pending_invoices:
            self._reconciled_npubs.add(npub)
            try:
                cashier = await self.ensure_cashier()
                from tollbooth.tools.credits import reconcile_pending_invoices
                recon = await reconcile_pending_invoices(
                    cashier, cache, npub,
                    tranche_lifetime_seconds=ttl,
                )
                if recon.get("reconciled", 0) > 0:
                    logger.info(
                        "Auto-reconciled %d invoice(s) for %s on cold start",
                        recon["reconciled"], npub[:20],
                    )
                    ledger = await cache.get(npub)
            except Exception as exc:
                logger.debug("Auto-reconciliation skipped for %s: %s", npub[:20], exc)
        elif npub not in self._reconciled_npubs:
            self._reconciled_npubs.add(npub)

        if ledger.balance_api_sats < effective_cost:
            return {
                "success": False,
                "error_code": ErrorCode.INSUFFICIENT_BALANCE,
                "error": (
                    f"Insufficient balance: {ledger.balance_api_sats} sats "
                    f"available, {effective_cost} required for {name}."
                ),
                "balance_sats": ledger.balance_api_sats,
                "cost_sats": effective_cost,
                "next_steps": [
                    f"purchase_credits(amount_sats=<at least {effective_cost - ledger.balance_api_sats}>, npub=<patron_npub>)",
                    "Pay the returned Lightning invoice",
                    "check_payment(invoice_id=<from purchase_credits>) to confirm settlement",
                ],
            }

        ledger.debit(name, effective_cost)
        cache.mark_dirty(npub)
        return effective_cost

    async def rollback_debit(
        self, tool_id: str, npub: str,
        *, tool_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Rollback a debit after a tool execution failure."""
        try:
            npub = resolve_npub(npub)
            cache = await self.ledger_cache()
            identity = self._tool_registry.get(tool_id)
            if identity is None or identity.category in ("free", "restricted"):
                return
            resolver = await self.pricing_resolver()
            pricing = await resolver.get_tool_pricing(tool_id)
            cost = pricing.compute(**(tool_kwargs or {}))
            if cost > 0:
                ledger = await cache.get(npub)
                ledger.credit_deposit(cost, f"rollback:{self.mcp_name_for(tool_id)}")
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
        # Ensure courier has its credential vault (triggers late-attach on cold start)
        try:
            await self.courier()
        except Exception:
            pass

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
                        "lifecycle": spec.lifecycle,
                    })
                else:
                    entry = {
                        "field": name,
                        "category": "secret",
                        "status": "missing",
                        "lifecycle": spec.lifecycle,
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
            return {
                "ready": True,
                "configured": [],
                "missing": [],
                "summary": "No patron credentials required — this service uses automatic authentication.",
                "credential_type": "none_or_dynamic",
                "credential_service": "",
            }

        # Ensure courier has its credential vault (triggers late-attach on cold start)
        try:
            await self.courier()
        except Exception:
            pass

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
                    "lifecycle": spec.lifecycle,
                })
            else:
                entry = {
                    "field": name,
                    "category": "patron_secret",
                    "status": "missing",
                    "lifecycle": spec.lifecycle,
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
            "credential_type": "set_once",
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
            if courier is None:
                logger.debug("No courier available for credential load (service=%s)", service)
                return {}
            has_vault = (hasattr(courier, '_exchange')
                         and courier._exchange._credential_vault is not None)
            if not has_vault:
                logger.warning(
                    "Courier has no credential vault — cannot load credentials "
                    "(service=%s, npub=%s)",
                    service, (npub_override or "operator")[:20],
                )
                return {}
            npub = npub_override or self.operator_npub()
            result = await load_vault_credentials(courier, service, npub)
            if result:
                logger.info("Loaded %d credential fields for %s (service=%s)",
                            len(result), npub[:20], service)
            else:
                logger.info("No credentials found for %s (service=%s)", npub[:20], service)
            return result or {}
        except Exception as exc:
            logger.warning("Credential vault load failed (%s): %s", type(exc).__name__, exc)
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
    # Patron session persistence (general-purpose)
    # ------------------------------------------------------------------

    async def store_patron_session(
        self,
        patron_npub: str,
        credentials: dict[str, str],
        *,
        service: str | None = None,
    ) -> bool:
        """Persist patron session credentials to the encrypted vault.

        Call this after any successful patron authentication (OAuth,
        API key delivery, Secure Courier, etc.). The credentials are
        encrypted with the operator's key and stored in Neon, keyed
        by (service, patron_npub). Survives process restarts.

        Args:
            patron_npub: The patron's npub (vault key).
            credentials: Dict of credential fields to store (e.g.,
                ``{"token_json": "...", "account_hash": "..."}``).
            service: Credential service name. Defaults to
                patron_credential_service.

        Returns True on success, False on failure.
        """
        svc = service or self.patron_credential_service
        if not svc:
            logger.warning("No patron credential service configured.")
            return False
        try:
            courier = await self.courier()
            if courier is None or courier._exchange._credential_vault is None:
                logger.warning("No credential vault for patron session storage.")
                return False
            await courier._exchange._vault_store(svc, patron_npub, credentials)
            return True
        except Exception as exc:
            logger.warning("Patron session store failed: %s", exc)
            return False

    async def load_patron_session(
        self,
        patron_npub: str,
        *,
        service: str | None = None,
    ) -> dict[str, str] | None:
        """Restore patron session credentials from the encrypted vault.

        Call this on session miss (e.g., after process restart) before
        returning "no active session." If credentials exist in the vault,
        returns them as a dict. Returns None if not found.

        Args:
            patron_npub: The patron's npub (vault key).
            service: Credential service name. Defaults to
                patron_credential_service.
        """
        svc = service or self.patron_credential_service
        if not svc:
            return None
        return await self._load_vault_creds(svc, npub_override=patron_npub) or None

    # ------------------------------------------------------------------
    # Field-level patron credential CRUD (read-merge-write)
    # ------------------------------------------------------------------

    async def update_patron_credential(
        self,
        patron_npub: str,
        field: str,
        value: str,
        *,
        service: str | None = None,
    ) -> bool:
        """Add or update a single credential field, preserving all others.

        Decrypts the existing blob, merges the field, re-encrypts,
        and stores. Creates a new blob if none exists.
        """
        svc = service or self.patron_credential_service
        if not svc:
            return False
        existing = await self.load_patron_session(patron_npub, service=svc) or {}
        existing[field] = value
        return await self.store_patron_session(patron_npub, existing, service=svc)

    async def delete_patron_credential(
        self,
        patron_npub: str,
        field: str,
        *,
        service: str | None = None,
    ) -> bool:
        """Remove a single credential field, preserving all others."""
        svc = service or self.patron_credential_service
        if not svc:
            return False
        existing = await self.load_patron_session(patron_npub, service=svc) or {}
        if field not in existing:
            return True  # already absent
        del existing[field]
        return await self.store_patron_session(patron_npub, existing, service=svc)

    async def get_patron_credential(
        self,
        patron_npub: str,
        field: str,
        *,
        service: str | None = None,
    ) -> str | None:
        """Read a single credential field. Returns None if absent."""
        svc = service or self.patron_credential_service
        if not svc:
            return None
        existing = await self.load_patron_session(patron_npub, service=svc) or {}
        return existing.get(field)

    async def list_patron_credential_fields(
        self,
        patron_npub: str,
        *,
        service: str | None = None,
    ) -> list[str]:
        """List field names stored for a patron (names only, not values)."""
        svc = service or self.patron_credential_service
        if not svc:
            return []
        existing = await self.load_patron_session(patron_npub, service=svc) or {}
        return list(existing.keys())

    # ------------------------------------------------------------------
    # OAuth session restore (generic refresh-and-persist)
    # ------------------------------------------------------------------

    async def restore_oauth_session(
        self,
        patron_npub: str,
    ) -> tuple[dict[str, str] | None, str]:
        """Restore OAuth tokens from vault, refreshing if expired.

        Uses the ``OAuthProviderConfig`` attached to this runtime to
        determine the token endpoint, credential field names, and
        whether auto-refresh is enabled.  On successful refresh, the
        new tokens are persisted back to the vault so subsequent
        cold starts find fresh credentials.

        Returns:
            ``(creds, "")`` on success — *creds* contains at least
            ``access_token``, ``token_json``, and ``expires_at``.

            ``(None, situation)`` on failure — *situation* is one of
            ``"no_oauth_config"``, ``"no_credentials"``,
            ``"token_expired"``, or ``"vault_bootstrapping"``.
        """
        opc = self._oauth_provider
        if opc is None:
            return None, "no_oauth_config"

        svc = opc.service_name

        # Stage 1: load from vault
        try:
            creds = await self.load_patron_session(patron_npub, service=svc)
        except Exception:
            return None, "vault_bootstrapping"

        if not creds or not creds.get("access_token"):
            return None, "no_credentials"

        # Stage 2: check expiration
        import time as _time
        expires_at = float(creds.get("expires_at", "0"))
        if _time.time() <= expires_at:
            return creds, ""  # still valid

        # Stage 3: refresh if enabled
        if not opc.refresh_enabled:
            return None, "token_expired"

        refresh_token = creds.get("refresh_token", "")
        if not refresh_token:
            return None, "token_expired"

        try:
            op_creds = await self.load_credentials(
                [opc.client_id_field, opc.client_secret_field],
            )
        except Exception:
            return None, "token_expired"

        client_id = op_creds.get(opc.client_id_field, "")
        client_secret = op_creds.get(opc.client_secret_field, "")
        if not client_id or not client_secret:
            return None, "token_expired"

        try:
            from tollbooth.oauth2_collector import refresh_access_token
            new_token = await refresh_access_token(
                client_id, client_secret, refresh_token, opc.token_url,
            )
        except Exception as exc:
            logger.warning(
                "OAuth token refresh failed for %s: %s", patron_npub[:20], exc,
            )
            return None, "token_expired"

        # Build vault data from refreshed token
        import json as _json
        vault_data: dict[str, str] = {
            "token_json": _json.dumps(new_token),
            "access_token": new_token.get("access_token", ""),
            "refresh_token": new_token.get("refresh_token", refresh_token),
            "expires_at": str(new_token.get("expires_at", "")),
            "token_type": new_token.get("token_type", "Bearer"),
        }

        # Carry forward non-token fields from original creds (e.g., account_hash)
        for key, val in creds.items():
            if key not in vault_data:
                vault_data[key] = val

        # Operator callback (e.g., Schwab fetches account_hash on refresh too)
        if opc.on_token_received is not None:
            try:
                extra = await opc.on_token_received(patron_npub, new_token)
                if extra:
                    vault_data.update({k: str(v) for k, v in extra.items()})
            except Exception as exc:
                logger.warning("on_token_received callback failed on refresh: %s", exc)

        # Persist refreshed tokens
        await self.store_patron_session(patron_npub, vault_data, service=svc)
        logger.info("Refreshed and persisted OAuth token for %s.", patron_npub[:20])

        return vault_data, ""

    # ------------------------------------------------------------------
    # Identity-parameter validation helpers
    # ------------------------------------------------------------------

    def npub_validation_error(
        self, npub: str, *, param: str = "npub",
    ) -> dict[str, Any] | None:
        """Validate an npub parameter.  Returns ``None`` if valid, or a
        structured error dict (``{success, error_code, error}``) if not.

        Distinguishes the "missing" case (param empty/None) from the
        "invalid" case (bad format / bech32 decode failure) so the
        calling agent knows whether the patron forgot to pass it or
        passed a malformed value.

        Standardizes the validation pattern that every wheel-side and
        consumer-side tool needs to repeat (DRY):

        ::

            err = runtime.npub_validation_error(npub, param="patron_npub")
            if err is not None:
                return err
        """
        from tollbooth.constants import ErrorCode

        if not npub:
            return {
                "success": False,
                "error_code": ErrorCode.NPUB_MISSING,
                "error": (
                    f"{param} is required. Pass your Nostr public key "
                    "(npub1...) to identify yourself."
                ),
            }
        try:
            resolve_npub(npub)
        except ValueError as exc:
            return {
                "success": False,
                "error_code": ErrorCode.NPUB_INVALID,
                "error": str(exc),
            }
        return None

    def proof_validation_error(
        self, proof: str, *, param: str = "proof_token",
    ) -> dict[str, Any] | None:
        """Validate a proof_token parameter for the npub-proof flow.
        Returns ``None`` if non-empty, or a structured error dict if
        absent.

        This is the "param missing" check only — actual cache lookup
        and Schnorr verification happen in ``debit_or_deny`` and
        produce the more specific ``PROOF_REFRESH_NEEDED`` /
        ``PROOF_INVALID`` codes when needed.
        """
        from tollbooth.constants import ErrorCode

        if not proof:
            slug = self._slug or ""
            prefix = f"{slug}_" if slug else ""
            return {
                "success": False,
                "error_code": ErrorCode.PROOF_MISSING,
                "error": (
                    f"{param} is required. Establish npub ownership first "
                    "via the proof exchange."
                ),
                "next_steps": [
                    f"{prefix}request_npub_proof(patron_npub=<patron_npub>)",
                    "Sign the DM challenge from your Nostr client",
                    f"{prefix}receive_npub_proof(patron_npub=<patron_npub>) — remember the returned proof_token",
                    "Pass proof_token as the proof parameter on every subsequent paid tool call",
                ],
            }
        return None

    # ------------------------------------------------------------------
    # OAuth situation → structured patron-facing error
    # ------------------------------------------------------------------

    def oauth_situation_response(self, situation: str) -> dict[str, Any]:
        """Build a structured patron-facing error for a session-restoration situation.

        Maps the ``situation`` string returned by ``restore_oauth_session``
        to a canonical ``error_code`` + ``next_steps`` recipe.  The
        mapping is **1:1** — situations with the same recovery flow
        still get distinct ``error_code`` values, because a calling
        agent needs to distinguish "your existing session aged out"
        from "you've never authorized" even though both end at
        ``begin_oauth``.

        Tool names in ``next_steps`` are qualified with this operator's
        slug so the response is directly invocable.

        Standard situations:

        - ``token_expired`` → ``oauth_token_expired`` (returning patron,
          refresh token revoked or aged out)
        - ``no_credentials`` → ``oauth_not_yet_authorized`` (first-time
          patron with no vault entry)
        - ``vault_bootstrapping`` → ``warming_up``
        - ``operator_not_configured`` → ``operator_credentials_missing``
          (vault loaded but operator app credentials absent)
        - ``no_oauth_config`` → ``oauth_not_wired`` (operator MCP never
          declared an OAuthProviderConfig — deployment bug)
        - *unrecognized* → ``oauth_situation_unknown`` with the raw
          situation echoed in the message (preserves diagnostic
          signal instead of silently collapsing to a routine code)

        Consumers with operator-specific situations (e.g. schwab-mcp's
        ``no_account_hash``) handle those inline and delegate the
        standard cases here.
        """
        from tollbooth.constants import ErrorCode

        slug = self._slug or ""
        prefix = f"{slug}_" if slug else ""
        oauth_recovery = [
            f"{prefix}begin_oauth(npub=<patron_npub>)",
            "Open the authorize_url, log in, click Allow",
            f"{prefix}check_oauth_status(npub=<patron_npub>) promptly after Allow",
        ]

        table: dict[str, dict[str, Any]] = {
            "token_expired": {
                "error_code": ErrorCode.OAUTH_TOKEN_EXPIRED,
                "error": (
                    "Your existing OAuth session has aged out — the refresh "
                    "token is no longer accepted by the upstream provider. "
                    "Reconnect to continue."
                ),
                "next_steps": oauth_recovery,
            },
            "no_credentials": {
                "error_code": ErrorCode.OAUTH_NOT_YET_AUTHORIZED,
                "error": (
                    "No OAuth session is linked to this npub yet. Authorize "
                    "with the upstream provider to begin."
                ),
                "next_steps": oauth_recovery,
            },
            "vault_bootstrapping": {
                "error_code": ErrorCode.WARMING_UP,
                "error": (
                    "The server is establishing its encrypted connection to "
                    "the credential vault. This happens once after a cold start."
                ),
                "next_steps": ["Repeat your request shortly — no re-authentication needed."],
            },
            "operator_not_configured": {
                "error_code": ErrorCode.OPERATOR_CREDENTIALS_MISSING,
                "error": (
                    "The operator's upstream API credentials have not been "
                    "delivered yet. This is an operator setup step, not a "
                    "patron action."
                ),
                "next_steps": ["Contact the operator", "Try again later"],
            },
            "no_oauth_config": {
                "error_code": ErrorCode.OAUTH_NOT_WIRED,
                "error": (
                    "This operator's MCP has no OAuth provider configured. "
                    "This is a deployment-side issue, not a patron action."
                ),
                "next_steps": ["Contact the operator"],
            },
        }

        spec = table.get(situation, {
            "error_code": ErrorCode.OAUTH_SITUATION_UNKNOWN,
            "error": (
                f"OAuth session unavailable for an unrecognized reason "
                f"({situation!r}). Reconnecting may help; if it persists, "
                "this likely warrants a closer look at operator logs."
            ),
            "next_steps": oauth_recovery,
        })
        return {"success": False, **spec}

    # ------------------------------------------------------------------
    # BTCPay client (from operator credential vault)
    # ------------------------------------------------------------------

    async def ensure_cashier(self) -> Any:
        """Return a BTCPayClient constructed from vault credentials.

        Loads btcpay_host, btcpay_api_key, btcpay_store_id from the
        operator credential vault.  Cached after first successful load.
        """
        if self._cashier is not None:
            return self._cashier

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
        # Auto-fix common URL typos before rejecting
        if host:
            host = host.strip()
            if host.startswith("htps://"):
                host = "https://" + host[7:]
            elif host.startswith("http://") and not host.startswith("https://"):
                host = "https://" + host[7:]
        if host and not host.startswith("https://"):
            raise ValueError(
                f"btcpay_host must start with 'https://' (got '{host[:20]}...'). "
                "Re-deliver corrected credentials via Secure Courier."
            )

        self._cashier = BTCPayClient(
            host=host, api_key=api_key, store_id=store_id,
        )
        return self._cashier

    async def resolve_tranche_lifetime(self) -> int | None:
        """Return the tranche lifetime in seconds, or None if credits never expire.

        Reads the ``tranche_lifetime`` field from the active pricing model.
        """
        try:
            store = await self.ensure_pricing_store()
            from tollbooth.tools.pricing import get_pricing_model_tool
            result = await get_pricing_model_tool(store, self.operator_npub())
            if result.get("status") == "ok":
                tl = result.get("tranche_lifetime")
                if isinstance(tl, dict) and tl.get("ttl_days") is not None:
                    return int(tl["ttl_days"]) * 86400
        except Exception:
            pass
        return None

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

    _SUPPLY_TOTAL_KEY = "__total__"

    async def get_global_demand(self, tool_name: str) -> dict[str, int]:
        """Get current hourly demand and lifetime total for a tool."""
        result: dict[str, int] = {}
        try:
            vault = await self.vault()
            hourly = await vault.get_demand(tool_name, self._demand_window_key())
            if hourly:
                result[tool_name] = hourly
            total = await vault.get_demand(tool_name, self._SUPPLY_TOTAL_KEY)
            if total:
                result[f"{tool_name}:{self._SUPPLY_TOTAL_KEY}"] = total
        except Exception:
            pass
        return result

    def fire_and_forget_demand_increment(self, tool_name: str) -> None:
        """Increment hourly demand counter for a tool (non-blocking)."""
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

    def fire_and_forget_supply_increment(self, tool_name: str) -> None:
        """Increment lifetime supply counter for a tool (non-blocking)."""
        import asyncio

        async def _increment() -> None:
            try:
                vault = await self.vault()
                await vault.increment_demand(
                    tool_name, self._SUPPLY_TOTAL_KEY,
                )
            except Exception:
                pass

        asyncio.create_task(_increment())

    # ------------------------------------------------------------------
    # Opportunistic OTS notarization
    # ------------------------------------------------------------------

    _OTS_INTERVAL_SECONDS = 3600  # 60 minutes between notarizations
    _ots_last_check: float = 0.0  # monotonic timestamp of last check
    _ots_running: bool = False     # prevent concurrent notarizations

    def fire_and_forget_notarize_if_stale(self) -> None:
        """Trigger an OTS notarization if the last one is stale.

        Piggybacks on tool calls — checked on every paid_tool invocation.
        Uses monotonic time to rate-limit vault queries to once per interval.
        Non-blocking: notarization runs as a background task.
        """
        if not self._ots_enabled or self._ots_running:
            return

        import asyncio
        import time as _time

        now = _time.monotonic()
        if (now - self._ots_last_check) < self._OTS_INTERVAL_SECONDS:
            return  # checked recently, skip
        self._ots_last_check = now

        async def _maybe_notarize() -> None:
            self._ots_running = True
            try:
                vault = await self.vault()
                anchors = await vault.list_anchors(limit=1)
                if anchors:
                    from datetime import datetime, timezone
                    created = anchors[0].get("created_at", "")
                    if isinstance(created, str) and created:
                        last_ts = datetime.fromisoformat(created).timestamp()
                    elif hasattr(created, "timestamp"):
                        last_ts = created.timestamp()
                    else:
                        last_ts = 0.0
                    age = datetime.now(timezone.utc).timestamp() - last_ts
                    if age < self._OTS_INTERVAL_SECONDS:
                        return  # recent enough

                from tollbooth.tools.notarization import notarize_ledger_tool
                result = await notarize_ledger_tool(vault, self._ots_calendars)
                if result.get("success"):
                    logger.info(
                        "Opportunistic OTS notarization: %s (%d leaves, %d calendars)",
                        result.get("root_hash", "")[:16],
                        result.get("leaf_count", 0),
                        result.get("calendars_submitted", 0),
                    )
                else:
                    logger.warning("Opportunistic OTS notarization failed: %s", result.get("error"))
            except Exception as exc:
                logger.debug("Opportunistic OTS notarization skipped: %s", exc)
            finally:
                self._ots_running = False

        asyncio.create_task(_maybe_notarize())

    # ------------------------------------------------------------------
    # Paid tool decorator
    # ------------------------------------------------------------------

    def paid_tool(
        self,
        tool_id: str,
        *,
        catch_errors: bool = True,
    ) -> Callable:
        """Decorator that wraps a domain function with debit/rollback/warning.

        Args:
            tool_id: The tool's UUID (from ToolIdentity.tool_id).
            catch_errors: If True (default), catch exceptions from the body
                and return ``{"success": False, "error": ...}`` after rollback.

        The decorated function **must** accept ``npub`` and ``proof`` as keyword arguments.

        Example::

            @tool
            @runtime.paid_tool(TOOL_REGISTRY["get_weather"].tool_id)
            async def get_weather(lat: float, lon: float, npub: str = "", proof: str = ""):
                return await weather.get(lat, lon)
        """
        rt = self

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                sig = inspect.signature(fn)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # Extract npub and proof from kwargs directly —
                # inspect.signature().bind() can lose optional kwargs
                # when FastMCP passes arguments through Pydantic models.
                npub = kwargs.get("npub", "") or bound.arguments.get("npub", "")
                proof = kwargs.get("proof", "") or bound.arguments.get("proof", "")

                call_kwargs = dict(bound.arguments)
                result_or_cost = await rt.debit_or_deny(
                    tool_id, npub,
                    proof=proof,
                    tool_kwargs=call_kwargs,
                )
                if isinstance(result_or_cost, dict):
                    return result_or_cost  # denial

                # Store the computed cost on the runtime so the body can
                # read it without recomputing (e.g., certify_credits).
                rt._last_debit_cost = result_or_cost

                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    await rt.rollback_debit(tool_id, npub, tool_kwargs=call_kwargs)
                    if catch_errors:
                        from tollbooth.constants import ErrorCode as _EC
                        logger.error("Tool %s failed: %s", tool_id, exc, exc_info=True)
                        # Map upstream-API auth failures to a friendlier code
                        # so calling agents can route directly to the OAuth
                        # refresh flow without parsing prose.
                        msg = str(exc).lower()
                        is_upstream_auth = (
                            "401" in msg
                            or "unauthorized" in msg
                            or "invalid_grant" in msg
                            or "token expired" in msg
                            or "token has expired" in msg
                        )
                        if is_upstream_auth:
                            return {
                                "success": False,
                                "error_code": _EC.UPSTREAM_AUTH_REFRESH_NEEDED,
                                "error": (
                                    "Upstream API authorization needs to be re-granted. "
                                    "This is routine — your previous grant has been "
                                    "revoked or has aged out."
                                ),
                                "next_steps": [
                                    "begin_oauth(npub=<patron_npub>)",
                                    "Open the authorize_url, log in, click Allow",
                                    "check_oauth_status(npub=<patron_npub>) promptly after Allow",
                                ],
                            }
                        return {
                            "success": False,
                            "error_code": _EC.TOOL_EXECUTION_FAILED,
                            "error": "Tool execution failed. Check operator logs.",
                        }
                    raise

                rt.fire_and_forget_demand_increment(rt.mcp_name_for(tool_id))
                rt.fire_and_forget_supply_increment(rt.mcp_name_for(tool_id))
                rt.fire_and_forget_notarize_if_stale()
                if isinstance(result, dict):
                    result = await rt.inject_low_balance_warning(result, npub)
                return result

            wrapper._tool_id = tool_id  # type: ignore[attr-defined]
            # Record function name for MCP name stamping.
            # The MCP tool name is {slug}_{fn.__name__}, but the slug
            # isn't known until register_standard_tools runs. Store the
            # function name so the stamping step can compute mcp_name.
            rt._tool_func_names[tool_id] = fn.__name__
            return wrapper
        return decorator


def _build_initial_pricing_model(
    rt: OperatorRuntime,
    service_name: str,
) -> str:
    """Build an initial pricing model from the tool registry.

    Every registered tool gets an entry with its UUID, category, and
    intent.  Pricing hints on ToolIdentity seed sensible defaults —
    e.g. ad valorem tools start with their declared rate.
    """
    import json as _json

    tools = []
    for tool_id, identity in rt._tool_registry.items():
        has_hint = identity.pricing_hint_value > 0
        entry: dict[str, Any] = {
            "tool_id": tool_id,
            "tool_name": rt.mcp_name_for(tool_id),
            "price_sats": identity.pricing_hint_value,
            "priced": identity.category in ("free", "restricted") or has_hint,
            "category": identity.category,
            "intent": identity.intent,
            "price_type": identity.pricing_hint_type,
        }
        if identity.pricing_hint_param:
            entry["price_formula"] = identity.pricing_hint_param
        if identity.pricing_hint_min > 0:
            entry["min_cost"] = identity.pricing_hint_min
        tools.append(entry)

    model = {
        "name": f"{service_name or 'Operator'} Initial Pricing",
        "tools": tools,
    }
    return _json.dumps(model)


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
    oracle_tool = make_slug_tool(mcp, "oracle")
    rt._slug = slug
    rt._mcp_name_cache.clear()  # invalidate any cached names

    # -- Credit tools --------------------------------------------------

    @tool
    async def check_balance(npub: str = "", proof: str = "") -> dict[str, Any]:
        """Check a patron's credit balance at this operator.

        This is the patron's spending balance — credits purchased via
        Lightning for tool calls at this operator. For the operator's
        own balance at the Authority (needed to certify patron purchases),
        use authority_check_balance instead.

        Free — no credits required.

        Args:
            npub: Required. The patron's Nostr public key (npub1...).
        """
        try:
            npub = resolve_npub(npub)
            cache = await rt.ledger_cache()
        except ValueError as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        return await credits.check_balance_tool(cache, npub)

    @tool
    async def purchase_credits(amount_sats: int = 1000, npub: str = "", proof: str = "") -> dict[str, Any]:
        """Buy credits via Bitcoin Lightning.

        Creates a Lightning invoice. Pay it with any Lightning wallet,
        then call check_payment to confirm.

        Free — no credits required to call.

        Args:
            amount_sats: Satoshis to purchase (default 1000).
            npub: Required. Your Nostr public key (npub1...).
        """
        try:
            npub = resolve_npub(npub)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        if rt._purchase_mode == "direct":
            # Trust-root mode: no upstream certificate needed.
            try:
                cashier = await rt.ensure_cashier()
                cache = await rt.ledger_cache()
                from tollbooth.tools.credits import direct_purchase_tool
                return await direct_purchase_tool(
                    cashier, cache, npub, amount_sats,
                )
            except ValueError as e:
                return {"success": False, "error": str(e)}

        # Certified mode: obtain Authority certificate first.
        try:
            from tollbooth.authority_client import AuthorityCertifier
            from tollbooth.registry import resolve_authority_service
            auth_info = await resolve_authority_service(rt.operator_npub())
            cert_result = await AuthorityCertifier(
                auth_info["url"], rt.operator_npub(), rt._get_nsec(),
            ).certify_credits(amount_sats)
            certificate = cert_result.get("certificate", "")
        except Exception as e:
            return {"success": False, "error": f"Authority certification failed: {e}"}

        try:
            cashier = await rt.ensure_cashier()
            cache = await rt.ledger_cache()
            from tollbooth.tools import credits
            ttl = await rt.resolve_tranche_lifetime()
            return await credits.purchase_credits_tool(
                cashier, cache, npub, amount_sats, certificate,
                authority_npub=auth_info.get("npub", ""),
                tranche_lifetime_seconds=ttl,
            )
        except ValueError as e:
            return {"success": False, "error": str(e)}

    @tool
    async def check_payment(invoice_id: str, npub: str = "", proof: str = "") -> dict[str, Any]:
        """Check the payment status of a Lightning invoice.

        Call after paying the invoice from purchase_credits.
        Free — no credits required.

        Args:
            npub: Required. Your Nostr public key (npub1...).
        """
        try:
            npub = resolve_npub(npub)
            cashier = await rt.ensure_cashier()
            cache = await rt.ledger_cache()
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        ttl = await rt.resolve_tranche_lifetime()
        result = await credits.check_payment_tool(
            cashier, cache, npub, invoice_id,
            tranche_lifetime_seconds=ttl,
        )

        # Send a thank-you DM on successful settlement
        if result.get("status") == "Settled" and result.get("credits_granted", 0) > 0:
            try:
                courier = await rt.courier()
                if courier is not None:
                    credited = result["credits_granted"]
                    courier._exchange.send_dm(
                        npub,
                        f"\u26a1 Thank you for your purchase!\n\n"
                        f"Invoice {invoice_id} settled.\n"
                        f"{credited:,} credits added to your balance.\n\n"
                        f"— {service_name or slug}",
                    )
            except Exception:
                pass  # DM is a courtesy, never blocks

        return result

    @tool
    async def restore_credits(invoice_id: str, npub: str = "", proof: str = "") -> dict[str, Any]:
        """Restore credits from a previously paid invoice. Free.

        Args:
            npub: Required. Your Nostr public key (npub1...).
        """
        try:
            npub = resolve_npub(npub)
            cashier = await rt.ensure_cashier()
            cache = await rt.ledger_cache()
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        ttl = await rt.resolve_tranche_lifetime()
        return await credits.restore_credits_tool(
            cashier, cache, npub, invoice_id,
            tranche_lifetime_seconds=ttl,
        )

    @tool
    async def account_statement(npub: str = "", proof: str = "", days: int = 30) -> dict[str, Any]:
        """Generate a patron's account statement at this operator.

        Returns the patron's purchase history, active credit tranches,
        per-tool usage breakdown, and recent daily usage logs. This is
        the patron's spending account — not the operator's Authority
        tax balance.

        Free — no credits consumed.

        Args:
            npub: Required. The patron's Nostr public key (npub1...).
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
    async def account_statement_infographic(npub: str = "", proof: str = "", days: int = 30) -> dict[str, Any]:
        """Generate a visual SVG infographic of your account statement.

        Returns the same data as account_statement, rendered as a dark-themed
        SVG graphic with balance hero, metrics cards, health gauge, tranche
        table, and tool usage breakdown. Costs 1 api_sat per call.

        Args:
            npub: Required. Your Nostr public key (npub1...).
            days: Number of days of daily usage history to include (default 30).
        """
        try:
            npub = resolve_npub(npub)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tool_identity import capability_uuid
        result_or_cost = await rt.debit_or_deny(capability_uuid("account_statement_infographic"), npub, proof=proof)
        if isinstance(result_or_cost, dict):
            return result_or_cost
        try:
            cache = await rt.ledger_cache()
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        statement = await credits.account_statement_tool(cache, npub, days=days)
        from tollbooth.infographic import render_account_infographic
        svg = render_account_infographic(
            statement, service_name=service_name or slug,
        )
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
        # Trigger late-attach of credential vault if courier exists without one
        try:
            c = await rt.courier()
            courier_ok = (c is not None
                          and hasattr(c, '_exchange')
                          and c._exchange._credential_vault is not None)
        except Exception:
            courier_ok = False

        wheel_version = "unknown"
        try:
            import importlib.metadata
            wheel_version = importlib.metadata.version("tollbooth-dpyc")
        except Exception:
            pass

        # Collect FastMCP / Horizon build info from env
        build_info: dict[str, str] = {}
        for key in (
            "FASTMCP_CLOUD_URL",
            "FASTMCP_CLOUD_GIT_COMMIT_SHA",
            "FASTMCP_CLOUD_GIT_REPO",
        ):
            val = os.environ.get(key)
            if val:
                build_info[key.lower()] = val

        # Operator npub fingerprint — short hash for patron verification
        import hashlib
        op_npub = ""
        op_npub_hash = ""
        try:
            op_npub = rt.operator_npub()
            op_npub_hash = hashlib.sha256(op_npub.encode()).hexdigest()[:12]
        except Exception:
            pass

        result: dict[str, Any] = {
            "success": True,
            "service": service_name or slug,
            "slug": slug,
            "version": service_version,
            "tollbooth_dpyc_version": wheel_version,
            "vault_configured": vault_ok,
            "courier_has_vault": courier_ok,
            "operator_npub_hash": op_npub_hash,
            "process_id": os.getpid(),
            "build_info": build_info or None,
        }
        return result

    # -- Onboarding ----------------------------------------------------

    @tool
    async def get_operator_onboarding_status() -> dict[str, Any]:
        """Report this operator's configuration readiness.

        Shows which operator settings are configured, which are missing,
        and how to deliver each missing value. For patron-level credential
        status, use get_patron_onboarding_status instead. Free.
        """
        return await rt.onboarding_status()

    @tool
    async def get_patron_onboarding_status(patron_npub: str = "", proof: str = "") -> dict[str, Any]:
        """Report a patron's credential readiness for this operator.

        For set-once services (eXcalibur, TheBrain), shows which patron
        secrets are configured and which are missing. For dynamic/OAuth2
        services (Schwab), reports that no patron credentials are needed.
        Free.

        Args:
            patron_npub: Required. The patron's Nostr public key.
        """
        err = rt.npub_validation_error(patron_npub, param="patron_npub")
        if err is not None:
            return err
        return await rt.patron_onboarding_status(patron_npub)

    # -- Secure Courier ------------------------------------------------

    @tool
    async def session_status(patron_npub: str = "") -> dict[str, Any]:
        """Check operator readiness. Returns the operator lifecycle
        state and clear guidance on what to do next. Free.

        Lifecycle states:
        - ready: Operator is warm and fully operational. Proceed with tool calls.
        - warming_up: Operator is initializing (cold start). Try a tool call — it will warm up on demand.
        - not_registered: Operator has no Authority relationship yet. Call register_operator first.
        - no_identity: Operator nsec is not configured. Deployment issue.

        Args:
            patron_npub: Optional. If supplied, the response includes an
                ``upstream_oauth`` block with the patron's stored OAuth
                token expiry (runtime-derived from vault state) so a
                client can refresh proactively rather than reactively
                after a stale-token failure.
        """
        # 1. Identity check
        try:
            npub = rt.operator_npub()
        except (RuntimeError, ValueError):
            return {
                "success": True,
                "lifecycle": "no_identity",
                "message": "Operator identity (nsec) is not configured. "
                           "This is a deployment issue — set TOLLBOOTH_NOSTR_OPERATOR_NSEC.",
            }

        # 2. Vault / registration check
        vault_ok = rt._vault is not None
        if not vault_ok:
            # Try to bring up the vault to distinguish warming_up vs not_registered
            try:
                await rt.vault()
                vault_ok = True
            except Exception as exc:
                exc_str = str(exc)
                if "not registered" in exc_str.lower() or "no neon url" in exc_str.lower():
                    return {
                        "success": True,
                        "lifecycle": "not_registered",
                        "operator_npub": npub,
                        "message": "Operator is not yet registered with an Authority. "
                                   "Call register_operator to provision persistence.",
                    }
                # Bootstrap failed for another reason — still warming up
                return {
                    "success": True,
                    "lifecycle": "warming_up",
                    "operator_npub": npub,
                    "message": "Operator is initializing. Try a tool call — "
                               "it will warm up on demand. If this persists, "
                               "check the deployment logs.",
                    "detail": exc_str,
                }

        # 3. Check if vault just finished bootstrapping (< 15s ago)
        import time as _time
        if rt._vault_ready_at > 0 and (_time.monotonic() - rt._vault_ready_at) < 15:
            return {
                "success": True,
                "lifecycle": "warming_up",
                "operator_npub": npub,
                "message": "Operator vault just came online. Credential "
                           "and ledger caches are still hydrating. "
                           "Try a tool call — it will complete the warm-up.",
            }

        # 4. Fully ready
        result: dict[str, Any] = {
            "success": True,
            "lifecycle": "ready",
            "operator_npub": npub,
            "operator_credential_service": rt.operator_credential_service,
            "patron_credential_service": rt.patron_credential_service,
            "message": "Operator is ready. Proceed with tool calls.",
        }

        # Optional: per-patron upstream OAuth token expiry (runtime-derived).
        # No magic numbers in the docstring — the value comes from whatever
        # the upstream provider returned at token issuance.
        if patron_npub:
            try:
                resolved = resolve_npub(patron_npub)
            except ValueError:
                return result
            opc = rt._oauth_provider
            if opc is not None:
                try:
                    creds = await rt.load_patron_session(resolved, service=opc.service_name)
                except Exception:
                    creds = None
                if creds and creds.get("access_token"):
                    import time as _t
                    expires_at = 0.0
                    try:
                        expires_at = float(creds.get("expires_at", "0") or 0)
                    except (TypeError, ValueError):
                        expires_at = 0.0
                    has_refresh = bool(creds.get("refresh_token"))
                    block: dict[str, Any] = {
                        "service": opc.service_name,
                        "has_access_token": True,
                        "has_refresh_token": has_refresh,
                        "refresh_enabled": opc.refresh_enabled,
                    }
                    if expires_at > 0:
                        block["access_token_expires_at"] = expires_at
                        block["access_token_expires_in_seconds"] = max(
                            0, int(expires_at - _t.time()),
                        )
                    result["upstream_oauth"] = block
        return result

    @tool
    async def request_credential_channel(
        sender_npub: str = "",
        proof: str = "",
        service: str = "",
    ) -> dict[str, Any]:
        """Open a Secure Courier channel for credential delivery.

        Sends a welcome DM with a credential template. The recipient
        must read the DM in their Nostr client, fill in the fields,
        and reply manually. **This is a human-in-the-loop flow.**

        After calling this tool, STOP and tell the user what to do.
        Wait for the user to confirm they have replied before calling
        ``receive_credentials``. Do NOT poll or retry — each
        ``receive_credentials`` call destructively drains the relay
        mailbox.

        Args:
            sender_npub: Required. The npub to send the template to.
            service: Required. The credential service name (e.g.,
                from get_operator_onboarding_status or get_patron_onboarding_status).
        Free.
        """
        if not sender_npub:
            return {
                "success": False,
                "error": "sender_npub is required.",
            }
        if not service:
            return {
                "success": False,
                "error": (
                    "service is required. Use the credential_service "
                    "from get_operator_onboarding_status or get_patron_onboarding_status. Available: "
                    + ", ".join(
                        s for s in [
                            rt.operator_credential_service,
                            rt.patron_credential_service,
                        ] if s
                    )
                ),
            }
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
        proof: str = "",
        service: str = "",
        credential_card: str = "",
        force_relay: bool = False,
    ) -> dict[str, Any]:
        """Pick up credentials from the Secure Courier.

        **Call this only after the user confirms they have replied.**
        This tool destructively drains ALL DMs from the sender on the
        relay. If called before the user replies, their message will
        never be found. Do NOT poll, loop, or retry.

        Checks the vault first (instant), then reads Nostr relays for
        encrypted DMs. If a credential_card (ncred1...) is provided,
        redeems it directly without relay access. On success, the
        payment processor client is reinitialized from the new
        credentials — no server restart needed.

        Args:
            sender_npub: Required. The npub that sent the credentials.
            service: Required. The credential service name (must match
                the service used in request_credential_channel).
            force_relay: Skip the vault cache and read Nostr relays
                for new DMs. Use after resending corrected credentials.
        Free.
        """
        if not sender_npub:
            return {
                "success": False,
                "error": "sender_npub is required.",
            }
        if not service:
            return {
                "success": False,
                "error": (
                    "service is required. Use the same service from "
                    "request_credential_channel. Available: "
                    + ", ".join(
                        s for s in [
                            rt.operator_credential_service,
                            rt.patron_credential_service,
                        ] if s
                    )
                ),
            }
        courier = await rt.courier()
        if courier is None:
            return {"success": False, "error": "Secure Courier not configured."}
        try:
            if credential_card:
                result = await courier._exchange.redeem_credential_card(
                    credential_card, service=service,
                )
            else:
                result = await courier.receive(sender_npub, service=service, force_relay=force_relay)

            # Validate credentials via operator callback before accepting.
            # The courier strips credentials from the result for security,
            # so reload them from the vault where they were just stored.
            if result.get("success") and service == rt.operator_credential_service and rt._credential_validator:
                try:
                    creds = await rt.load_credentials(list(rt._operator_credential_template.fields.keys()))
                except Exception:
                    creds = {}
                errors = rt._credential_validator(creds)
                if errors:
                    # Reject: forget the bad creds
                    try:
                        vault = courier._exchange._credential_vault
                        if vault:
                            await vault.store_credentials(service, rt.operator_npub(), "")
                    except Exception:
                        pass
                    # DM the sender about the problem
                    rejection_msg = (
                        "Credential rejection from " + (service_name or slug) + ":\n\n"
                        + "\n".join(f"  - {e}" for e in errors)
                        + "\n\nPlease correct and resend."
                    )
                    try:
                        await courier.send(
                            recipient_npub=sender_npub,
                            message=rejection_msg,
                        )
                        result["rejection_dm_sent"] = True
                    except Exception:
                        pass
                    return {
                        "success": False,
                        "validation_errors": errors,
                        "error": "Credentials received but failed validation: " + "; ".join(errors),
                        "message": "A rejection DM has been sent. Please correct and resend.",
                    }
                rt._cashier = None
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    async def forget_credentials(
        service: str = "",
        npub: str = "",
        proof: str = "",
    ) -> dict[str, Any]:
        """Delete vaulted credentials for a specific service and npub.

        For operator credentials, omit npub (defaults to operator).
        For patron credentials, pass the patron's npub.

        Args:
            service: Required. The credential service to forget.
            npub: The npub whose credentials to forget. Defaults to
                operator npub for operator services.
        Free.
        """
        if not service:
            return {
                "success": False,
                "error": (
                    "service is required. Available: "
                    + ", ".join(
                        s for s in [
                            rt.operator_credential_service,
                            rt.patron_credential_service,
                        ] if s
                    )
                ),
            }
        target_npub = npub if npub else rt.operator_npub()
        courier = await rt.courier()
        if courier is None:
            return {"success": False, "error": "Secure Courier not configured."}
        try:
            result = await courier.forget(target_npub, service)
            if service == rt.operator_credential_service:
                rt._cashier = None
            # Fire on_forget callback so operators can clear caches
            if rt._on_forget and result.get("success"):
                try:
                    rt._on_forget(service, target_npub)
                except Exception:
                    pass
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -- Patron credential tools (only if patron template is set) ------

    if rt._patron_credential_template is not None:
        @tool
        async def request_patron_credentials(
            sender_npub: str = "",
            proof: str = "",
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
            proof: str = "",
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
                        credential_card, service=service,
                    )
                return await courier.receive(sender_npub, service)
            except Exception as e:
                return {"success": False, "error": str(e)}

    # Prune registry entries for conditionally-skipped tools so they
    # don't appear in the pricing model or mismatch detection.
    if rt._patron_credential_template is None:
        from tollbooth.tool_identity import capability_uuid
        for cap in ("request_patron_credentials", "receive_patron_credentials"):
            rt._tool_registry.pop(capability_uuid(cap), None)

    # -- Patron credential CRUD (field-level) --------------------------------

    @tool
    async def update_patron_credential(
        npub: str = "", field: str = "", value: str = "",
        proof: str = "",
    ) -> dict[str, Any]:
        """Add or update a single patron credential field.

        Merges into existing stored credentials without affecting
        other fields. Useful for setting an account identifier
        after OAuth, changing a default brain, etc. Free.

        Args:
            npub: Required. The patron's npub.
            field: Required. The credential field name to set.
            value: Required. The value to store.
        """
        err = rt.npub_validation_error(npub)
        if err is not None:
            return err
        if not field:
            return {"success": False, "error": "field is required."}
        if not value:
            return {"success": False, "error": "value is required."}
        try:
            ok = await rt.update_patron_credential(npub, field, value)
            if ok:
                return {
                    "success": True,
                    "message": f"Field '{field}' updated.",
                }
            return {"success": False, "error": "No credential service configured."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    async def delete_patron_credential(
        npub: str = "", field: str = "",
        proof: str = "",
    ) -> dict[str, Any]:
        """Remove a single patron credential field.

        Deletes one field from stored credentials without affecting
        other fields. Free.

        Args:
            npub: Required. The patron's npub.
            field: Required. The credential field name to remove.
        """
        err = rt.npub_validation_error(npub)
        if err is not None:
            return err
        if not field:
            return {"success": False, "error": "field is required."}
        try:
            ok = await rt.delete_patron_credential(npub, field)
            if ok:
                return {
                    "success": True,
                    "message": f"Field '{field}' removed.",
                }
            return {"success": False, "error": "No credential service configured."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    async def get_patron_credential_fields(
        npub: str = "",
    ) -> dict[str, Any]:
        """List stored patron credential field names (not values).

        Returns the names of fields stored for a patron. Values
        are never exposed — use this to verify which fields are
        configured. Free.

        Args:
            npub: Required. The patron's npub.
        """
        err = rt.npub_validation_error(npub)
        if err is not None:
            return err
        try:
            fields = await rt.list_patron_credential_fields(npub)
            return {
                "success": True,
                "fields": fields,
                "count": len(fields),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -- OAuth2 tools (only if oauth_provider is configured) ----------------

    if rt._oauth_provider is not None:
        _opc = rt._oauth_provider  # OAuthProviderConfig
        _OAUTH_SERVICE = _opc.service_name

        @tool
        async def begin_oauth(npub: str = "") -> dict[str, Any]:
            """Start the OAuth2 authorization flow.

            Returns an authorization URL. Open it in a browser to log in
            and authorize. Then call ``check_oauth_status`` with the
            same npub to complete. Free.

            Args:
                npub: Required. Your DPYC patron npub (npub1...).
            """
            err = rt.npub_validation_error(npub)
            if err is not None:
                return err
            resolved = resolve_npub(npub)

            # Load operator credentials using vendor field names
            _id_field = _opc.client_id_field
            _secret_field = _opc.client_secret_field
            try:
                creds = await rt.load_credentials(
                    [_id_field, _secret_field],
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": (
                        f"Operator API credentials ({_id_field}) have not been delivered yet. "
                        "This is not an error — the operator needs to deliver credentials "
                        "via Secure Courier (request_credential_channel / receive_credentials) "
                        "before OAuth can start."
                    ),
                }

            client_id = creds.get(_id_field, "")
            if not client_id:
                return {
                    "success": False,
                    "error": (
                        f"Operator API credentials ({_id_field}) have not been delivered yet. "
                        "This is not an error — the operator needs to deliver credentials "
                        "via Secure Courier (request_credential_channel / receive_credentials) "
                        "before OAuth can start."
                    ),
                }

            # Resolve collector redirect URI
            try:
                from tollbooth.registry import resolve_service_by_name
                svc = await resolve_service_by_name("tollbooth-oauth2-callback")
                redirect_uri = svc["url"].rstrip("/")
            except Exception as e:
                return {"success": False, "error": f"OAuth2 collector not found: {e}"}

            from tollbooth.oauth2_collector import (
                build_authorize_url,
                generate_pkce_pair,
            )

            extra_params: dict[str, str] = {}
            verifier = ""
            if _opc.pkce:
                verifier, challenge = generate_pkce_pair()
                extra_params["code_challenge"] = challenge
                extra_params["code_challenge_method"] = "S256"

            authorize_url = build_authorize_url(
                _opc.authorize_url,
                client_id,
                redirect_uri,
                resolved,  # state = patron npub
                scope=_opc.scopes or None,
                extra_params=extra_params or None,
            )

            # Store PKCE verifier and redirect_uri for check_oauth_status
            vault_data: dict[str, str] = {"redirect_uri": redirect_uri}
            if verifier:
                vault_data["pkce_verifier"] = verifier
            await rt.store_patron_session(
                resolved, vault_data, service=f"_oauth_pending_{_OAUTH_SERVICE}",
            )

            # Try to shorten the URL
            short_url = None
            try:
                from tollbooth.shortlinks import create_shortlink
                short_url = await create_shortlink(authorize_url)
            except Exception:
                pass

            result: dict[str, Any] = {
                "success": True,
                "status": "pending",
                "authorize_url": authorize_url,
                "message": (
                    "Open authorize_url in the browser (the full URL, not the "
                    "short one — redirects may truncate query parameters). "
                    "Then call check_oauth_status with the same npub."
                ),
            }
            if short_url:
                result["authorize_url_short"] = short_url
                result["message"] += (
                    f" For display: {short_url}"
                )
            return result

        @tool
        async def check_oauth_status(npub: str = "") -> dict[str, Any]:
            """Check whether the OAuth2 authorization flow has completed.

            Call after opening the authorization URL from ``begin_oauth``
            and completing the login in your browser. Free.

            Args:
                npub: Required. The same npub used in begin_oauth.
            """
            err = rt.npub_validation_error(npub)
            if err is not None:
                return err
            resolved = resolve_npub(npub)

            # Load pending state (PKCE verifier, redirect_uri)
            pending = await rt.load_patron_session(
                resolved, service=f"_oauth_pending_{_OAUTH_SERVICE}",
            )
            if not pending or "redirect_uri" not in pending:
                return {
                    "success": False,
                    "error": "No pending OAuth flow. Call begin_oauth first.",
                }
            redirect_uri = pending["redirect_uri"]
            verifier = pending.get("pkce_verifier")

            # Load operator credentials (vendor field names → OAuth protocol names)
            _cid_field = _opc.client_id_field
            _csec_field = _opc.client_secret_field
            try:
                creds = await rt.load_credentials(
                    [_cid_field, _csec_field],
                )
            except Exception as e:
                logger.error("Failed to load operator credentials: %s", e, exc_info=True)
                return {"success": False, "error": "Operator credentials could not be loaded. Check operator logs."}

            client_id = creds.get(_cid_field, "")
            client_secret = creds.get(_csec_field, "")

            # Poll collector for the authorization code
            try:
                from tollbooth.registry import resolve_service_by_name
                svc = await resolve_service_by_name("tollbooth-oauth2-collector")
                collector_url = svc["url"]
            except Exception as e:
                return {"success": False, "error": f"OAuth2 collector: {e}"}

            from tollbooth.oauth2_collector import (
                retrieve_code_from_collector,
                exchange_code_for_token,
            )

            code = await retrieve_code_from_collector(collector_url, resolved)
            if code is None:
                return {
                    "success": True,
                    "status": "pending",
                    "message": (
                        "Waiting for browser authorization. "
                        "Open the URL from begin_oauth."
                    ),
                }

            # Exchange code for tokens
            try:
                token = await exchange_code_for_token(
                    code, client_id, client_secret, redirect_uri,
                    _opc.token_url,
                    code_verifier=verifier,
                )
            except Exception as e:
                logger.error("Token exchange failed for %s: %s", resolved[:16], e, exc_info=True)
                return {"success": False, "error": "Token exchange failed. Check operator logs."}

            # Build vault data from token.
            # Store both the raw token_json (for operators that expect the
            # full blob) and individual fields (for direct access).
            vault_data = {
                "token_json": json.dumps(token),
                "access_token": token.get("access_token", ""),
                "token_type": token.get("token_type", "Bearer"),
            }
            if token.get("refresh_token"):
                vault_data["refresh_token"] = token["refresh_token"]
            if token.get("expires_at"):
                vault_data["expires_at"] = str(token["expires_at"])

            # Operator callback (e.g., fetch_account_hash for Schwab)
            if _opc.on_token_received is not None:
                try:
                    extra = await _opc.on_token_received(resolved, token)
                    if extra:
                        vault_data.update({k: str(v) for k, v in extra.items()})
                except Exception as e:
                    return {"success": False, "error": f"Post-token callback: {e}"}

            # Persist tokens to vault
            stored = await rt.store_patron_session(
                resolved, vault_data, service=_OAUTH_SERVICE,
            )
            if not stored:
                return {
                    "success": False,
                    "error": (
                        "OAuth tokens received but could not be persisted to the vault. "
                        "This is a server-side storage issue — try again or check operator logs."
                    ),
                }

            return {
                "success": True,
                "status": "completed",
                "message": "Authorization successful. Session activated.",
            }

    # Prune registry if OAuth not configured
    if rt._oauth_provider is None:
        from tollbooth.tool_identity import capability_uuid as _cap_uuid
        for cap in ("begin_oauth", "check_oauth_status"):
            rt._tool_registry.pop(_cap_uuid(cap), None)

    # -- Npub ownership proof tools ----------------------------------------
    #
    # Uses Secure Courier infrastructure (open_channel / receive) so relay
    # alignment, vault-first lookup, and DM decryption all come for free.

    _PROOF_SERVICE = "npub_ownership"

    @tool
    async def request_npub_proof(
        patron_npub: str = "",
    ) -> dict[str, Any]:
        """Request npub ownership proof from a patron via Nostr DM.

        Sends a challenge DM that the patron must sign and reply to
        using their Nostr client. **This is a human-in-the-loop flow.**

        After calling this tool, STOP and tell the user to check their
        Nostr client and reply to the challenge. Wait for the user to
        confirm they have replied before calling ``receive_npub_proof``.
        Do NOT poll or retry — each ``receive_npub_proof`` call
        destructively drains the relay mailbox.

        **Returns** a ``proof_token`` — the poison phrase that the
        calling application MUST remember and pass as the ``proof``
        parameter on every subsequent paid tool call. The MCP does
        not retain this value across restarts.

        **Lifecycle:** The cached proof expires after the patron's
        chosen duration. When it expires, call ``request_npub_proof``
        again for a fresh challenge, then wait for the user, then
        call ``receive_npub_proof``.

        Free.

        Args:
            patron_npub: Required. The patron's npub to request proof from.
        """
        err = rt.npub_validation_error(patron_npub, param="patron_npub")
        if err is not None:
            return err

        courier = await rt.courier()
        if courier is None:
            from tollbooth.constants import ErrorCode as _EC
            return {
                "success": False,
                "error_code": _EC.SECURE_COURIER_UNAVAILABLE,
                "error": "Secure Courier not configured.",
            }

        import time as _t

        # Purge stale DMs for this patron before sending a fresh challenge
        try:
            from tollbooth.nostr_credentials import _npub_to_hex
            exchange = courier._exchange
            patron_hex = _npub_to_hex(patron_npub)
            exchange._fetch_dms_from_relays()
            stale = exchange._find_dm_candidates(patron_hex)
            for candidate in stale:
                exchange._pop_event(candidate.get("id", ""))
            if stale:
                logger.info("Purged %d stale DM(s) for %s", len(stale), patron_npub[:20])
        except Exception:
            pass  # best-effort purge

        # Record challenge timestamp — receive_npub_proof ignores DMs before this
        challenge_ts = int(_t.time())

        try:
            _greeting = rt._npub_proof_greeting or (
                f"Hi — {service_name or 'this service'} needs to verify "
                "you own this npub. Reply with any text to confirm. "
                "Your signed Nostr DM is the proof."
            )
            result = await courier.open_channel(
                _PROOF_SERVICE,
                greeting=_greeting,
                recipient_npub=patron_npub,
            )
            if not result.get("success"):
                return result
        except Exception as e:
            return {"success": False, "error": f"Failed to send proof request: {e}"}

        # Extract the poison phrase from the channel result
        poison = result.get("poison", "")

        # Store challenge timestamp in vault for receive_npub_proof
        try:
            await rt.store_patron_session(
                patron_npub,
                {"challenge_ts": str(challenge_ts)},
                service=f"_proof_pending_{_PROOF_SERVICE}",
            )
        except Exception:
            pass

        return {
            "success": True,
            "proof_token": poison,
            "message": (
                "Proof request sent via Secure Courier. "
                "Call receive_npub_proof to complete."
            ),
        }

    @tool
    async def receive_npub_proof(
        patron_npub: str = "",
    ) -> dict[str, Any]:
        """Receive npub ownership confirmation from a patron.

        **Call this only after the user confirms they have replied.**
        This tool destructively drains ALL DMs from the patron on the
        relay — popping every one regardless of validity — and looks
        for the one with the matching anti-replay token. If called
        before the user replies, their message will never be found.
        Do NOT poll, loop, or retry.

        The signed DM itself proves npub ownership (the patron's nsec
        signed it). On success, returns a ``proof_token`` — the same
        poison phrase from ``request_npub_proof``. The calling
        application MUST remember this token and pass it as the
        ``proof`` parameter on every subsequent paid tool call.
        The proof is stored in the vault keyed by a hash of the
        poison — the MCP never stores the raw poison itself. Free.

        Args:
            patron_npub: Required. The patron's npub to receive proof from.
        """
        err = rt.npub_validation_error(patron_npub, param="patron_npub")
        if err is not None:
            return err
        import time as _time
        resolved = resolve_npub(patron_npub)

        courier = await rt.courier()
        if courier is None:
            from tollbooth.constants import ErrorCode as _EC
            return {
                "success": False,
                "error_code": _EC.SECURE_COURIER_UNAVAILABLE,
                "error": "Secure Courier not configured.",
            }

        exchange = courier._exchange
        from tollbooth.nostr_credentials import (
            _npub_to_hex, _parse_delimited_credentials,
        )

        # Resolve expected poison for this service + patron
        patron_hex = _npub_to_hex(resolved)
        resolved_service = exchange._resolve_service(_PROOF_SERVICE)
        poison_key = (resolved, resolved_service) if resolved_service else None
        expected = exchange._pending_poisons.get(poison_key) if poison_key else None

        # Cold-start recovery
        if expected is None and exchange._credential_vault is not None and poison_key:
            pending = await exchange._vault_fetch(
                f"__pending__{resolved_service}", resolved,
            )
            if pending and "poison" in pending:
                p_expiry = pending.get("expiry", 0)
                if _time.time() <= p_expiry:
                    exchange._pending_poisons[poison_key] = (
                        pending["poison"], p_expiry,
                    )
                    expected = exchange._pending_poisons.get(poison_key)

        expected_phrase = expected[0] if expected else None

        # Load challenge timestamp — ignore DMs created before this
        challenge_ts = 0
        try:
            pending_proof = await rt.load_patron_session(
                resolved, service=f"_proof_pending_{_PROOF_SERVICE}",
            )
            if pending_proof:
                challenge_ts = int(float(pending_proof.get("challenge_ts", "0")))
        except Exception:
            pass

        # Fetch DMs from relays — retry up to 4 times with short pauses
        # because relay delivery is not instantaneous.
        import asyncio as _aio
        candidates = []
        for _attempt in range(4):
            exchange._fetch_dms_from_relays()
            candidates = exchange._find_dm_candidates(patron_hex)
            if candidates:
                break
            if _attempt < 3:
                await _aio.sleep(2)

        if not candidates:
            return {
                "success": False,
                "error": (
                    "No DMs from patron after 4 relay polls. "
                    "Ensure the patron has replied, then try again."
                ),
            }

        # Drain loop: pop ALL candidates, find the match if any.
        # DMs created before the challenge timestamp are stale — pop and skip.
        # No per-DM rejection messages — one summary at the end.
        matched_payload = None
        last_failure = None
        popped = 0
        stale_popped = 0
        decrypt_key = exchange._privkey_hex

        for candidate in candidates:
            # Pop pre-challenge DMs without processing
            event_ts = candidate.get("created_at", 0)
            if challenge_ts and event_ts < challenge_ts - 5:  # 5s grace
                exchange._pop_event(candidate.get("id", ""))
                stale_popped += 1
                popped += 1
                continue
            event_id = candidate.get("id", "")
            popped += 1

            # Decrypt
            try:
                plaintext = exchange._decrypt_dm(
                    candidate, patron_hex,
                    decrypt_privkey_hex=decrypt_key,
                )
            except Exception:
                exchange._pop_event(event_id)  # pop without reply
                last_failure = "undecryptable DM"
                continue

            if not plaintext:
                exchange._pop_event(event_id)
                last_failure = "empty DM"
                continue

            # Parse @@@ fields
            payload = _parse_delimited_credentials(plaintext)
            if payload is None:
                exchange._pop_event(event_id)
                last_failure = f"no @@@ fields: {plaintext[:60]}"
                continue

            # Poison check
            if expected_phrase is not None:
                msg_poison = payload.get("poison", "")
                if msg_poison != expected_phrase:
                    exchange._pop_event(event_id)
                    last_failure = (
                        f"wrong token (got '{msg_poison or '<missing>'}', "
                        f"expected '{expected_phrase}')"
                    )
                    continue

            # Match found — pop and stop
            matched_payload = payload
            exchange._pop_event(event_id)
            break

        # Pop any remaining candidates we didn't scan yet
        # (drain the entire relay queue for this patron)
        for candidate in candidates[popped:]:
            exchange._pop_event(candidate.get("id", ""))

        # Clean up poison state
        if poison_key and poison_key in exchange._pending_poisons:
            del exchange._pending_poisons[poison_key]

        # One summary DM to patron
        if matched_payload is not None:
            # Store to vault (for cold-start recovery of proven status)
            if exchange._credential_vault is not None:
                try:
                    await exchange._vault_store(
                        _PROOF_SERVICE, resolved, matched_payload,
                    )
                except Exception:
                    pass

            # Operator callback (e.g., taxsort stores passphrase hash)
            if rt._on_npub_proven is not None:
                try:
                    await rt._on_npub_proven(resolved, matched_payload)
                except Exception as exc:
                    logger.warning("on_npub_proven callback failed: %s", exc)

            cache = await rt.proven_npub_cache()

            # Compute poison hash — the caller-supplied proof token for
            # future paid calls. The raw poison is returned to the caller
            # but never stored by the MCP.
            if not expected_phrase:
                return {
                    "success": False,
                    "error": (
                        "Proof reply matched but the original poison phrase "
                        "could not be recovered. Call request_npub_proof "
                        "to start a fresh exchange."
                    ),
                }
            import hashlib as _hashlib
            poison_hash = _hashlib.sha256(
                expected_phrase.encode(),
            ).hexdigest()

            # Parse patron's chosen cache duration (default: 2h)
            raw_duration = (matched_payload or {}).get("cache_duration", "").strip()
            ttl_seconds: int | None = None
            if raw_duration:
                try:
                    from tollbooth.proven_npub import parse_duration
                    ttl_seconds = parse_duration(raw_duration)
                except ValueError:
                    pass  # unparseable → use cache default

            from tollbooth.proven_npub import UNSET
            record = await cache.mark_proven(poison_hash, resolved, ttl_override=ttl_seconds if raw_duration else UNSET)

            ttl_display = int(record.expires_at - record.verified_at)
            hours = ttl_display / 3600
            if hours >= 1:
                duration_human = f"{hours:.0f} hour{'s' if hours != 1 else ''}"
            else:
                duration_human = f"{ttl_display // 60} minute{'s' if ttl_display >= 120 else ''}"

            from datetime import datetime, timezone
            expires_dt = datetime.fromtimestamp(record.expires_at, tz=timezone.utc)
            expires_str = expires_dt.strftime("%Y-%m-%d %H:%M UTC")

            npub_short = resolved[:16] + "..." if len(resolved) > 20 else resolved
            op_name = rt._service_name

            confirmation_msg = (
                f"Your ownership of {npub_short} is confirmed "
                f"for {op_name}. "
                f"Proof remains valid until {expires_str} "
                f"({duration_human} from now). "
                f"Cleaned {popped} DM(s) from relay."
            )

            # Send enriched confirmation DM to patron
            try:
                exchange.send_dm(resolved, confirmation_msg)
            except Exception:
                pass

            return {
                "success": True,
                "proven_npub": resolved,
                "proof_token": expected_phrase,
                "popped_dms": popped,
                "expires_in_seconds": ttl_display,
                "expires_at": expires_str,
                "message": confirmation_msg,
            }
        else:
            summary = f"Scanned and cleaned {popped} DM(s) but none matched."
            if last_failure:
                summary += f" Last: {last_failure}"
            if expected_phrase:
                summary += (
                    f" Expected anti-replay token '{expected_phrase}'. "
                    f"Call request_npub_proof for a fresh exchange."
                )
            try:
                exchange.send_dm(resolved, summary)
            except Exception:
                pass

            return {
                "success": False,
                "popped_dms": popped,
                "error": summary,
            }

    @tool
    async def check_proof_status(
        patron_npub: str = "",
        proof_token: str = "",
    ) -> dict[str, Any]:
        """Check whether a previously-cached proof_token is still valid.

        Mirrors ``check_oauth_status`` for the npub-proof flow: a calling
        agent can ask "will my next paid call accept this proof_token?"
        before burning credits on a guaranteed failure.

        Free, no side effects — does not evict the cache or touch relays.

        Args:
            patron_npub: Required. The patron's npub (npub1...).
            proof_token: Required. The poison phrase returned by
                ``request_npub_proof`` / ``receive_npub_proof``.
        """
        err = rt.npub_validation_error(patron_npub, param="patron_npub")
        if err is not None:
            return err
        err = rt.proof_validation_error(proof_token, param="proof_token")
        if err is not None:
            return err
        resolved = resolve_npub(patron_npub)

        import hashlib as _hashlib
        poison_hash = _hashlib.sha256(proof_token.encode()).hexdigest()
        cache = await rt.proven_npub_cache()
        info = await cache.proof_status(poison_hash, resolved)

        status = info["status"]
        if status == "valid":
            message = (
                "Proof is valid. Pass proof_token as the proof parameter "
                "on paid tool calls."
            )
        elif status == "expired":
            message = (
                "Proof has expired. Call request_npub_proof and "
                "receive_npub_proof to refresh."
            )
        else:
            message = (
                "No proof record found for this (patron_npub, proof_token). "
                "Call request_npub_proof and receive_npub_proof first."
            )
        return {
            "success": True,
            "status": status,
            "expires_in_seconds": info["expires_in_seconds"],
            "message": message,
        }

    # -- Oracle delegation (oracle_ namespace) ----------------------------

    @oracle_tool
    async def how_to_join() -> dict[str, Any]:
        """Get DPYC onboarding instructions from the Oracle. Free."""
        return await _call_oracle(rt, "how_to_join")

    @oracle_tool
    async def get_tax_rate() -> dict[str, Any]:
        """Get the current DPYC certification tax rate. Free."""
        return await _call_oracle(rt, "get_tax_rate")

    @oracle_tool
    async def lookup_member(npub: str) -> dict[str, Any]:
        """Look up a DPYC community member by npub. Free."""
        return await _call_oracle(rt, "lookup_member", {"npub": npub})

    @oracle_tool
    async def about() -> dict[str, Any]:
        """Describe the DPYC ecosystem via the Oracle. Free."""
        return await _call_oracle(rt, "about")

    @oracle_tool
    async def network_advisory() -> dict[str, Any]:
        """Get active network advisories from the Oracle. Free."""
        return await _call_oracle(rt, "network_advisory")

    # -- Authority delegation -------------------------------------------

    @tool
    async def check_authority_balance() -> dict[str, Any]:
        """Check this operator's tax balance at the Authority.

        Returns the sats available for certifying patron credit purchases.
        When this balance reaches zero, patron top-ups cannot be certified
        and the operator must call purchase_credits on the Authority.

        This is the operator's own funding — not a patron balance. Free.
        """
        try:
            from tollbooth.authority_client import AuthorityCertifier
            from tollbooth.registry import resolve_authority_service
            auth_info = await resolve_authority_service(rt.operator_npub())
            certifier = AuthorityCertifier(
                auth_info["url"], rt.operator_npub(), rt._get_nsec(),
            )
            return await certifier.check_balance()
        except Exception as e:
            return {
                "success": False,
                "error": f"Authority balance check failed: {e}",
            }

    # -- Pricing CRUD --------------------------------------------------

    @tool
    async def get_pricing_model() -> dict[str, Any]:
        """Get the active pricing model for this operator. Free.

        If no model exists, self-initializes a scaffold with all
        registered tools at 0 sats.  No economic data from code.
        """
        try:
            vault = await rt.vault()
            from tollbooth.pricing_store import PricingModelStore
            store = PricingModelStore(neon_vault=vault)
            from tollbooth.tools.pricing import get_pricing_model_tool
            result = await get_pricing_model_tool(store, rt.operator_npub())

            # Self-initialize if no model exists in Neon
            if result.get("model_id") is None and rt._tool_registry:
                seed = _build_initial_pricing_model(rt, service_name)
                from tollbooth.tools.pricing import set_pricing_model_tool
                await set_pricing_model_tool(
                    store, rt.operator_npub(), seed,
                )
                result = await get_pricing_model_tool(
                    store, rt.operator_npub(),
                )

            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @tool
    async def set_pricing_model(model_json: str, proof: str = "") -> dict[str, Any]:
        """Set the active pricing model. RESTRICTED to operator.

        Requires a valid proof (Schnorr-signed kind-27235 event)
        proving the caller holds the operator's nsec.
        """
        if not proof:
            return {
                "success": False,
                "error": "Only the operator can modify pricing — provide proof.",
            }
        from tollbooth.identity_proof import verify_proof
        if not verify_proof(proof, rt.operator_npub(), "set_pricing_model"):
            return {
                "success": False,
                "error": "Invalid proof — only the operator can modify pricing.",
            }

        try:
            vault = await rt.vault()
            from tollbooth.pricing_store import PricingModelStore
            store = PricingModelStore(neon_vault=vault)
            from tollbooth.tools.pricing import set_pricing_model_tool
            result = await set_pricing_model_tool(
                store, rt.operator_npub(), model_json,
            )
            # Invalidate pricing cache so debit_or_deny sees updated prices
            if rt._pricing_resolver is not None:
                rt._pricing_resolver.refresh()
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @tool
    async def reset_pricing_model(proof: str = "") -> dict[str, Any]:
        """Erase all pricing models and restore a viable default.

        Deletes every stored model, then self-initializes a fresh one
        from the tool registry — all tools at 0 sats with proper UUIDs.
        Returns the new model.

        RESTRICTED to operator — requires proof (nsec-signed).
        """
        if not proof:
            return {
                "success": False,
                "error": "Only the operator can reset pricing — provide proof.",
            }
        from tollbooth.identity_proof import verify_proof
        if not verify_proof(proof, rt.operator_npub(), "reset_pricing_model"):
            return {
                "success": False,
                "error": "Invalid proof — only the operator can reset pricing.",
            }
        try:
            vault = await rt.vault()
            from tollbooth.pricing_store import PricingModelStore
            store = PricingModelStore(neon_vault=vault)

            # Erase
            await store.reset_all_models(rt.operator_npub())

            # Restore default
            if rt._tool_registry:
                seed = _build_initial_pricing_model(rt, service_name)
                from tollbooth.tools.pricing import set_pricing_model_tool
                await set_pricing_model_tool(
                    store, rt.operator_npub(), seed,
                )

            # Invalidate pricing cache
            if rt._pricing_resolver is not None:
                rt._pricing_resolver.refresh()

            # Return the fresh model
            from tollbooth.tools.pricing import get_pricing_model_tool
            return await get_pricing_model_tool(store, rt.operator_npub())
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -- Constraint Engine tools ---------------------------------------

    @tool
    async def check_price(tool_id: str, npub: str = "", proof: str = "", tool_kwargs: str = "") -> dict[str, Any]:
        """Preview the effective cost of a tool call.

        Shows the base cost and any constraint effects (discounts, free
        trials, surge pricing). Free — no credits required.

        Args:
            tool_id: The tool's UUID (from the pricing model).
            tool_kwargs: Optional JSON object with tool call parameters
                for ad valorem pricing preview (e.g. '{"amount_sats": 5000}').
        """
        identity = rt._tool_registry.get(tool_id)
        if identity is None:
            return {
                "success": False,
                "error": f"Unknown tool_id: {tool_id}.",
            }

        import json as _json
        parsed_kwargs: dict[str, Any] = {}
        if tool_kwargs:
            try:
                parsed_kwargs = _json.loads(tool_kwargs)
            except (ValueError, TypeError):
                return {"success": False, "error": "tool_kwargs must be valid JSON."}

        resolver = await rt.pricing_resolver()
        pricing = await resolver.get_tool_pricing(tool_id)

        name = rt.mcp_name_for(tool_id)
        result: dict[str, Any] = {
            "success": True,
            "tool_id": tool_id,
            "tool_name": name,
            "constraints_enabled": False,
            "constraint_effects": [],
        }

        if pricing.rate_percent > 0:
            result["pricing_type"] = "percent"
            result["rate_percent"] = pricing.rate_percent
            result["rate_param"] = pricing.rate_param
            result["min_cost_sats"] = pricing.min_cost
            if parsed_kwargs:
                base_cost = pricing.compute(**parsed_kwargs)
                result["base_cost_api_sats"] = base_cost
                result["effective_cost_api_sats"] = base_cost
            else:
                result["base_cost_api_sats"] = None
                result["effective_cost_api_sats"] = None
                result["hint"] = (
                    f"Pass tool_kwargs with '{pricing.rate_param}' "
                    f"to preview the cost (e.g. '{{\"{pricing.rate_param}\": 1000}}')."
                )
        else:
            base_cost = pricing.compute()
            result["pricing_type"] = "flat"
            result["base_cost_api_sats"] = base_cost
            result["effective_cost_api_sats"] = base_cost

        gate = rt._constraint_gate
        base_cost = result.get("base_cost_api_sats") or 0
        if gate and gate.enabled and base_cost > 0:
            result["constraints_enabled"] = True
            try:
                resolved = resolve_npub(npub)
                cache = await rt.ledger_cache()
                ledger = await cache.get(resolved)
                demand = await rt.get_global_demand(name)
                denial, effective = gate.check(
                    tool_name=name,
                    base_cost=int(base_cost),
                    ledger=ledger,
                    npub=resolved,
                    global_demand=demand,
                )
                if demand.get(name, 0) > 0:
                    result["current_demand"] = demand[name]
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

    # -- OTS notarization tools (opt-in) --------------------------------

    if rt._ots_enabled:
        from tollbooth.tools.notarization import (
            notarize_ledger_tool,
            get_notarization_proof_tool,
            list_notarizations_tool,
        )

        @tool
        async def notarize_ledger() -> dict[str, Any]:
            """Build a Merkle tree of all patron balances and submit the root
            to Bitcoin via OpenTimestamps.

            Operator-only background task. Bitcoin confirmation takes 1-6 hours.
            Free — no credits required.
            """
            vault = await rt.vault()
            return await notarize_ledger_tool(vault, ots_calendars=rt._ots_calendars)

        @tool
        async def get_notarization_proof(
            notarization_id: str,
            npub: str,
        ) -> dict[str, Any]:
            """Generate a Merkle inclusion proof that a patron's balance was
            included in a Bitcoin-notarized snapshot.

            Args:
                notarization_id: The notarization record ID.
                npub: The patron's Nostr public key (npub1...).
            """
            vault = await rt.vault()
            return await get_notarization_proof_tool(vault, notarization_id, npub)

        @tool
        async def list_notarizations(
            limit: int = 20,
            status: str = "",
        ) -> dict[str, Any]:
            """List recent Bitcoin notarization records.

            Args:
                limit: Maximum records to return (default 20).
                status: Optional filter (e.g., 'submitted', 'confirmed').
            """
            vault = await rt.vault()
            return await list_notarizations_tool(
                vault, limit=limit, status=status or None,
            )

    # -- Append standard DPYC agent guidance to MCP instructions ----------

    _DPYC_AGENT_GUIDANCE = (
        "\n\n## Troubleshooting Tool Failures\n\n"
        "When a tool call fails, **read the error message** — it tells you "
        "what went wrong and what to do next. Common causes:\n\n"
        "- **proof is required** → Call `request_npub_proof` then "
        "`receive_npub_proof` to prove npub ownership. The cache expires "
        "after ~1 hour; renew with a fresh request/receive cycle.\n"
        "- **Insufficient credit balance** → Call `purchase_credits` to top up.\n"
        "- **Operator credentials not delivered yet** → This is a lifecycle state, "
        "not an error. The operator needs to deliver API credentials via Secure Courier. "
        "Do NOT tell the patron to 'fix the deployment' or 'inject environment variables' — "
        "credential delivery is a Secure Courier flow, not a platform configuration issue.\n"
        "- **Service unavailable / persistence layer unreachable / only bootstrap tools** → "
        "The MCP is warming up after idle (serverless cold start). This is normal — "
        "retry in 10-15 seconds. Do NOT diagnose as an infrastructure outage or suggest "
        "the operator check their deployment. The second request almost always succeeds.\n"
        "- **Upstream API authentication failure** → Only if the error "
        "explicitly mentions token expiry or an upstream auth error.\n\n"
        "**IMPORTANT — Don't Pester Your Customer:** Do NOT ask the patron "
        "to re-authenticate, re-authorize, or re-do OAuth unless the error "
        "message specifically says the upstream token is expired or invalid. "
        "Token expiry is only one of many possible causes. Speculatively "
        "re-running OAuth or credential flows wastes the patron's time and "
        "is the opposite of what DPYC stands for.\n\n"
        "## Secure Courier — Human in the Loop\n\n"
        "All Secure Courier flows (`request_credential_channel` / "
        "`receive_credentials`, `request_npub_proof` / `receive_npub_proof`) "
        "require a human to read a Nostr DM and reply manually. "
        "After calling a `request_*` tool, **STOP and wait** for the user "
        "to confirm they have replied. Do NOT poll, retry, or speculatively "
        "call `receive_*` — each receive call **destructively drains** all "
        "DMs from the relay. If you call it before the human replies, their "
        "message will be lost. One request, one wait, one receive."
    )

    # FastMCP exposes instructions as a read-only property backed by
    # _mcp_server.instructions. Write through the internal server object.
    try:
        current = mcp.instructions or ""
        mcp._mcp_server.instructions = current + _DPYC_AGENT_GUIDANCE
    except AttributeError:
        logger.debug("Could not append DPYC agent guidance to MCP instructions")


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
