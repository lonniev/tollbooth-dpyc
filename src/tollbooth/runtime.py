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
        tool_registry={MY_TOOL_UUID: ToolIdentity(tool_id=MY_TOOL_UUID, capability="my_tool", category="read", intent="...")},
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
import logging
import os
import signal
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tollbooth.tool_identity import ToolIdentity

from tollbooth.identity_proof import require_proof
from tollbooth.tool_identity import capability_uuid

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
        self._coupons_vault: Any | None = None  # lazy CouponsVault
        self._operator_npub: str | None = None
        self._nsec: str | None = None
        self._reconciled_npubs: set[str] = set()  # dedup auto-reconciliation
        self._proven_npub_ttl = proven_npub_ttl_seconds
        self._proven_npub_cache: Any | None = None  # lazy ProvenNpubCache
        self._npub_proof_field = npub_proof_field
        self._npub_proof_greeting = npub_proof_greeting
        self._on_npub_proven = on_npub_proven  # async callback(npub, payload)
        self._oauth_provider = oauth_provider  # OAuthProviderConfig or None
        # Claim-check async jobs (see tollbooth/async_jobs.py)
        self._job_runners: dict[str, Callable[..., Any]] = {}
        self._async_jobs_purge_last: float = 0.0  # monotonic, rate-limits purges
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

    def runtime_name(self, capability: str) -> str:
        """Resolve the full runtime tool name from a capability seed.

        The runtime name is what crosses every server boundary — the wire,
        the pricing model, identity proofs, audit logs. The capability is
        the Python function identifier and stays internal to this process.
        """
        return self.mcp_name_for(capability_uuid(capability))

    async def require_caller_proof(
        self,
        npub: str,
        proof: str,
        capability: str,
    ) -> dict[str, Any] | None:
        """Canonical caller-identity gate for free / bootstrap standard tools.

        Wraps ``identity_proof.require_proof`` with this runtime's proven-npub
        cache and the runtime name for ``capability``. Paid tools are gated by
        ``debit_or_deny``; this helper is for the free tools that still need
        to assert "the caller is who they claim to be" (check_balance,
        purchase_credits, account_statement, etc.).

        Returns ``None`` on success (caller proceeds) or a structured error
        dict to return verbatim.
        """
        return await require_proof(
            npub, proof, self.runtime_name(capability),
            proven_cache=await self.proven_npub_cache(),
        )

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

    async def coupons_vault(self) -> Any:
        """Lazy accessor for the :class:`CouponsVault` (requires vault)."""
        if self._coupons_vault is None:
            vault = await self.vault()
            from tollbooth.coupons import CouponsVault
            self._coupons_vault = CouponsVault(neon_vault=vault)
        return self._coupons_vault

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

        identity = self._tool_registry.get(tool_id)

        # ── Identity ──────────────────────────────────────────
        if identity is None:
            return {
                "success": False,
                "error_code": ErrorCode.TOOL_NOT_REGISTERED,
                "error": f"Unknown tool '{tool_id}' — not registered with this operator.",
            }

        name = self.mcp_name_for(tool_id)      # the ONE external identifier
        category = identity.category           # set in code, never by the pricing model

        # ── Proof verification ────────────────────────────────
        # Single gate for both tactics (cached poison and inline Schnorr).
        # Restricted tools verify against the operator's npub — the proof
        # must be signed by the operator, regardless of which caller npub
        # is acting. The access check below enforces caller == operator.
        # The proof's u-tag matches the runtime name (`name`) — capability
        # is a Python-function-identifier seed that never crosses the wire.
        if npub:
            try:
                resolved = resolve_npub(npub)
            except ValueError as e:
                return {
                    "success": False,
                    "error_code": ErrorCode.NPUB_INVALID,
                    "error": str(e),
                }

            proof_npub = self.operator_npub() if category == "restricted" else resolved
            if err := await require_proof(
                proof_npub,
                proof,
                name,
                proven_cache=await self.proven_npub_cache(),
            ):
                return err

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
        cost, price_err = await self._resolve_pricing(tool_id, name, category, tool_kwargs)
        if price_err is not None:
            return price_err

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
        # The tool's per-tool constraint chain (if any) walks the
        # price step by step.  Steps can deny access (temporal
        # windows, supply caps) or transform the price (discounts,
        # surcharges, even drive it negative into a credit).
        # Without an npub, constraints cannot evaluate.
        effective_cost = cost
        consumed_coupon_ids: list[str] = []
        if npub and category != "free":
            effective_cost, consumed_coupon_ids, denial = await self._evaluate_constraints(
                tool_id, name, npub, cost, proof,
            )
            if denial is not None:
                return denial

        # ── Billing ───────────────────────────────────────────
        return await self._apply_billing(npub, name, effective_cost, consumed_coupon_ids)

    async def _resolve_pricing(
        self,
        tool_id: str,
        name: str,
        category: str,
        tool_kwargs: dict[str, Any] | None,
    ) -> tuple[int, dict[str, Any] | None]:
        """Pricing stage of ``debit_or_deny``: resolve the base cost.

        Returns ``(cost, None)`` to proceed, or ``(0, error_dict)`` to deny.
        Code says "free" → cost is 0, Neon not consulted. Everything else
        requires Neon, a model, and an explicit price.
        """
        from tollbooth.constants import ErrorCode

        if category == "free":
            return 0, None

        resolver = await self.pricing_resolver()
        await resolver._ensure_fresh()
        if not resolver.neon_available:
            if resolver.last_error_permanent:
                return 0, {
                    "success": False,
                    "error_code": ErrorCode.PERSISTENCE_MISCONFIGURED,
                    "error": (
                        "The operator's database rejected the pricing-model "
                        "query with a permanent error — this will NOT resolve "
                        "by retrying. The operator must repair the database "
                        f"(detail: {resolver.last_error_summary}). "
                        "Free tools remain available."
                    ),
                    "next_steps": [
                        "Notify the operator — this is an operator-side "
                        "database repair, not a patron-actionable error"
                    ],
                }
            return 0, {
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
            return 0, {
                "success": False,
                "error_code": ErrorCode.TOOL_NOT_PRICED,
                "error": (
                    f"Tool '{name}' is not yet in the pricing model. "
                    f"Add it to the pricing model before use."
                ),
            }
        if not await resolver.is_priced(tool_id):
            return 0, {
                "success": False,
                "error_code": ErrorCode.TOOL_NOT_PRICED,
                "error": (
                    f"Tool '{name}' has not been priced yet (TBD). "
                    f"Set a price in the pricing model before use."
                ),
            }
        pricing = await resolver.get_tool_pricing(tool_id)
        return pricing.compute(**(tool_kwargs or {})), None

    async def _evaluate_constraints(
        self,
        tool_id: str,
        name: str,
        npub: str,
        cost: int,
        proof: str,
    ) -> tuple[int, list[str], dict[str, Any] | None]:
        """Constraint stage of ``debit_or_deny``.

        Walks the tool's per-tool constraint chain (if any): steps can deny
        access (temporal windows, supply caps) or transform the price
        (discounts, surcharges, even drive it negative into a credit).
        Returns ``(effective_cost, consumed_coupon_ids, denial_or_None)``. A
        chain-evaluation exception falls through with the base cost — it never
        blocks the call. Caller gates this on ``npub and category != "free"``.
        """
        from tollbooth.constants import ErrorCode

        effective_cost = cost
        consumed_coupon_ids: list[str] = []
        try:
            resolver = await self.pricing_resolver()
            chain = await resolver.get_chain(tool_id)
            if chain:
                cache = await self.ledger_cache()
                ledger = await cache.get(npub)
                demand = await self.get_global_demand(name)
                gate = self._constraint_gate
                if gate is None:
                    from tollbooth.constraints.gate import ConstraintGate
                    gate = ConstraintGate()
                    gate.attach_resolver(resolver)
                    self._constraint_gate = gate
                # Pre-load coupon redemptions for any coupon steps so
                # the chain walk stays synchronous.
                coupon_map = None
                coupon_ids = gate.collect_coupon_ids(chain)
                if coupon_ids:
                    try:
                        cv = await self.coupons_vault()
                        redemptions = await cv.fetch_redemptions_for_chain(
                            npub, coupon_ids,
                        )
                        from tollbooth.coupons import CouponRedemptionMap
                        coupon_map = CouponRedemptionMap(
                            entries=tuple(redemptions.items()),
                        )
                    except Exception as ce:
                        logger.warning(
                            "Coupon redemption pre-load failed for %s: %s",
                            name, ce,
                        )
                denial, effective_signed, consumed_coupon_ids = gate.evaluate_chain(
                    chain=chain,
                    tool_name=name,
                    base_cost=cost,
                    ledger=ledger,
                    npub=npub,
                    global_demand=demand,
                    proof=proof,
                    coupon_redemptions=coupon_map,
                )
                if denial is not None:
                    denial["error_code"] = ErrorCode.CONSTRAINT_DENIED
                    return cost, [], denial
                effective_cost = effective_signed
        except Exception as exc:
            logger.warning("Constraint evaluation failed for %s: %s", name, exc)
            # Fall through with base cost — don't block the call
        return effective_cost, consumed_coupon_ids, None

    async def _apply_billing(
        self,
        npub: str,
        name: str,
        effective_cost: int,
        consumed_coupon_ids: list[str],
    ) -> dict[str, Any] | int:
        """Billing stage of ``debit_or_deny``: no-charge / credit / debit.

        Returns the (signed) cost to proceed, or an INSUFFICIENT_BALANCE error.
        """
        from tollbooth.constants import ErrorCode

        async def _burn_consumed_coupons() -> None:
            """Burn one use per consumed coupon id.  Best-effort —
            errors are logged but don't unwind the debit (the chain
            already evaluated allowed=True and the patron got the
            discount; a transient burn failure is recoverable on
            replay because the same use_count would just re-burn)."""
            if not consumed_coupon_ids or not npub:
                return
            try:
                cv = await self.coupons_vault()
            except Exception as exc:
                logger.warning("coupons_vault unavailable for burn: %s", exc)
                return
            for cid in consumed_coupon_ids:
                try:
                    await cv.burn_use(cid, npub)
                except Exception as exc:
                    logger.warning(
                        "burn_use failed (coupon=%s patron=%s): %s",
                        cid, npub[:20], exc,
                    )

        # ── No charge ─────────────────────────────────────────
        if effective_cost == 0:
            if npub:
                cache = await self.ledger_cache()
                ledger = await cache.get(npub)
                ledger.debit(name, 0)
                cache.mark_dirty(npub)
            await _burn_consumed_coupons()
            return 0

        # ── Credit (chain drove price negative) ───────────────
        # The constraint chain transformed the toll into a credit
        # paid to the patron.  Skip the balance / insufficient-funds
        # gate (you can't be short of money you're about to receive)
        # and add the credit to the patron's balance directly.
        if effective_cost < 0:
            cache = await self.ledger_cache()
            ledger = await cache.get(npub)
            credit_sats = abs(effective_cost)
            ledger.credit_deposit(credit_sats, f"chain_credit:{name}")
            cache.mark_dirty(npub)
            await _burn_consumed_coupons()
            return effective_cost  # signed — caller can detect credit case

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
        await _burn_consumed_coupons()
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

    def _patron_storage_service(self, service: str | None) -> str:
        """Resolve the vault service name for patron-credential CRUD.

        Resolution order:
        1. Explicit ``service`` argument (caller knows what they want).
        2. ``patron_credential_service`` (set when the operator declared
           a ``patron_credential_template`` — set-once secrets like
           eXcalibur API keys, TheBrain tokens).
        3. The OAuth provider's ``service_name`` if configured.  This
           is the natural home for patron *preferences* on OAuth-only
           operators (account_hash, default_brain_id, etc.) — they
           live alongside the tokens and survive refresh cycles via
           ``restore_oauth_session``.

        Returns ``""`` if none resolve.
        """
        if service:
            return service
        if self.patron_credential_service:
            return self.patron_credential_service
        if self._oauth_provider is not None:
            return self._oauth_provider.service_name
        return ""

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

        On OAuth-only operators (no ``patron_credential_template``),
        falls back to writing into the OAuth provider's vault entry
        so per-patron preferences (e.g. ``account_hash``,
        ``default_brain_id``) live alongside the tokens.
        """
        svc = self._patron_storage_service(service)
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
        svc = self._patron_storage_service(service)
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
        svc = self._patron_storage_service(service)
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
        svc = self._patron_storage_service(service)
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
    # Claim-check async jobs
    # ------------------------------------------------------------------
    #
    # A slow tool (LLM round-trip, web-search generation) returns a claim
    # check instead of the end item; the work runs as a concurrent asyncio
    # task in this process and the result persists in the operator's Neon.
    # The operator defines a companion tool that delegates to
    # fetch_async_job() to redeem the claim. The wheel exposes nothing on
    # the wire — these are library helpers only.

    _ASYNC_JOB_MAX_ATTEMPTS = 3
    _ASYNC_JOBS_PURGE_INTERVAL_SECONDS = 300

    def register_job_runner(self, kind: str, runner: Callable[..., Any]) -> None:
        """Map a job kind to the async callable that performs the work.

        Operators register runners at startup. Runners receive the job's
        persisted params as kwargs and must return a JSON-serializable
        dict — either the output content itself or state the operator can
        use to find the output elsewhere (e.g. ``{"entry_id": ...}``).
        Registration by name (not closure) is what lets a fresh container
        resume a job orphaned by a serverless recycle.
        """
        self._job_runners[kind] = runner

    async def async_job_store(self) -> Any:
        """Build an AsyncJobStore over the operator's vault."""
        from tollbooth.async_jobs import AsyncJobStore
        vault = await self.vault()
        return AsyncJobStore(vault)

    async def start_async_job(
        self,
        kind: str,
        npub: str,
        params: dict[str, Any],
        *,
        tool_id: str,
        max_runtime_seconds: int,
        result_ttl_seconds: int,
    ) -> dict[str, Any]:
        """Persist a job, kick it off concurrently, return its claim check.

        Call from inside a ``@paid_tool`` body — the fee was just assessed
        for *requesting* the work (that is when the operator incurs the
        expense). The durations are coded by the calling tool:
        ``max_runtime_seconds`` is how long one attempt may take (and the
        staleness threshold for watchdog recovery), ``result_ttl_seconds``
        is how long a finished result is kept before it expires.
        """
        if kind not in self._job_runners:
            raise RuntimeError(f"No job runner registered for kind {kind!r}")
        npub = resolve_npub(npub)
        store = await self.async_job_store()
        claim = await store.create(
            npub=npub,
            kind=kind,
            tool_id=tool_id,
            params=params,
            max_runtime_seconds=max_runtime_seconds,
            result_ttl_seconds=result_ttl_seconds,
        )
        import asyncio
        asyncio.create_task(self._run_job(claim))
        return {
            "success": True,
            "claim_check": claim,
            "status": "pending",
            "poll_after_seconds": 3,
        }

    async def _run_job(self, claim: str) -> None:
        """Claim and execute a job; persist its outcome.

        Survives the tool response because Horizon containers are
        long-lived SSE servers — same mechanism as the fire-and-forget
        notarization above. Safe to spawn redundantly: the atomic
        claim_for_run ensures exactly one container runs an attempt.
        """
        import asyncio
        try:
            store = await self.async_job_store()
            while True:
                job = await store.claim_for_run(claim)
                if job is None:
                    return  # another container owns it, or it already finished
                kind = job["kind"]
                attempt = job["attempts"]
                runner = self._job_runners.get(kind)
                if runner is None:
                    await store.fail(
                        claim,
                        f"No runner registered for job kind {kind!r}. "
                        "The operator must register it at startup.",
                    )
                    await self.rollback_debit(
                        job["tool_id"], job["npub"], tool_kwargs=job["params"],
                    )
                    return
                try:
                    result = await runner(**job["params"])
                except Exception as exc:
                    # Generic message to the patron (an exception string can
                    # carry anything); full detail to operator logs only —
                    # same posture as paid_tool's catch_errors path. Domain
                    # errors a patron *should* see belong in the runner's
                    # returned dict, which flows through complete().
                    logger.error(
                        "Async job %s (%s) attempt %d/%d failed: %s",
                        claim, kind, attempt, self._ASYNC_JOB_MAX_ATTEMPTS,
                        exc, exc_info=True,
                    )
                    if attempt >= self._ASYNC_JOB_MAX_ATTEMPTS:
                        await store.fail(
                            claim, "Job execution failed. Check operator logs.",
                        )
                        await self.rollback_debit(
                            job["tool_id"], job["npub"], tool_kwargs=job["params"],
                        )
                        return
                    await store.release(claim)
                    await asyncio.sleep(2 * attempt)
                    continue
                if not isinstance(result, dict):
                    result = {"result": result}
                await store.complete(claim, result)
                return
        except Exception as exc:
            # Store-level failure (Neon unreachable mid-job). Leave the row
            # as-is — the watchdog reclaims it once max_runtime elapses.
            logger.warning("Async job %s aborted: %s", claim, exc)

    async def fetch_async_job(self, claim: str, npub: str) -> dict[str, Any]:
        """Redeem a claim check — the body of an operator's companion tool.

        Every lifecycle state returns guidance, not an error. Doubles as
        the watchdog: a pending or stalled job found here is re-kicked on
        this (alive) container, recovering work orphaned by a serverless
        recycle. No cron required — the patron's own polling drives
        recovery.
        """
        import asyncio
        npub = resolve_npub(npub)
        store = await self.async_job_store()
        self._fire_and_forget_purge_expired_jobs(store)
        job = await store.get(claim, npub)
        if job is None or job["expired"]:
            return {
                "success": True,
                "status": "expired",
                "next_steps": (
                    "This claim check is unknown or its result has expired. "
                    "Start a new request to get a fresh claim check."
                ),
            }
        if job["status"] == "done":
            return {"success": True, "status": "done", "result": job["result"]}
        if job["status"] == "error":
            return {
                "success": True,
                "status": "error",
                "error": job["error"],
                "refunded": True,
                "next_steps": "The fee was refunded. Start a new request to retry.",
            }
        # pending, or running — re-kick when no live container owns it
        # (pending: creator died before claiming, or released for retry;
        # stalled: running past its own max_runtime). claim_for_run's
        # atomicity makes a redundant spawn harmless.
        recovered = False
        if job["status"] == "pending" or job["stalled"]:
            asyncio.create_task(self._run_job(claim))
            recovered = job["stalled"]
        response: dict[str, Any] = {
            "success": True,
            "status": "running",
            "poll_after_seconds": 3,
        }
        if recovered:
            response["recovered"] = True
        return response

    def _fire_and_forget_purge_expired_jobs(self, store: Any) -> None:
        """Opportunistic cleanup, rate-limited like the OTS check."""
        import asyncio
        import time as _time

        now = _time.monotonic()
        if (now - self._async_jobs_purge_last) < self._ASYNC_JOBS_PURGE_INTERVAL_SECONDS:
            return
        self._async_jobs_purge_last = now

        async def _purge() -> None:
            try:
                purged = await store.purge_expired()
                if purged:
                    logger.info("Purged %d expired async job(s)", purged)
            except Exception:
                pass

        asyncio.create_task(_purge())

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
        if identity.pricing_hint_multipliers:
            # Serialize the frozen tuple back to nested-dict for JSON.
            entry["multipliers"] = {
                param: {k: float(v) for k, v in lookup}
                for param, lookup in identity.pricing_hint_multipliers
            }
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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register all standard DPYC tools on a FastMCP app.

    Call once at module level. Operators only write domain-specific tools.

    Returns the slug-prefixed ``@tool`` decorator so the operator can use
    the same one for its own paid tools — no need to repeat ``slug`` in a
    separate ``make_slug_tool`` call.

    Args:
        mcp: The FastMCP app instance.
        slug: Tool name prefix (e.g., ``"weather"``).
        rt: The OperatorRuntime instance.
        service_name: Service name for service_status (e.g., ``"tollbooth-sample"``).
        service_version: Version string for service_status.

    Returns:
        The ``@tool`` decorator bound to ``slug`` — apply to operator-defined
        functions so they register as ``<slug>_<function_name>``.
    """
    from tollbooth.slug_tools import make_slug_tool
    tool = make_slug_tool(mcp, slug)
    # Oracle-delegated tools live under the operator's namespace too —
    # e.g. `schwab_oracle_about` rather than a bare `oracle_about`. This
    # keeps every wire-exposed name under a single slug per operator so
    # downstream clients can derive the slug unambiguously.
    oracle_tool = make_slug_tool(mcp, f"{slug}_oracle")
    rt._slug = slug
    rt._mcp_name_cache.clear()  # invalidate any cached names

    # -- Credit tools --------------------------------------------------

    @tool
    async def check_balance(npub: str, proof: str) -> dict[str, Any]:
        """Check a patron's credit balance at this operator.

        This is the patron's spending balance — credits purchased via
        Lightning for tool calls at this operator. For the operator's
        own balance at the Authority (needed to certify patron purchases),
        use authority_check_balance instead.

        Free — no credits required. Proof of npub ownership is required
        to prevent anyone-with-the-registry from enumerating balances.

        Args:
            npub: The Nostr public key (npub1...) whose balance to check.
            proof: A kind-27235 Nostr event signed by npub for this tool.
        """
        if err := await rt.require_caller_proof(npub, proof, "check_balance"):
            return err
        try:
            npub = resolve_npub(npub)
            cache = await rt.ledger_cache()
        except ValueError as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        return await credits.check_balance_tool(cache, npub)

    @tool
    async def purchase_credits(npub: str, proof: str, amount_sats: int = 1000) -> dict[str, Any]:
        """Buy credits via Bitcoin Lightning.

        Creates a Lightning invoice. Pay it with any Lightning wallet,
        then call check_payment to confirm. Proof of npub ownership is
        required so credits land in the correct ledger.

        Free — no credits required to call.

        Args:
            npub: The Nostr public key (npub1...) the credits will fund.
            proof: A kind-27235 Nostr event signed by npub for this tool.
            amount_sats: Satoshis to purchase (default 1000).
        """
        if err := await rt.require_caller_proof(npub, proof, "purchase_credits"):
            return err
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
    async def check_payment(invoice_id: str, npub: str, proof: str) -> dict[str, Any]:
        """Check the payment status of a Lightning invoice.

        Call after paying the invoice from purchase_credits.
        Free — no credits required. Proof of npub ownership is required
        to prevent credit-grant front-running by an observer of the
        invoice ID.

        Args:
            invoice_id: The invoice ID returned by purchase_credits.
            npub: The Nostr public key (npub1...) that purchased the invoice.
            proof: A kind-27235 Nostr event signed by npub for this tool.
        """
        if err := await rt.require_caller_proof(npub, proof, "check_payment"):
            return err
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
    async def restore_credits(invoice_id: str, patron_npub: str, proof: str) -> dict[str, Any]:
        """Credit a patron's ledger from a BTCPay-settled invoice.

        **RESTRICTED to the operator** — the operator owns the books and is
        the only party who can issue a manual credit grant. Patrons who
        believe they paid but never got credits must escalate to the
        operator's support, who then invokes this tool on their behalf.

        Use cases: cold-start vault races during check_payment, ncred
        delivery hiccups, patrons closing Top-Off sheets before settle,
        any infrastructure incident that left an invoice settled at BTCPay
        but uncredited on the operator's ledger.

        Idempotent — if the invoice is already credited (in the patron's
        ``credited_invoices``), returns success with credits_granted=0.

        Args:
            invoice_id: The BTCPay invoice ID to verify and credit.
            patron_npub: The patron's npub whose ledger receives the grant.
            proof: A kind-27235 Nostr event signed by the OPERATOR's nsec
                for this tool. Patron proofs are rejected.
        """
        if not proof:
            return {
                "success": False,
                "error_code": "operator_proof_required",
                "error": "Only the operator can restore credits — provide a proof signed by the operator's nsec.",
            }
        # Accept both inline kind-27235 and cached poison-keyed proof tokens.
        # The wheel's require_caller_proof checks the cache first (filled by
        # the request/receive_npub_proof handshake) and falls through to
        # inline Schnorr verification.
        err = await rt.require_caller_proof(
            rt.operator_npub(), proof, "restore_credits",
        )
        if err:
            return err
        try:
            patron_npub = resolve_npub(patron_npub)
            cashier = await rt.ensure_cashier()
            cache = await rt.ledger_cache()
        except (ValueError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        ttl = await rt.resolve_tranche_lifetime()
        return await credits.restore_credits_tool(
            cashier, cache, patron_npub, invoice_id,
            tranche_lifetime_seconds=ttl,
        )

    @tool
    async def account_statement(npub: str, proof: str, days: int = 30) -> dict[str, Any]:
        """Generate a patron's account statement at this operator.

        Returns the patron's purchase history, active credit tranches,
        per-tool usage breakdown, and recent daily usage logs. This is
        the patron's spending account — not the operator's Authority
        tax balance.

        Free — no credits consumed. Proof of npub ownership is required
        to prevent statement-scraping of arbitrary patrons.

        Args:
            npub: The patron's Nostr public key (npub1...).
            proof: A kind-27235 Nostr event signed by npub for this tool.
            days: Number of days of daily usage history to include (default 30).
        """
        if err := await rt.require_caller_proof(npub, proof, "account_statement"):
            return err
        try:
            npub = resolve_npub(npub)
            cache = await rt.ledger_cache()
        except ValueError as e:
            return {"success": False, "error": str(e)}
        from tollbooth.tools import credits
        return await credits.account_statement_tool(cache, npub, days=days)

    @tool
    async def account_statement_infographic(npub: str, proof: str, days: int = 30) -> dict[str, Any]:
        """Generate a visual SVG infographic of your account statement.

        Returns the same data as account_statement, rendered as a dark-themed
        SVG graphic with balance hero, metrics cards, health gauge, tranche
        table, and tool usage breakdown. Costs 1 api_sat per call. Proof is
        verified by ``debit_or_deny`` before any cost is incurred.

        Args:
            npub: The Nostr public key (npub1...) whose statement to render.
            proof: A kind-27235 Nostr event signed by npub for this tool.
            days: Number of days of daily usage history to include (default 30).
        """
        try:
            npub = resolve_npub(npub)
        except ValueError as e:
            return {"success": False, "error": str(e)}
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
        # Trigger lazy init so the field reflects whether the vault CAN
        # be opened (config present + reachable), not whether some prior
        # tool happened to touch it. ``rt.vault()`` raises on missing
        # NEON_DATABASE_URL or unreachable backend; we catch and report
        # false so service_status stays a non-failing diagnostic.
        vault_ok = False
        try:
            v = await rt.vault()
            vault_ok = v is not None
        except Exception:
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

        # Operator npub: None signals "couldn't resolve" so the assembler can
        # distinguish that from a resolved-but-empty value (still fingerprinted).
        try:
            op_npub: str | None = rt.operator_npub()
        except Exception:
            op_npub = None

        from tollbooth.tools.status import build_service_status
        return build_service_status(
            service=service_name or slug,
            slug=slug,
            version=service_version,
            tollbooth_version=wheel_version,
            vault_ok=vault_ok,
            courier_ok=courier_ok,
            operator_npub=op_npub,
            process_id=os.getpid(),
            env=os.environ,
        )

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
    async def get_patron_onboarding_status(patron_npub: str, proof: str) -> dict[str, Any]:
        """Report a patron's credential readiness for this operator.

        For set-once services (eXcalibur, TheBrain), shows which patron
        secrets are configured and which are missing. For dynamic/OAuth2
        services (Schwab), reports that no patron credentials are needed.
        Free. Proof of npub ownership is required because credential
        presence is sensitive information about the patron's setup.

        Args:
            patron_npub: The patron's Nostr public key (npub1...).
            proof: A kind-27235 Nostr event signed by patron_npub for this tool.
        """
        if err := await rt.require_caller_proof(patron_npub, proof, "get_patron_onboarding_status"):
            return err
        return await rt.patron_onboarding_status(patron_npub)

    # -- Secure Courier ------------------------------------------------

    @tool
    async def session_status(patron_npub: str = "") -> dict[str, Any]:
        """Check operator readiness. Returns the operator lifecycle
        state and clear guidance on what to do next. Free.

        Lifecycle states:
        - ready: Operator is warm and fully operational — vault AND pricing
          model verified. Proceed with tool calls.
        - warming_up: Operator is initializing (cold start). Try a tool call — it will warm up on demand.
        - misconfigured: Persistence rejected a query with a permanent SQL
          error (permission denied, missing relation). Paid tools will fail
          until the operator repairs the database — retrying does not help.
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

        # 4. Pricing-layer probe — "ready means ready". The paid-tool gate
        # depends on the pricing model loading from Neon; a vault that
        # answers while the pricing table rejects queries must not report
        # ready (that green light cost a real outage its true diagnosis).
        resolver = await rt.pricing_resolver()
        await resolver._ensure_fresh()
        if not resolver.neon_available:
            if resolver.last_error_permanent:
                return {
                    "success": True,
                    "lifecycle": "misconfigured",
                    "operator_npub": npub,
                    "message": "Persistence rejected the pricing-model query "
                               "with a permanent SQL error. Paid tools will "
                               "fail until the operator repairs the database — "
                               "retrying does not help.",
                    "detail": resolver.last_error_summary,
                }
            return {
                "success": True,
                "lifecycle": "warming_up",
                "operator_npub": npub,
                "message": "Operator vault is up but the pricing model has "
                           "not loaded yet. Retry shortly.",
                "detail": resolver.last_error_summary,
            }

        # 5. Fully ready
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
                import time as _t
                from tollbooth.tools.status import build_upstream_oauth_block
                block = build_upstream_oauth_block(
                    creds,
                    service_name=opc.service_name,
                    refresh_enabled=opc.refresh_enabled,
                    now_ts=_t.time(),
                )
                if block is not None:
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
        from tollbooth.tools.courier import request_credential_channel_tool
        return await request_credential_channel_tool(rt, sender_npub, service)

    @tool
    async def receive_credentials(
        sender_npub: str = "",
        service: str = "",
        poison: str = "",
        proof: str = "",
        credential_card: str = "",
    ) -> dict[str, Any]:
        """Pick up credentials from the Secure Courier.

        **Call this only after the user confirms they have replied.**
        Deterministic, one-shot retrieval: name the response you want with
        ``(sender_npub, service, poison)`` and the tool drains ONLY the
        rendezvous relay that channel was pinned to. Every popped DM with the
        wrong session phrase is deleted and its sender is NACK'd; the first DM
        with the matching phrase is accepted (ACK'd) and the scan stops. If
        none match, the queue is drained and a ``courier_not_found`` result is
        returned. Do NOT poll, loop, or retry.

        If a credential_card (ncred1...) is provided, it is redeemed directly
        without any relay access (poison not required for that path). On
        success, the payment processor client is reinitialized from the new
        credentials — no server restart needed.

        Args:
            sender_npub: Required. The npub that sent the credentials.
            service: Required. The credential service name (must match
                the service used in request_credential_channel).
            poison: Required. The session phrase returned by
                request_credential_channel for this exact channel.
            credential_card: Optional. An ncred1... card to redeem directly
                (bypasses the relay drain; poison not needed).
        Free.
        """
        from tollbooth.tools.courier import receive_credentials_tool
        return await receive_credentials_tool(
            rt, sender_npub, service, poison, credential_card,
            service_name=service_name, slug=slug,
        )

    @tool
    async def forget_credentials(
        service: str,
        npub: str,
        proof: str,
    ) -> dict[str, Any]:
        """Delete vaulted credentials for a specific service and npub.

        For operator credentials, pass the operator's own npub. For patron
        credentials, pass the patron's npub. Always requires proof of
        npub ownership — a deletion is as destructive as a write.

        Args:
            service: The credential service to forget.
            npub: The Nostr public key (npub1...) whose credentials to forget.
            proof: A kind-27235 Nostr event signed by npub for this tool.
        Free.
        """
        from tollbooth.tools.courier import forget_credentials_tool
        return await forget_credentials_tool(rt, service, npub, proof)

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
            from tollbooth.tools.courier import request_patron_credentials_tool
            return await request_patron_credentials_tool(rt, sender_npub)

        @tool
        async def receive_patron_credentials(
            sender_npub: str = "",
            poison: str = "",
            proof: str = "",
            credential_card: str = "",
        ) -> dict[str, Any]:
            """Pick up patron credentials from the Secure Courier.

            Deterministic, one-shot retrieval: name the response with
            ``(sender_npub, poison)`` and the tool drains ONLY the pinned
            rendezvous relay for that channel, stopping at the matching DM.
            Provide an ncred1... credential_card to redeem directly instead
            (poison not required for that path). Do NOT poll or retry.
            Free.
            """
            from tollbooth.tools.courier import receive_patron_credentials_tool
            return await receive_patron_credentials_tool(
                rt, sender_npub, poison, credential_card,
            )

    # Prune registry entries for conditionally-skipped tools so they
    # don't appear in the pricing model or mismatch detection.
    if rt._patron_credential_template is None:
        for cap in ("request_patron_credentials", "receive_patron_credentials"):
            rt._tool_registry.pop(capability_uuid(cap), None)

    # -- Patron credential CRUD (field-level) --------------------------------

    @tool
    async def update_patron_credential(
        npub: str, proof: str, field: str, value: str,
    ) -> dict[str, Any]:
        """Add or update a single patron credential field.

        Merges into existing stored credentials without affecting
        other fields. Useful for setting an account identifier
        after OAuth, changing a default brain, etc. Free. Proof of
        npub ownership is required — this is a write to the patron's
        sensitive credential vault.

        Args:
            npub: The patron's Nostr public key (npub1...).
            proof: A kind-27235 Nostr event signed by npub for this tool.
            field: The credential field name to set.
            value: The value to store.
        """
        if err := await rt.require_caller_proof(npub, proof, "update_patron_credential"):
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
        npub: str, proof: str, field: str,
    ) -> dict[str, Any]:
        """Remove a single patron credential field.

        Deletes one field from stored credentials without affecting
        other fields. Free. Proof of npub ownership is required —
        this is a write to the patron's sensitive credential vault.

        Args:
            npub: The patron's Nostr public key (npub1...).
            proof: A kind-27235 Nostr event signed by npub for this tool.
            field: The credential field name to remove.
        """
        if err := await rt.require_caller_proof(npub, proof, "delete_patron_credential"):
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
        npub: str, proof: str,
    ) -> dict[str, Any]:
        """List stored patron credential field names (not values).

        Returns the names of fields stored for a patron. Values
        are never exposed — use this to verify which fields are
        configured. Free. Proof of npub ownership is required: the
        list of configured fields is itself sensitive (reveals
        which integrations a patron has set up).

        Args:
            npub: The patron's Nostr public key (npub1...).
            proof: A kind-27235 Nostr event signed by npub for this tool.
        """
        if err := await rt.require_caller_proof(npub, proof, "get_patron_credential_fields"):
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
        # Tool bodies live in tollbooth.tools.oauth (they read rt._oauth_provider).

        @tool
        async def begin_oauth(npub: str, proof: str) -> dict[str, Any]:
            """Start the OAuth2 authorization flow.

            Returns an authorization URL. Open it in a browser to log in
            and authorize. Then call ``check_oauth_status`` with the
            same npub to complete. Free. Proof of npub ownership is
            required so an observer cannot DOS your account by
            initiating OAuth flows in your name.

            Args:
                npub: Your DPYC patron npub (npub1...).
                proof: A kind-27235 Nostr event signed by npub for this tool.
            """
            from tollbooth.tools.oauth import begin_oauth_tool
            return await begin_oauth_tool(rt, npub, proof)

        @tool
        async def check_oauth_status(npub: str, proof: str) -> dict[str, Any]:
            """Check whether the OAuth2 authorization flow has completed.

            Call after opening the authorization URL from ``begin_oauth``
            and completing the login in your browser. Free. Proof of npub
            ownership is required: OAuth status exposes which upstream
            services a patron has connected.

            Args:
                npub: The same Nostr public key (npub1...) used in begin_oauth.
                proof: A kind-27235 Nostr event signed by npub for this tool.
            """
            from tollbooth.tools.oauth import check_oauth_status_tool
            return await check_oauth_status_tool(rt, npub, proof)

    # Prune registry if OAuth not configured
    if rt._oauth_provider is None:
        for cap in ("begin_oauth", "check_oauth_status"):
            rt._tool_registry.pop(capability_uuid(cap), None)

    # -- Npub ownership proof tools ----------------------------------------
    #
    # Uses Secure Courier infrastructure (open_channel / receive) so relay
    # alignment, vault-first lookup, and DM decryption all come for free.
    # Tool bodies live in tollbooth.tools.proof (PROOF_SERVICE constant there).

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
        from tollbooth.tools.proof import request_npub_proof_tool
        return await request_npub_proof_tool(
            rt, patron_npub, service_name=service_name,
        )

    @tool
    async def receive_npub_proof(
        patron_npub: str = "",
        poison: str = "",
    ) -> dict[str, Any]:
        """Receive npub ownership confirmation from a patron.

        **Call this only after the user confirms they have replied.**
        Deterministic, one-shot retrieval: name the response with
        ``(patron_npub, poison)`` — the poison being the ``proof_token``
        returned by ``request_npub_proof``. The tool drains ONLY the pinned
        rendezvous relay that challenge was published on, stopping at the DM
        whose phrase matches. Mismatched DMs are deleted and NACK'd (without
        revealing the expected phrase). If called before the user replies,
        their message will never be found. Do NOT poll, loop, or retry.

        The signed DM itself proves npub ownership (the patron's nsec
        signed it). On success, returns the ``proof_token`` — the same
        poison phrase. The calling application MUST remember this token and
        pass it as the ``proof`` parameter on every subsequent paid tool
        call. The proof is stored in the vault keyed by a hash of the
        poison — the MCP never stores the raw poison itself. Free.

        Args:
            patron_npub: Required. The patron's npub to receive proof from.
            poison: Required. The proof_token returned by request_npub_proof.
        """
        from tollbooth.tools.proof import receive_npub_proof_tool
        return await receive_npub_proof_tool(rt, patron_npub, poison)

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
        from tollbooth.tools.proof import check_proof_status_tool
        return await check_proof_status_tool(rt, patron_npub, proof_token)

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
        # Accept BOTH proof tactics — inline kind-27235 and cached
        # poison-keyed token from request/receive_npub_proof.
        # require_caller_proof checks the cache first, falls through to
        # inline verification.
        err = await rt.require_caller_proof(
            rt.operator_npub(), proof, "set_pricing_model",
        )
        if err:
            return err

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
        # Accept both inline kind-27235 and cached poison-keyed proof tokens.
        err = await rt.require_caller_proof(
            rt.operator_npub(), proof, "reset_pricing_model",
        )
        if err:
            return err
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

    @tool
    async def restore_neon_schema(proof: str = "") -> dict[str, Any]:
        """Re-run ``ensure_schema()`` on every NeonVault this operator uses.

        Diagnostic / recovery tool for the case where the Neon HTTP SQL API
        is returning persistent 4xx errors and the operator suspects the
        schema isn't there or grants are wrong. Idempotent — uses
        ``CREATE TABLE IF NOT EXISTS`` so a successful re-run is harmless.

        Returns the per-step result. If any step raises, surfaces the Neon
        error message inline (0.31.0 reads the SQL error body that earlier
        wheels swallowed behind ``raise_for_status``).

        RESTRICTED to operator — requires proof (nsec-signed).
        """
        if not proof:
            return {
                "success": False,
                "error": "Only the operator can re-run schema setup — provide proof.",
            }
        # Accept both inline kind-27235 and cached poison-keyed proof tokens.
        err = await rt.require_caller_proof(
            rt.operator_npub(), proof, "restore_neon_schema",
        )
        if err:
            return err

        steps: list[dict[str, Any]] = []

        async def _try(label: str, coro: Any) -> None:
            try:
                await coro
                steps.append({"step": label, "ok": True})
            except Exception as exc:
                steps.append({
                    "step": label,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                })

        try:
            vault = await rt.vault()
        except Exception as exc:
            return {
                "success": False,
                "error_type": type(exc).__name__,
                "error": f"Could not open vault: {exc}",
            }

        await _try("NeonVault.ensure_schema", vault.ensure_schema())

        try:
            from tollbooth.pricing_store import PricingModelStore
            store = PricingModelStore(neon_vault=vault)
            await _try("PricingModelStore.ensure_schema", store.ensure_schema())
        except Exception as exc:
            steps.append({"step": "PricingModelStore.ensure_schema", "ok": False, "error": str(exc)[:500]})

        # Credential vault and proven-npub cache live on the same Neon —
        # only re-create their schemas if those subsystems are already
        # configured (operators without credentials don't need this).
        if getattr(rt, "_credential_vault", None) is not None:
            await _try("CredentialVault.ensure_schema", rt._credential_vault.ensure_schema())

        all_ok = all(s.get("ok") for s in steps)
        return {
            "success": all_ok,
            "steps": steps,
            "message": (
                "Schema setup re-run completed cleanly."
                if all_ok else
                "Some schema steps failed — see steps[] for per-step error messages."
            ),
        }

    # -- Canonical-identity introspection ------------------------------

    @tool
    async def list_canonical_identities() -> dict[str, Any]:
        """Return canonical (tool_id, mcp_name, …) for every registered tool.

        The authoritative source for any client (Studio, agents, FE)
        that needs to know how this MCP identifies its tools.
        Reconcile uses this output to UUID-join against the stored
        pricing model — no name-based UUID derivation, no guessing.

        If the operator renames a function or rebrands a slug, the
        mcp_name in this output changes but tool_id stays. That's the
        whole point of the canonical-UUID design.

        Free, no side effects.
        """
        from tollbooth.tools.identities import build_canonical_identities
        return build_canonical_identities(
            rt._tool_registry, rt.mcp_name_for, rt.operator_npub(),
        )

    # -- Constraint Engine tools ---------------------------------------

    @tool
    async def check_price(tool_id: str, npub: str = "", proof: str = "", tool_kwargs: str = "") -> dict[str, Any]:
        """Preview the effective cost of a tool call.

        Shows the base cost and any constraint effects (discounts, free
        trials, surge pricing). Free — no credits required.

        Args:
            tool_id: Either the tool's UUID (from the pricing model) or a
                bare capability string (e.g. ``"deal_scenario"``). FE
                callers usually have the capability name; this resolves
                both so the FE doesn't need to derive UUIDs locally.
            tool_kwargs: Optional JSON object with tool call parameters
                for ad valorem / categorical-multiplier pricing preview
                (e.g. '{"amount_sats": 5000}' or
                '{"difficulty": "sovereign", "mode": "live"}').
        """
        identity = rt._tool_registry.get(tool_id)
        # Fallback: caller passed a capability name instead of a UUID.
        if identity is None:
            from tollbooth.tool_identity import capability_uuid as _cap_uuid
            try:
                resolved_id = _cap_uuid(tool_id)
                identity = rt._tool_registry.get(resolved_id)
                if identity is not None:
                    tool_id = resolved_id
            except Exception:
                pass
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
        from tollbooth.tools.pricing import (
            apply_constraint_preview,
            build_pricing_preview,
        )
        result = build_pricing_preview(tool_id, name, pricing, parsed_kwargs)

        base_cost = result.get("base_cost_api_sats") or 0
        # Preview the chain for this tool, if it has one.  The cached
        # model is already warmed by the resolver.get_tool_pricing call
        # above, so chain_for is a synchronous in-memory lookup.
        chain = (
            resolver._cached_model.chain_for(tool_id)
            if resolver._cached_model is not None
            else []
        )
        if chain and base_cost != 0:
            result["constraints_enabled"] = True
            try:
                resolved = resolve_npub(npub)
                cache = await rt.ledger_cache()
                ledger = await cache.get(resolved)
                demand = await rt.get_global_demand(name)
                gate = rt._constraint_gate
                if gate is None:
                    from tollbooth.constraints.gate import ConstraintGate
                    gate = ConstraintGate()
                    gate.attach_resolver(resolver)
                    rt._constraint_gate = gate
                # Preview: pre-load redemptions so a coupon-discounted
                # price shows accurately, but don't burn any uses.
                coupon_map = None
                coupon_ids = gate.collect_coupon_ids(chain)
                if coupon_ids and resolved:
                    try:
                        cv = await rt.coupons_vault()
                        redemptions = await cv.fetch_redemptions_for_chain(
                            resolved, coupon_ids,
                        )
                        from tollbooth.coupons import CouponRedemptionMap
                        coupon_map = CouponRedemptionMap(
                            entries=tuple(redemptions.items()),
                        )
                    except Exception as ce:
                        logger.warning(
                            "check_price coupon pre-load failed: %s", ce,
                        )
                denial, effective, _consumed = gate.evaluate_chain(
                    chain=chain,
                    tool_name=name,
                    base_cost=int(base_cost),
                    ledger=ledger,
                    npub=resolved,
                    global_demand=demand,
                    coupon_redemptions=coupon_map,
                )
                apply_constraint_preview(
                    result, int(base_cost), effective, denial, demand.get(name, 0),
                )
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

    # -- Coupon CRUD + redemption -------------------------------------------
    # Tool bodies live in tollbooth.tools.coupons; these shims handle the proof
    # gate + dependency resolution (coupons vault, operator/patron npub), then
    # delegate. Function names are the wire API (mcp_name_for derives
    # {slug}_{name}) — keep them stable.

    @tool
    async def mint_coupon(
        name: str,
        discount_percent: float,
        valid_from: str,
        valid_until: str,
        uses_per_patron: int | None = 1,
        total_uses: int | None = None,
        proof: str = "",
    ) -> dict[str, Any]:
        """Create a new operator-owned discount coupon.

        Args:
            name: The catchy code patrons type to redeem
                (operator-scoped uniqueness).
            discount_percent: Percentage off the base price (0-100).
            valid_from: ISO-8601 datetime when the coupon becomes active.
            valid_until: ISO-8601 datetime when the coupon expires.
            uses_per_patron: How many tool calls one patron can claim the
                discount on (default 1; pass null/None for unlimited
                within the window).
            total_uses: Aggregate cap across all patrons (default None =
                unlimited).

        Returns the new coupon row. RESTRICTED to operator — requires
        proof (nsec-signed kind-27235 or cached poison token).
        """
        if not proof:
            return {
                "success": False,
                "error": "Only the operator can mint coupons — provide proof.",
            }
        err = await rt.require_caller_proof(
            rt.operator_npub(), proof, "mint_coupon",
        )
        if err:
            return err

        from tollbooth.tools import coupons as _coupons
        cv = await rt.coupons_vault()
        return await _coupons.mint_coupon_tool(
            cv, rt.operator_npub(),
            name=name,
            discount_percent=discount_percent,
            valid_from=valid_from,
            valid_until=valid_until,
            uses_per_patron=uses_per_patron,
            total_uses=total_uses,
        )

    @tool
    async def list_coupons(proof: str = "") -> dict[str, Any]:
        """List every coupon this operator has minted (newest first).

        Each row carries the current ``times_redeemed`` counter — the
        Studio renders a progress bar from this against ``total_uses``.
        RESTRICTED to operator — requires proof.
        """
        if not proof:
            return {
                "success": False,
                "error": "Only the operator can list coupons — provide proof.",
            }
        err = await rt.require_caller_proof(
            rt.operator_npub(), proof, "list_coupons",
        )
        if err:
            return err

        from tollbooth.tools import coupons as _coupons
        cv = await rt.coupons_vault()
        return await _coupons.list_coupons_tool(cv, rt.operator_npub())

    @tool
    async def update_coupon(
        coupon_id: str,
        name: str | None = None,
        discount_percent: float | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        uses_per_patron: int | None = None,
        total_uses: int | None = None,
        clear_uses_per_patron: bool = False,
        clear_total_uses: bool = False,
        proof: str = "",
    ) -> dict[str, Any]:
        """Patch a coupon's editable fields.

        Pass only the fields you want to change.  To set a cap to
        unlimited (NULL in the schema), pass ``clear_uses_per_patron=true``
        or ``clear_total_uses=true``.  Renaming the code is allowed —
        existing patron redemption rows survive (they key on coupon id).

        RESTRICTED to operator — requires proof.
        """
        if not proof:
            return {
                "success": False,
                "error": "Only the operator can update coupons — provide proof.",
            }
        err = await rt.require_caller_proof(
            rt.operator_npub(), proof, "update_coupon",
        )
        if err:
            return err

        from tollbooth.tools import coupons as _coupons
        cv = await rt.coupons_vault()
        return await _coupons.update_coupon_tool(
            cv, rt.operator_npub(), coupon_id,
            name=name,
            discount_percent=discount_percent,
            valid_from=valid_from,
            valid_until=valid_until,
            uses_per_patron=uses_per_patron,
            total_uses=total_uses,
            clear_uses_per_patron=clear_uses_per_patron,
            clear_total_uses=clear_total_uses,
        )

    @tool
    async def delete_coupon(coupon_id: str, proof: str = "") -> dict[str, Any]:
        """Delete a coupon.  Cascades to all patron redemptions.

        Any chain step referencing the deleted coupon_id becomes a
        no-op (the constraint returns neutral on unknown ids) — the
        Studio surfaces orphan references as warnings.

        RESTRICTED to operator — requires proof.
        """
        if not proof:
            return {
                "success": False,
                "error": "Only the operator can delete coupons — provide proof.",
            }
        err = await rt.require_caller_proof(
            rt.operator_npub(), proof, "delete_coupon",
        )
        if err:
            return err

        from tollbooth.tools import coupons as _coupons
        cv = await rt.coupons_vault()
        return await _coupons.delete_coupon_tool(cv, rt.operator_npub(), coupon_id)

    @tool
    async def redeem_coupon(npub: str, code: str, proof: str = "") -> dict[str, Any]:
        """Claim a coupon by its name (the code the operator shared).

        Looks up the operator's coupon by ``code``, validates the window
        and total cap, and records a per-patron redemption row.
        Subsequent paid tool calls on this MCP auto-apply the discount
        until ``uses_per_patron`` is exhausted.

        Free — no credits required.  Requires proof of ``npub``.
        Idempotent: redeeming the same code twice returns the existing
        redemption.
        """
        try:
            resolved = resolve_npub(npub)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        err = await rt.require_caller_proof(resolved, proof, "redeem_coupon")
        if err:
            return err

        from tollbooth.tools import coupons as _coupons
        cv = await rt.coupons_vault()
        return await _coupons.redeem_coupon_tool(
            cv, rt.operator_npub(), resolved, code,
        )

    @tool
    async def list_my_coupons(npub: str, proof: str = "") -> dict[str, Any]:
        """List the coupons this patron has redeemed on this operator.

        Returns both active and exhausted redemptions with a per-row
        ``status`` (``active`` / ``window_closed`` / ``patron_limit`` /
        ``total_limit``).  Free — requires proof of ``npub``.
        """
        try:
            resolved = resolve_npub(npub)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        err = await rt.require_caller_proof(resolved, proof, "list_my_coupons")
        if err:
            return err

        from tollbooth.tools import coupons as _coupons
        cv = await rt.coupons_vault()
        return await _coupons.list_my_coupons_tool(cv, rt.operator_npub(), resolved)

    @tool
    async def forget_coupon(
        npub: str, coupon_id: str, proof: str = "",
    ) -> dict[str, Any]:
        """Remove a coupon from this patron's redemption list.

        Cosmetic only — the coupon itself still exists at the operator,
        and the patron can re-redeem the same code later while the
        window allows.  Free — requires proof of ``npub``.
        """
        try:
            resolved = resolve_npub(npub)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        err = await rt.require_caller_proof(resolved, proof, "forget_coupon")
        if err:
            return err

        from tollbooth.tools import coupons as _coupons
        cv = await rt.coupons_vault()
        return await _coupons.forget_coupon_tool(cv, resolved, coupon_id)

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

    return tool


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
