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
    err = await runtime.debit_or_deny(my_tool_uuid, npub)
    if err:
        return err
    # ... do work ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import signal
from typing import Any, Callable

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
    ) -> None:
        self._nsec_env_var = nsec_env_var
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
        operator_proof: str = "",
        patron_proof: str = "",
        tool_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Gate a tool call: identity → access → pricing → constraints → billing.

        Returns ``None`` to proceed.  Returns an error dict to deny.

        The code-declared *category* is the immutable floor for billing.
        The constraint pipeline can add gates but never remove them.
        """
        from tollbooth.operator_proof import verify_operator_proof

        identity = self._tool_registry.get(tool_id)

        # ── Identity ──────────────────────────────────────────
        if identity is None:
            return {
                "success": False,
                "error": f"Unknown tool '{tool_id}' — not registered with this operator.",
            }

        name = self.mcp_name_for(tool_id)        # full MCP name for display/logging
        cap = identity.capability              # short name for proof verification only
        category = identity.category           # set in code, never by the pricing model

        # ── Access: operator-restricted ───────────────────────
        # No billing, no constraints.  Only the operator's npub
        # (or a valid operator_proof signer) may call these.
        if category == "restricted":
            try:
                caller = resolve_npub(npub)
            except ValueError as e:
                return {"success": False, "error": str(e)}
            if caller == self.operator_npub():
                return None
            if operator_proof and verify_operator_proof(
                operator_proof, self.operator_npub(), cap,
            ):
                return None
            return {"success": False, "error": "This tool is restricted to the operator."}

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
                    "error": (
                        "Service unavailable — persistence layer is unreachable. "
                        "Only bootstrap tools are available."
                    ),
                }

            if not await resolver.has_tool(tool_id):
                return {
                    "success": False,
                    "error": (
                        f"Tool '{name}' is not yet in the pricing model. "
                        f"Add it to the pricing model before use."
                    ),
                }
            if not await resolver.is_priced(tool_id):
                return {
                    "success": False,
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
                    "error": (
                        "npub is required. Pass your Nostr public key (npub1...) "
                        "to identify yourself."
                    ),
                }
            npub = ""

        # ── Constraints ───────────────────────────────────────
        # The pipeline applies to every non-restricted tool —
        # including free ones.  patron_proof verification, rate
        # limits, temporal windows, etc. can tighten access but
        # never loosen the code-declared category floor.
        # Without an npub, constraints cannot be evaluated.
        effective_cost = cost
        if npub and self._constraint_gate and self._constraint_gate.enabled:
            try:
                cache = await self.ledger_cache()
                ledger = await cache.get(npub)
                denial, effective_cost = self._constraint_gate.check(
                    tool_name=name,
                    base_cost=cost,
                    ledger=ledger,
                    npub=npub,
                    patron_proof=patron_proof,
                )
                if denial:
                    return denial
            except Exception:
                if category != "free":
                    return {
                        "success": False,
                        "error": "Service unavailable — cannot evaluate constraints.",
                    }
                logger.debug("Constraint evaluation skipped for free tool %s", name)

        # ── No charge ─────────────────────────────────────────
        if effective_cost == 0:
            return None

        # ── Billing ───────────────────────────────────────────
        cache = await self.ledger_cache()
        ledger = await cache.get(npub)

        if npub not in self._reconciled_npubs and ledger.pending_invoices:
            self._reconciled_npubs.add(npub)
            try:
                cashier = await self.ensure_cashier()
                ttl = await self.resolve_credit_ttl()
                from tollbooth.tools.credits import reconcile_pending_invoices
                recon = await reconcile_pending_invoices(
                    cashier, cache, npub,
                    default_credit_ttl_seconds=ttl,
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
                "error": (
                    f"Insufficient balance: {ledger.balance_api_sats} sats "
                    f"available, {effective_cost} required for {name}."
                ),
                "balance_sats": ledger.balance_api_sats,
                "cost_sats": effective_cost,
            }

        ledger.debit(name, effective_cost)
        cache.mark_dirty(npub)
        return None

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

    async def resolve_credit_ttl(self) -> int | None:
        """Return the effective credit TTL in seconds from the demurrage constraint.

        Scans the active pricing model's pipeline for a demurrage
        step and returns ttl_days * 86400. Returns None if no such constraint
        exists (credits never expire).
        """
        try:
            store = await self.ensure_pricing_store()
            from tollbooth.tools.pricing import get_pricing_model_tool
            result = await get_pricing_model_tool(store, self.operator_npub())
            if result.get("status") == "ok":
                for step in result.get("pipeline", []):
                    step_data = step if isinstance(step, dict) else step.to_dict()
                    if step_data.get("type") == "demurrage":
                        params = step_data.get("params", step_data)
                        days = int(params.get("ttl_days", step_data.get("ttl_days", 15)))
                        return days * 86400
        except Exception:
            pass
        return None

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

        The decorated function **must** accept ``npub`` as a keyword argument.

        Example::

            @tool
            @runtime.paid_tool(TOOL_REGISTRY["get_weather"].tool_id)
            async def get_weather(lat: float, lon: float, npub: str = ""):
                return await weather.get(lat, lon)
        """
        rt = self

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                sig = inspect.signature(fn)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                npub = bound.arguments.get("npub", "")
                operator_proof = bound.arguments.get("operator_proof", "")
                patron_proof = bound.arguments.get("patron_proof", "")

                call_kwargs = dict(bound.arguments)
                err = await rt.debit_or_deny(
                    tool_id, npub,
                    operator_proof=operator_proof,
                    patron_proof=patron_proof,
                    tool_kwargs=call_kwargs,
                )
                if err is not None:
                    return err

                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    await rt.rollback_debit(tool_id, npub, tool_kwargs=call_kwargs)
                    if catch_errors:
                        return {"success": False, "error": str(exc)}
                    raise

                rt.fire_and_forget_demand_increment(rt.mcp_name_for(tool_id))
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
    async def check_balance(npub: str = "") -> dict[str, Any]:
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
    async def purchase_credits(amount_sats: int = 1000, npub: str = "") -> dict[str, Any]:
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
                auth_info["url"], rt.operator_npub(),
            ).certify_credits(amount_sats)
            certificate = cert_result.get("certificate", "")
        except Exception as e:
            return {"success": False, "error": f"Authority certification failed: {e}"}

        try:
            cashier = await rt.ensure_cashier()
            cache = await rt.ledger_cache()
            from tollbooth.tools import credits
            ttl = await rt.resolve_credit_ttl()
            return await credits.purchase_credits_tool(
                cashier, cache, npub, amount_sats, certificate,
                authority_npub=auth_info.get("npub", ""),
                default_credit_ttl_seconds=ttl,
            )
        except ValueError as e:
            return {"success": False, "error": str(e)}

    @tool
    async def check_payment(invoice_id: str, npub: str = "") -> dict[str, Any]:
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
        ttl = await rt.resolve_credit_ttl()
        result = await credits.check_payment_tool(
            cashier, cache, npub, invoice_id,
            default_credit_ttl_seconds=ttl,
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
    async def restore_credits(invoice_id: str, npub: str = "") -> dict[str, Any]:
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
        ttl = await rt.resolve_credit_ttl()
        return await credits.restore_credits_tool(
            cashier, cache, npub, invoice_id,
            default_credit_ttl_seconds=ttl,
        )

    @tool
    async def account_statement(npub: str = "", days: int = 30) -> dict[str, Any]:
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
    async def account_statement_infographic(npub: str = "", days: int = 30) -> dict[str, Any]:
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
        err = await rt.debit_or_deny(capability_uuid("account_statement_infographic"), npub)
        if err:
            return err
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
    async def get_patron_onboarding_status(patron_npub: str = "") -> dict[str, Any]:
        """Report a patron's credential readiness for this operator.

        For set-once services (eXcalibur, TheBrain), shows which patron
        secrets are configured and which are missing. For dynamic/OAuth2
        services (Schwab), reports that no patron credentials are needed.
        Free.

        Args:
            patron_npub: Required. The patron's Nostr public key.
        """
        if not patron_npub or not patron_npub.startswith("npub1"):
            return {"success": False, "error": "patron_npub is required."}
        return await rt.patron_onboarding_status(patron_npub)

    # -- Secure Courier ------------------------------------------------

    @tool
    async def session_status() -> dict[str, Any]:
        """Check operator readiness. Returns the operator lifecycle
        state and clear guidance on what to do next. Free.

        Lifecycle states:
        - ready: Operator is warm and fully operational. Proceed with tool calls.
        - warming_up: Operator is initializing (cold start). Try a tool call — it will warm up on demand.
        - not_registered: Operator has no Authority relationship yet. Call register_operator first.
        - no_identity: Operator nsec is not configured. Deployment issue.
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
        return {
            "success": True,
            "lifecycle": "ready",
            "operator_npub": npub,
            "operator_credential_service": rt.operator_credential_service,
            "patron_credential_service": rt.patron_credential_service,
            "message": "Operator is ready. Proceed with tool calls.",
        }

    @tool
    async def request_credential_channel(
        sender_npub: str = "",
        service: str = "",
    ) -> dict[str, Any]:
        """Open a Secure Courier channel for credential delivery.

        Sends a welcome DM with a credential template. All fields must
        be re-provided; there is no partial update. After the recipient
        replies, call receive_credentials with the same service.

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
        service: str = "",
        credential_card: str = "",
        force_relay: bool = False,
    ) -> dict[str, Any]:
        """Pick up credentials from the Secure Courier.

        Checks the vault first (instant), then polls Nostr relays for
        encrypted DMs. If a credential_card (ncred1...) is provided,
        redeems it directly without relay polling. On success, the
        payment processor client is reinitialized from the new
        credentials — no server restart needed.

        Args:
            sender_npub: Required. The npub that sent the credentials.
            service: Required. The credential service name (must match
                the service used in request_credential_channel).
            force_relay: Skip the vault cache and poll Nostr relays
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
                    credential_card, service,
                )
            else:
                result = await courier.receive(sender_npub, service=service, force_relay=force_relay)
            # Invalidate cached BTCPay client when operator creds change
            if result.get("success") and service == rt.operator_credential_service:
                rt._cashier = None
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    async def forget_credentials(
        service: str = "",
        npub: str = "",
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

    # Prune registry entries for conditionally-skipped tools so they
    # don't appear in the pricing model or mismatch detection.
    if rt._patron_credential_template is None:
        from tollbooth.tool_identity import capability_uuid
        for cap in ("request_patron_credentials", "receive_patron_credentials"):
            rt._tool_registry.pop(capability_uuid(cap), None)

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
                auth_info["url"], rt.operator_npub(),
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
    async def set_pricing_model(model_json: str) -> dict[str, Any]:
        """Set the active pricing model. RESTRICTED to operator.

        The model_json must contain an ``operator_proof`` field — a
        Nostr-signed proof that the caller holds the operator's nsec.
        Without a valid proof, the request is rejected.
        """
        import json as _json

        # Extract and verify operator_proof from inside model_json
        operator_proof = ""
        try:
            parsed = _json.loads(model_json)
            if isinstance(parsed, dict):
                operator_proof = parsed.pop("operator_proof", "")
                model_json = _json.dumps(parsed)
        except (ValueError, TypeError):
            pass

        if not operator_proof:
            return {
                "success": False,
                "error": "Only the operator can modify pricing — provide operator_proof.",
            }
        from tollbooth.operator_proof import verify_operator_proof
        if not verify_operator_proof(operator_proof, rt.operator_npub(), "set_pricing_model"):
            return {
                "success": False,
                "error": "Invalid operator_proof — only the operator can modify pricing.",
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
    async def reset_pricing_model(operator_proof: str = "") -> dict[str, Any]:
        """Erase all pricing models and restore a viable default.

        Deletes every stored model, then self-initializes a fresh one
        from the tool registry — all tools at 0 sats with proper UUIDs.
        Returns the new model.

        RESTRICTED to operator — requires operator_proof (nsec-signed).
        """
        if not operator_proof:
            return {
                "success": False,
                "error": "Only the operator can reset pricing — provide operator_proof.",
            }
        from tollbooth.operator_proof import verify_operator_proof
        if not verify_operator_proof(operator_proof, rt.operator_npub(), "reset_pricing_model"):
            return {
                "success": False,
                "error": "Invalid operator_proof — only the operator can reset pricing.",
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
    async def check_price(tool_id: str, npub: str = "", tool_kwargs: str = "") -> dict[str, Any]:
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
