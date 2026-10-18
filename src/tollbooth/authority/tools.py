"""Authority-class ``@tool`` definitions and the ``register_authority_tools`` mixin.

An Authority MCP service mounts the standard Authority tool surface on
its FastMCP instance by calling :func:`register_authority_tools`. Each
Authority deployment contributes only actor-specific configuration —
the FastMCP instance name, instructions, identity (nsec via env), and
optionally a custom credential template; everything else lives here.

Typical Authority ``server.py``::

    from fastmcp import FastMCP
    from tollbooth.authority import register_authority_tools
    from tollbooth.runtime import OperatorRuntime, register_standard_tools

    mcp = FastMCP("tollbooth-authority-mine", instructions="…")

    runtime = OperatorRuntime(
        tool_registry={**STANDARD_IDENTITIES, **AUTHORITY_TOOL_REGISTRY},
        vault_source="env",       # Authority self-provisions its Neon
        purchase_mode="auto",     # certify-up derived from the registry chain
        service_name="My Authority",
        ots_enabled=True,
        operator_credential_template=OPERATOR_CREDENTIAL_TEMPLATE,
    )

    register_standard_tools(mcp, "authority", runtime, ...)
    register_authority_tools(mcp, runtime)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Annotated, Any

from pydantic import Field

from tollbooth.authority import neon_alert_store, operator_requests
from tollbooth.authority.nostr_signing import AuthorityNostrSigner
from tollbooth.authority.onboarding import ONBOARDING_TEMPLATES, OnboardingState
from tollbooth.authority.replay import ReplayTracker
from tollbooth.authority.settings import AuthoritySettings
from tollbooth.constants import ErrorCode
from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.nostr_diagnostics import resolve_relays as _resolve_relays
from tollbooth.registry import (
    DEFAULT_REGISTRY_URL,
    DPYCRegistry,
    RegistryError,
    resolve_my_parent_npub,
)
from tollbooth.runtime import OperatorRuntime, resolve_npub
from tollbooth.slug_tools import make_slug_tool
from tollbooth.tool_identity import (
    CONSUMES_REMOTE,
    CREATES,
    CREATES_REMOTE,
    DELETES_REMOTE,
    READS,
    READS_REMOTE,
    WRITES,
    WRITES_REMOTE,
    ToolIdentity,
    capability_uuid,
)
from tollbooth.vault_backend import LedgerUnavailableError, LedgerWriteError

logger = logging.getLogger(__name__)


# ======================================================================
# Authority operator credential template (BTCPay)
# ======================================================================

OPERATOR_CREDENTIAL_TEMPLATE = CredentialTemplate(
    service="tollbooth-authority-operator",
    version=1,
    description="BTCPay Lightning payment credentials for the Authority cashier",
    fields={
        "btcpay_host": FieldSpec(
            required=True, sensitive=True,
            description="The URL of your BTCPay Server instance (e.g. https://btcpay.example.com).",
        ),
        "btcpay_api_key": FieldSpec(
            required=True, sensitive=True,
            description="Your BTCPay Server API key with btcpay.store.cancreateinvoice permission.",
        ),
        "btcpay_store_id": FieldSpec(
            required=True, sensitive=True,
            description="Your BTCPay Store ID (visible in Store Settings).",
        ),
        # Optional, courier-delivered (never env): the org-scoped Neon control-plane key +
        # its org id, enabling the PROACTIVE per-project compute-quota watch surfaced in
        # Persistence Status. Absent them, the watch degrades to reactive self-detection.
        "neon_api_key": FieldSpec(
            required=False, sensitive=True,
            description="Neon control-plane API key (org-scoped) for proactive per-project "
                        "compute-quota monitoring. Optional — absent it, only reactive "
                        "402 detection is available.",
        ),
        "neon_org_id": FieldSpec(
            required=False, sensitive=False,
            description="Neon organization id that scopes the project listing for the "
                        "proactive compute-quota watch. Pairs with neon_api_key.",
        ),
    },
)


# ======================================================================
# Authority domain tool registry
# ======================================================================

# Frozen UUIDs for Authority's domain tools. Derived once from
# capability_uuid("<name>") at refactor time; pinned as constants
# thereafter so rebrands and renames don't break pricing-model rows.
CERTIFY_CREDITS_UUID          = "2b132223-a5c4-51e3-a854-a8445c0d7e0f"
UPDATE_OPERATOR_UUID          = "a7856d24-0665-5a3d-ae81-87a54fa35f26"
DEREGISTER_OPERATOR_UUID      = "abbaa449-2b72-5632-812a-fb95c219e7d3"
GET_OPERATOR_CONFIG_UUID      = "f77b4457-28cc-5e14-8bca-64a09d3ac5a0"
OPERATOR_STATUS_UUID          = "1c7bdb45-cdeb-5b3f-9a81-2b8257e171b4"
CHECK_DPYC_MEMBERSHIP_UUID    = "644bfdbc-aefd-58df-914f-e155aef2a94e"
REGISTER_AUTHORITY_NPUB_UUID  = "0ce23a58-eab1-5a66-b9e4-f62219f5f6d2"
CONFIRM_AUTHORITY_CLAIM_UUID  = "0c32c30f-4a1f-51d9-9b46-785de2d3e15a"
CHECK_AUTHORITY_APPROVAL_UUID = "c901360d-a32e-53ea-b965-df35a66af68a"
# Deferred operator adoption (the courtship)
RECEIVE_ADOPTION_REQUEST_UUID = "b77747af-dda0-51be-9952-33e815b4dc5f"
LIST_OPERATOR_REQUESTS_UUID   = "d68a9b0b-dae6-5095-8d9a-1ed6f0b96edc"
APPROVE_OPERATOR_REQUEST_UUID = "104c741c-4647-5e8c-bd6d-3ba49ea44be7"
REJECT_OPERATOR_REQUEST_UUID  = "7dbcfa85-f776-5306-a6dc-9032f1dcb9b3"
GET_ADOPTION_STATUS_UUID      = "45c1b7b4-93ac-52e3-ad5a-8fac1b05fad1"
# Tenant-schema ownership repair (owner consent)
REPAIR_OPERATOR_SCHEMA_UUID   = "b626ee48-dcd7-5ef2-93dd-0be12364acbb"
# Neon persistence health — 402 alerts (inbound from operators) + proactive watch
RECEIVE_NEON_402_ALERT_UUID   = "bdb94fb1-454a-5804-8885-42ba93f1446d"
LIST_NEON_ALERTS_UUID         = "b8079cbd-bc1e-579c-86fa-897a0a0641bd"
NETWORK_PERSISTENCE_HEALTH_UUID     = "07706702-fac9-50d0-b69d-d706a71b9101"
# Bootstrap config propagation — the weekly foreground audit
AUDIT_BOOTSTRAP_CONFIGS_UUID  = "4fde55a7-fbe3-533c-a338-1e7a58482712"


AUTHORITY_DOMAIN_TOOLS: list[ToolIdentity] = [
    ToolIdentity(
        tool_id=CERTIFY_CREDITS_UUID,
        capability="certify_credits",
        category="write",
        title="Certify credits",
        behavior=CREATES,
        intent="Certify a purchase order with Schnorr-signed certificate.",
        pricing_hint_type="percent",
        pricing_hint_value=2,
        pricing_hint_param="amount_sats",
        pricing_hint_min=10,
    ),
    ToolIdentity(
        tool_id=UPDATE_OPERATOR_UUID,
        capability="update_operator",
        category="free",
        title="Request operator update",
        behavior=CREATES,
        intent="An operator asks to change its registry entry; the Authority owner approves.",
    ),
    ToolIdentity(
        tool_id=DEREGISTER_OPERATOR_UUID,
        capability="deregister_operator",
        category="free",
        title="Deregister operator",
        behavior=DELETES_REMOTE,
        intent="Remove an operator from the registry: the operator leaving, or the Authority disowning it.",
    ),
    ToolIdentity(
        tool_id=GET_OPERATOR_CONFIG_UUID,
        capability="get_operator_config",
        category="restricted",
        title="Operator config",
        behavior=READS,
        intent="Retrieve operator bootstrap configuration.",
    ),
    ToolIdentity(
        tool_id=OPERATOR_STATUS_UUID,
        capability="operator_status",
        category="free",
        title="Operator status",
        behavior=READS,
        intent="View registration status, balance, and Authority npub.",
    ),
    ToolIdentity(
        tool_id=CHECK_DPYC_MEMBERSHIP_UUID,
        capability="check_dpyc_membership",
        category="free",
        title="DPYC membership",
        behavior=READS_REMOTE,
        intent="Look up an npub in the DPYC community registry.",
    ),
    ToolIdentity(
        tool_id=REGISTER_AUTHORITY_NPUB_UUID,
        capability="register_authority_npub",
        category="free",
        title="Register Authority npub",
        behavior=CREATES_REMOTE,
        intent="Step 1/3 of Authority onboarding — send DM challenge.",
    ),
    ToolIdentity(
        tool_id=CONFIRM_AUTHORITY_CLAIM_UUID,
        capability="confirm_authority_claim",
        category="free",
        title="Confirm Authority claim",
        behavior=CONSUMES_REMOTE,
        intent="Step 2/3 of Authority onboarding — verify candidate DM.",
    ),
    ToolIdentity(
        tool_id=CHECK_AUTHORITY_APPROVAL_UUID,
        capability="check_authority_approval",
        category="free",
        title="Authority approval",
        behavior=READS_REMOTE,
        intent="Step 3/3 of Authority onboarding — check parent approval.",
    ),
    # -- Deferred operator adoption --
    ToolIdentity(
        tool_id=RECEIVE_ADOPTION_REQUEST_UUID,
        capability="receive_adoption_request",
        category="free",
        title="Receive adoption request",
        behavior=CREATES,
        intent="Inbound: record an operator's request to be adopted (operator-proof gated).",
    ),
    ToolIdentity(
        tool_id=LIST_OPERATOR_REQUESTS_UUID,
        capability="list_operator_requests",
        category="restricted",
        title="Operator requests",
        behavior=READS,
        intent="Owner queue: pending operator requests (adoptions and registry updates).",
    ),
    ToolIdentity(
        tool_id=APPROVE_OPERATOR_REQUEST_UUID,
        capability="approve_operator_request",
        category="restricted",
        title="Approve operator request",
        behavior=CREATES_REMOTE,
        intent="Approve a pending operator request: provision an adoption, or apply an update.",
    ),
    ToolIdentity(
        tool_id=REJECT_OPERATOR_REQUEST_UUID,
        capability="reject_operator_request",
        category="restricted",
        title="Reject operator request",
        behavior=WRITES,
        intent="Reject a pending operator request.",
    ),
    ToolIdentity(
        tool_id=GET_ADOPTION_STATUS_UUID,
        capability="get_adoption_status",
        category="free",
        title="Adoption request status",
        behavior=READS,
        intent="Read the status of an operator's adoption request.",
    ),
    ToolIdentity(
        tool_id=REPAIR_OPERATOR_SCHEMA_UUID,
        capability="repair_operator_schema",
        category="restricted",
        title="Repair operator schema",
        behavior=WRITES,
        intent="Owner repair: reassign all table ownership in an operator's tenant schema to its own role.",
    ),
    # -- Neon persistence health --
    ToolIdentity(
        tool_id=RECEIVE_NEON_402_ALERT_UUID,
        capability="receive_neon_402_alert",
        category="free",
        title="Receive database quota alert",
        behavior=CREATES,
        intent="Inbound: an operator reports its Neon store is 402-locked (operator-proof gated).",
    ),
    ToolIdentity(
        tool_id=LIST_NEON_ALERTS_UUID,
        capability="list_neon_alerts",
        category="restricted",
        title="Database quota alerts",
        behavior=READS,
        intent="Owner queue: list operators that reported a Neon-402 (store locked).",
    ),
    ToolIdentity(
        tool_id=NETWORK_PERSISTENCE_HEALTH_UUID,
        capability="network_persistence_health",
        category="restricted",
        title="Network persistence health",
        behavior=READS_REMOTE,
        intent="Owner view: proactive per-project Neon compute-quota posture + reactive alerts.",
    ),
    ToolIdentity(
        tool_id=AUDIT_BOOTSTRAP_CONFIGS_UUID,
        capability="audit_bootstrap_configs",
        category="free",
        title="Audit bootstrap configs",
        behavior=WRITES_REMOTE,
        intent="Weekly audit: republish any operator bootstrap config held too thinly or too long ago.",
    ),
]

AUTHORITY_TOOL_REGISTRY: dict[str, ToolIdentity] = {
    ti.tool_id: ti for ti in AUTHORITY_DOMAIN_TOOLS
}


# ======================================================================
# Module-level state — set lazily, scoped per-process
# ======================================================================
# Each Authority MCP runs as its own Python process, so the module-level
# state below is effectively per-Authority. `register_authority_tools` is
# the only public surface that mutates it.

_settings: AuthoritySettings | None = None
_nostr_signer: AuthorityNostrSigner | None = None
_replay_tracker: ReplayTracker | None = None
_onboarding = OnboardingState()
_cached_authority_npub: str | None = None
_dpyc_registry: DPYCRegistry | None = None
_runtime: OperatorRuntime | None = None
_exchange: Any = None


def _get_settings() -> AuthoritySettings:
    global _settings
    if _settings is None:
        _settings = AuthoritySettings()
    return _settings


def _get_runtime() -> OperatorRuntime:
    if _runtime is None:
        raise RuntimeError(
            "OperatorRuntime not set. Did you call register_authority_tools()?"
        )
    return _runtime


def _get_nostr_signer() -> AuthorityNostrSigner:
    global _nostr_signer
    if _nostr_signer is not None:
        return _nostr_signer
    s = _get_settings()
    if not s.tollbooth_nostr_operator_nsec:
        raise ValueError(
            "TOLLBOOTH_NOSTR_OPERATOR_NSEC is required. "
            "Generate a Nostr keypair (e.g., `nak key generate`) and set the nsec."
        )
    _nostr_signer = AuthorityNostrSigner(s.tollbooth_nostr_operator_nsec)
    logger.info("Authority Nostr signer initialized (npub=%s).", _nostr_signer.npub)
    return _nostr_signer


def _get_replay_tracker() -> ReplayTracker:
    global _replay_tracker
    if _replay_tracker is not None:
        return _replay_tracker
    s = _get_settings()
    _replay_tracker = ReplayTracker(ttl_seconds=s.certificate_ttl_seconds)
    return _replay_tracker


def _get_dpyc_registry() -> DPYCRegistry | None:
    global _dpyc_registry
    s = _get_settings()
    if not s.dpyc_enforce_membership:
        return None
    if _dpyc_registry is None:
        _dpyc_registry = DPYCRegistry(
            url=DEFAULT_REGISTRY_URL,
            cache_ttl_seconds=s.dpyc_registry_cache_ttl_seconds,
        )
    return _dpyc_registry


async def _nostr_exchange() -> Any:
    """This Authority's Secure Courier exchange for onboarding DMs.

    One per process, so a reply is found by the nonce its channel was opened
    with, and backed by the Authority's vault, so the open channel outlives a
    cold start.
    """
    global _exchange
    if _exchange is None:
        from tollbooth.nostr_credentials import NostrCredentialExchange
        from tollbooth.vaults.neon import NeonCredentialVault

        _exchange = NostrCredentialExchange(
            nsec=_get_settings().tollbooth_nostr_operator_nsec,
            relays=_resolve_relays(),
            templates=ONBOARDING_TEMPLATES,
            credential_vault=NeonCredentialVault(neon_vault=await _get_runtime().vault()),
        )
    return _exchange


async def _dm_owner(runtime: OperatorRuntime, text: str) -> None:
    """Best-effort DM to the Authority's owner. Never raises."""
    try:
        exchange = await _nostr_exchange()
        await asyncio.to_thread(exchange.send_dm, runtime.operator_npub(), text)
    except Exception as exc:  # noqa: BLE001
        logger.info("Owner DM skipped: %s", exc)


def _resolve_npub_or_operator(npub: str) -> str:
    """Resolve npub, falling back to operator's own npub if empty."""
    try:
        return resolve_npub(npub)
    except ValueError:
        return _get_runtime().operator_npub()


# ======================================================================
# Authority config persistence (vault-backed)
# ======================================================================


async def _get_authority_npub() -> str | None:
    global _cached_authority_npub
    if _cached_authority_npub is not None:
        return _cached_authority_npub
    try:
        vault = await _get_runtime().vault()
        npub = await vault.get_config("authority_npub")
        if npub:
            _cached_authority_npub = npub
            return npub
    except Exception:
        logger.warning(
            "authority_npub vault read failed; treating as unset", exc_info=True,
        )
    return None


async def _set_authority_npub(npub: str) -> None:
    # Persist the certification-critical authority npub durably BEFORE caching
    # it. A vault-write failure must propagate — caching in memory while the
    # write failed would report false durability (the key vanishes on restart,
    # silently breaking certificate verification). The caller aborts activation
    # on this exception rather than reporting a phantom success.
    global _cached_authority_npub
    vault = await _get_runtime().vault()
    await vault.set_config("authority_npub", npub)
    _cached_authority_npub = npub


# ======================================================================
# Oracle MCP-to-MCP helpers
# ======================================================================


class OracleRegistryError(Exception):
    """Raised when the Oracle's registry tool returned ``success: false``.

    Previously the helpers below silently squashed inner failures into the
    outer ``commit_url`` field as raw JSON, which caused outer tools to
    return ``success: true`` with a garbage commit_url. Raising forces the
    outer ``try/except`` to handle the failure explicitly — log-and-continue
    for ``register_operator`` (local ledger still consistent), propagate as
    ``{"success": false, "error": ...}`` for update/deregister.
    """

    def __init__(self, message: str, *, raw: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw = raw or {}


def _parse_oracle_commit_url(result: Any) -> str:
    """Extract a ``commit_url`` from an Oracle tool result, or raise.

    Oracle tools always return JSON of the form::

        {"success": true, "commit_url": "https://github.com/...", ...}

    or::

        {"success": false, "error": "service_url is required", ...}

    The legacy ``.get("commit_url", block.text)`` fallback fed the failure
    body straight into the outer field. This parser detects ``success:
    false`` and raises so the caller's exception path runs.
    """
    import json

    if not hasattr(result, "content"):
        return str(result)
    for block in result.content:
        if not hasattr(block, "text"):
            continue
        try:
            payload = json.loads(block.text)
        except (json.JSONDecodeError, TypeError):
            return block.text
        if isinstance(payload, dict) and payload.get("success") is False:
            raise OracleRegistryError(
                payload.get("error") or "Oracle registry call returned success=false.",
                raw=payload,
            )
        if isinstance(payload, dict):
            return str(payload.get("commit_url") or "")
        return block.text
    return str(result)


async def _oracle_client() -> Any:
    """A client of the Oracle whose every request this Authority signs.

    The Oracle commits registry changes only for a registered Authority, which
    it knows by the request's signature.
    """
    from tollbooth.mcp_client import service_client
    from tollbooth.registry import resolve_oracle_service

    oracle_info = await resolve_oracle_service(_get_nostr_signer().npub)
    return service_client(oracle_info["url"], _get_runtime()._get_nsec())


async def _register_operator_via_oracle(
    operator_npub: str,
    display_name: str,
    service_url: str,
    authority_npub: str,
) -> str:
    async with await _oracle_client() as client:
        result = await client.call_tool(
            "register_operator",
            {
                "operator_npub": operator_npub,
                "display_name": display_name,
                "service_url": service_url,
                "authority_npub": authority_npub,
            },
        )
        return _parse_oracle_commit_url(result)


async def _update_operator_via_oracle(
    operator_npub: str,
    service_url: str,
    display_name: str,
    authority_npub: str,
) -> str:
    args: dict = {"operator_npub": operator_npub, "authority_npub": authority_npub}
    if service_url:
        args["service_url"] = service_url
    if display_name:
        args["display_name"] = display_name

    async with await _oracle_client() as client:
        result = await client.call_tool("update_operator", args)
        return _parse_oracle_commit_url(result)


async def _deregister_operator_via_oracle(
    operator_npub: str,
    authority_npub: str,
) -> str:
    async with await _oracle_client() as client:
        result = await client.call_tool(
            "deregister_operator",
            {"operator_npub": operator_npub, "authority_npub": authority_npub},
        )
        return _parse_oracle_commit_url(result)


async def _register_via_oracle(
    authority_npub: str,
    display_name: str,
    service_url: str,
    upstream_authority_npub: str,
) -> str:
    async with await _oracle_client() as client:
        result = await client.call_tool(
            "register_authority",
            {
                "authority_npub": authority_npub,
                "display_name": display_name,
                "service_url": service_url,
                "upstream_authority_npub": upstream_authority_npub,
            },
        )
        return _parse_oracle_commit_url(result)


async def _resolve_own_service_url() -> str:
    signer = _get_nostr_signer()
    s = _get_settings()
    registry = DPYCRegistry(
        url=DEFAULT_REGISTRY_URL,
        cache_ttl_seconds=s.dpyc_registry_cache_ttl_seconds,
    )
    try:
        member = await registry.check_membership(signer.npub)
        services = member.get("services", [])
        if services:
            return services[0]["url"]
        raise ValueError(
            f"Authority {signer.npub[:16]}... has no services registered."
        )
    except RegistryError:
        raise ValueError(
            f"Authority {signer.npub[:16]}... not found in DPYC registry."
        )
    finally:
        await registry.close()


def _signed_by(npub: str) -> bool:
    """True when the request in flight was signed by ``npub``'s own key."""
    from tollbooth.holder_auth import signed_npub
    from tollbooth.runtime import resolve_npub

    signer = signed_npub()
    try:
        return signer is not None and resolve_npub(signer) == resolve_npub(npub)
    except ValueError:
        return False


async def _require_owner(runtime: OperatorRuntime) -> dict[str, Any] | None:
    """The request must prove the Authority's own npub: its owner is acting.

    Returns ``None`` on success or a structured ``AUTHORITY_CONSENT_REQUIRED``
    error to return verbatim.
    """
    from tollbooth.constants import ErrorCode as _EC

    authority_npub = runtime.operator_npub()
    if await runtime.prove_caller(authority_npub) is None:
        return None
    return {
        "success": False,
        "error_code": _EC.AUTHORITY_CONSENT_REQUIRED,
        "error": (
            f"Only the Authority's owner may do this: the request must prove "
            f"the Authority's npub ({authority_npub[:16]}...). Sign it with "
            f"that npub's key, or prove it into an OAuth connection with "
            f"request_npub_proof / receive_npub_proof."
        ),
        "authority_npub": authority_npub,
    }


# Bootstrap config is a NIP-33 replaceable (kind 30078), but free public relays
# still drop events — nos.lol purged months-old configs on 2026-09-29 and left
# four operators with no copy anywhere. The weekly audit (a GitHub Action calls
# ``audit_bootstrap_configs``) republishes any config held too thinly or last
# sent longer ago than this, so every run refreshes ``created_at``.
_BOOTSTRAP_REFRESH_AFTER_SECONDS = 6 * 24 * 3600
# Publishes in flight at once during an audit; each already fans out to every relay.
_AUDIT_CONCURRENCY = 4


async def _resend_bootstrap_dm(npub: str) -> bool:
    try:
        vault = await _get_runtime().vault()
        from tollbooth.authority.tenant_provisioner import (
            get_all_operator_config,
            store_operator_config,
        )
        config = await get_all_operator_config(vault, npub)
        neon_url = config.get("neon_database_url")
        schema = config.get("schema", "")
        if not neon_url:
            return False
        from tollbooth.bootstrap_relay import send_bootstrap_config
        signer = _get_nostr_signer()
        # The publish is blocking websocket I/O — keep it off the event loop.
        # fan_out inside send_bootstrap_config bounds the whole set to one
        # budget instead of a serial 10 s-per-relay walk.
        result = await asyncio.to_thread(
            send_bootstrap_config,
            authority_nsec=signer.nsec,
            operator_npub=npub,
            config={"neon_database_url": neon_url, "schema": schema},
        )
        accepted, rejected = int(result.accepted), int(result.rejected)
        if accepted > 0:
            await store_operator_config(
                vault, npub, "bootstrap_dm_sent_at", str(int(time.time())),
            )
            logger.info(
                "Bootstrap config (re)sent to operator %s: "
                "accepted=%d rejected=%d",
                npub[:16], accepted, rejected,
            )
            return True
        logger.warning(
            "Bootstrap config resend for %s did not publish: "
            "accepted=%d rejected=%d",
            npub[:16], accepted, rejected,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bootstrap DM resend failed for %s: %s", npub[:16], exc)
        return False


async def _audit_bootstrap_configs() -> dict[str, Any]:
    """Measure every operator's config coverage, republish what needs it, measure again.

    Runs in the foreground: Horizon freezes a process between requests, so a
    background sweep rarely finishes (most configs went unrefreshed from June to
    September). An operator is republished when fewer than three relays hold its
    config, or when it was last sent more than six days ago. The report carries
    counts and npub prefixes only — never a config value.
    """
    from pynostr.key import PublicKey  # type: ignore[import-untyped]

    from tollbooth.authority.tenant_provisioner import get_operator_config_value
    from tollbooth.bootstrap_relay import MIN_HOLDERS, config_coverage
    from tollbooth.relay_registry import get_relays

    vault = await _get_runtime().vault()
    t = vault._t if hasattr(vault, "_t") else (lambda x: x)
    result = await vault._execute(
        f"SELECT DISTINCT npub FROM {t('bootstrap_config')} "
        "WHERE key = 'neon_database_url'"
    )
    npubs = [
        (row[0] if isinstance(row, list) else row.get("npub", ""))
        for row in result.get("rows", [])
    ]
    npubs = [n for n in npubs if n]
    hexes = {n: PublicKey.from_npub(n).hex() for n in npubs}
    authority_hex = _get_nostr_signer().pubkey_hex
    relays = get_relays()

    def coverage() -> dict[str, tuple[str, ...]]:
        return config_coverage(authority_hex, list(hexes.values()), relays)

    before = await asyncio.to_thread(coverage)
    now = time.time()
    due: dict[str, str] = {}
    for npub in npubs:
        if len(before[hexes[npub]]) < MIN_HOLDERS:
            due[npub] = "thin"
            continue
        stamp = await get_operator_config_value(vault, npub, "bootstrap_dm_sent_at")
        if now - (int(stamp) if stamp else 0) >= _BOOTSTRAP_REFRESH_AFTER_SECONDS:
            due[npub] = "stale"

    gate = asyncio.Semaphore(_AUDIT_CONCURRENCY)

    async def republish(npub: str) -> bool:
        async with gate:
            return await _resend_bootstrap_dm(npub)

    sent = dict(zip(due, await asyncio.gather(*(republish(n) for n in due)), strict=True))
    after = await asyncio.to_thread(coverage) if due else before

    operators = [
        {
            "operator": f"{npub[:16]}...",
            "holders_before": len(before[hexes[npub]]),
            "holders_after": len(after[hexes[npub]]),
            "republished": sent.get(npub, False),
            "reason": due.get(npub, ""),
        }
        for npub in npubs
    ]
    thin = [f"{n[:16]}..." for n in npubs if len(after[hexes[n]]) < MIN_HOLDERS]
    logger.info(
        "Bootstrap config audit: %d operators, %d due, %d republished, %d still thin",
        len(npubs), len(due), sum(sent.values()), len(thin),
    )
    return {
        "success": True,
        "relays": len(relays),
        "operators": operators,
        "still_thin": thin,
    }


async def _provision_operator(
    runtime: OperatorRuntime,
    npub: str,
    service_url: str,
    display_name: str = "",
) -> dict[str, Any]:
    """Provision an operator: ledger row + isolated Neon tenant + registry.

    The single source of truth for *what adoption does*, run when the owner
    approves an operator's adoption request. Proof gating lives in the caller,
    never here.
    """
    cache = await runtime.ledger_cache()
    # Materialize the operator's ledger row. Nothing is being changed — the
    # WRITE is the point, so a later credit has a row to CAS against. Routed
    # through the write-through path so provisioning can't be left holding a
    # dirty cache entry that never lands.
    try:
        await cache.mutate(npub, lambda _ledger: True)
    except (LedgerUnavailableError, LedgerWriteError) as exc:
        # Not fatal: the row is created on first credit anyway. Provisioning
        # continues so the Neon tenant and registry entry still get made.
        logger.warning(
            "Could not materialize ledger row for %s during provisioning: %s: %s",
            npub[:20], type(exc).__name__, exc,
        )
    ledger = await cache.get(npub)

    # Provision isolated Neon schema with per-operator role
    neon_url = ""
    try:
        vault = await runtime.vault()
        from tollbooth.authority.tenant_provisioner import (
            ensure_bootstrap_table,
            neon_url_for_operator,
            provision_operator_schema,
            store_operator_config,
        )
        await ensure_bootstrap_table(vault)
        s = _get_settings()
        schema, password = await provision_operator_schema(
            vault, npub,
            base_url=s.neon_database_url,
            authority_nsec_hex=getattr(s, "tollbooth_nostr_operator_nsec_hex", ""),
        )
        if s.neon_database_url:
            neon_url = neon_url_for_operator(s.neon_database_url, schema, password)
            await store_operator_config(vault, npub, "neon_database_url", neon_url)
            await store_operator_config(vault, npub, "schema", schema)
            if getattr(vault, "_cipher", None):
                encrypted_pw = vault._encrypt(password)
            else:
                encrypted_pw = password
            await store_operator_config(vault, npub, "role_password", encrypted_pw)
            logger.info("Provisioned Neon tenant for operator %s schema=%s (role-isolated)", npub[:16], schema)
            await _resend_bootstrap_dm(npub)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neon tenant provisioning failed (non-fatal): %s", exc)

    # Register in community registry via Oracle
    commit_url = ""
    try:
        signer = _get_nostr_signer()
        commit_url = await _register_operator_via_oracle(
            operator_npub=npub,
            # The operator's chosen name (from the adopting app's UI); only fall back to a
            # truncated npub when no name was supplied, so the roster is never named by npub.
            display_name=display_name or (npub[:16] + "..."),
            service_url=service_url,
            authority_npub=signer.npub,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Oracle operator registration failed (non-fatal): %s", exc)

    return {
        "success": True,
        "npub": npub,
        "balance_sats": ledger.balance_api_sats,
        "dpyc_npub": npub,
        "neon_database_url": neon_url,
        "commit_url": commit_url,
        "message": f"Operator {npub} registered. Use purchase_credits to fund your balance.",
    }


# ======================================================================
# register_authority_tools — the public mixin
# ======================================================================


def register_authority_tools(
    mcp: Any,
    runtime: OperatorRuntime,
    slug: str = "authority",
) -> None:
    """Register all Authority @tool definitions on the given FastMCP app.

    Call once at module level after ``register_standard_tools``. Captures
    ``runtime`` and ``slug`` via closure; module-level state singletons
    (settings, signer, replay tracker, onboarding state) are lazy-init'd
    by the tools themselves on first call.

    Args:
        mcp: The FastMCP app instance.
        runtime: The OperatorRuntime instance, configured by the caller
            with ``vault_source="env"`` (Authorities self-provision Neon),
            ``purchase_mode="auto"`` (derive certify-up from the registry
            chain), ``ots_enabled=True``, and the Authority's tool_registry
            (``STANDARD_IDENTITIES`` merged with ``AUTHORITY_TOOL_REGISTRY``).
        slug: Tool name prefix (default ``"authority"``).
    """
    global _runtime
    _runtime = runtime

    # Inject search_path=authority into NEON_DATABASE_URL so the
    # Authority's own tables land in the "authority" schema, not "public".
    # OperatorRuntime reads NEON_DATABASE_URL lazily — must happen before
    # first vault() call.
    raw_neon_url = os.environ.get("NEON_DATABASE_URL", "")
    if raw_neon_url and "search_path" not in raw_neon_url:
        from tollbooth.authority.tenant_provisioner import neon_url_with_schema
        os.environ["NEON_DATABASE_URL"] = neon_url_with_schema(raw_neon_url, "authority")

    tool = make_slug_tool(mcp, slug, runtime.identity_for)

    # ------------------------------------------------------------------
    # Operator lifecycle tools
    # ------------------------------------------------------------------

    @tool
    async def update_operator(
        npub: Annotated[str, Field(description="Your operator npub; the request must prove it.")] = "",
        service_url: Annotated[str, Field(description="New MCP endpoint URL (leave empty to keep current).")] = "",
        display_name: Annotated[str, Field(description="New display name (leave empty to keep current).")] = "",
    ) -> dict[str, Any]:
        """Ask this Authority to change your community registry entry.

        The operator acts: the request must prove ``npub``. The change waits in
        the owner's queue (``list_operator_requests``) and takes effect when the
        Authority owner approves it, so the registry never changes under this
        Authority's name without its owner seeing it.
        """
        if err := await runtime.prove_caller(npub):
            return err
        if not service_url and not display_name:
            return {"success": False, "error": "Nothing to update. Provide service_url and/or display_name."}
        npub = resolve_npub(npub)
        try:
            vault = await runtime.vault()
            await operator_requests.ensure_schema(vault)
            await operator_requests.upsert_pending(
                vault, operator_requests.UPDATE, npub,
                service_url=service_url, display_name=display_name,
            )
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not record request: {exc}"}
        await _notify_owner(f"Operator {npub[:16]}... asks to update its registry entry.")
        return {
            "success": True,
            "status": operator_requests.PENDING,
            "operator_npub": npub,
            "message": "Update requested; it takes effect when the Authority's owner approves it.",
        }

    @tool
    async def deregister_operator(
        npub: Annotated[str, Field(description="Nostr npub of the Operator to remove.")] = "",
    ) -> dict[str, Any]:
        """Remove an Operator from the DPYC community registry.

        Either party may: the operator leaving (the request proves ``npub``),
        or the Authority's owner disowning it (the request proves the
        Authority's npub). Its ledger and database stay, so a later adoption
        resumes where it left off.
        """
        try:
            npub = resolve_npub(npub)
        except ValueError as exc:
            return {"success": False, "error_code": ErrorCode.NPUB_INVALID, "error": str(exc)}
        if await runtime.prove_caller(npub) is not None and (err := await _require_owner(runtime)):
            return err
        try:
            commit_url = await _deregister_operator_via_oracle(
                operator_npub=npub,
                authority_npub=_get_nostr_signer().npub,
            )
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Deregistration failed: {exc}"}
        try:
            vault = await runtime.vault()
            await operator_requests.ensure_schema(vault)
            await operator_requests.mark(vault, operator_requests.ADOPTION, npub, operator_requests.CLOSED)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not close the adoption record of %s: %s", npub[:16], exc)
        return {
            "success": True,
            "commit_url": commit_url,
            "message": f"Operator {npub[:16]}... removed from community registry.",
        }

    @tool
    async def get_operator_config(
        npub: Annotated[str, Field(description="Your Nostr npub (bech32).")] = "",
    ) -> dict[str, Any]:
        """Retrieve operator bootstrap configuration (Neon URL, schema).

        The request must prove ``npub``.
        """
        err = await runtime.prove_caller(npub)
        if err:
            return err

        try:
            vault = await runtime.vault()
            from tollbooth.authority.tenant_provisioner import get_all_operator_config
            config = await get_all_operator_config(vault, npub)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Failed to retrieve config: {exc}"}

        if not config:
            return {"success": False, "error": f"No configuration found for {npub[:16]}..."}

        await _resend_bootstrap_dm(npub)

        # Filter internal secrets from response
        filtered = {k: v for k, v in config.items() if k != "role_password"}

        return {
            "success": True,
            "npub": npub,
            "config": filtered,
            "message": f"Bootstrap configuration for {npub[:16]}... ({len(config)} entries).",
        }

    @tool
    async def operator_status(
        npub: Annotated[str, Field(description="Nostr public key (npub1...). Defaults to operator identity if empty.")] = "",
    ) -> dict[str, Any]:
        """View registration status, balance summary, and the Authority's Nostr npub.

        When an explicit ``npub`` is provided, the request must prove it —
        otherwise anyone could enumerate balances by walking the community
        registry. When ``npub`` is empty, falls back to the Authority's own
        operator identity (self-inspection is always allowed).
        """
        if npub:
            err = await runtime.prove_caller(npub)
            if err:
                return err
        user_id = _resolve_npub_or_operator(npub)
        s = _get_settings()
        nostr_signer = _get_nostr_signer()

        cache = await runtime.ledger_cache()
        ledger = await cache.get(user_id)

        result: dict[str, Any] = {
            "npub": user_id,
            "dpyc_npub": user_id,
            "registered": True,
            "balance_sats": ledger.balance_api_sats,
            "total_deposited_sats": ledger.total_deposited_api_sats,
            "total_consumed_sats": ledger.total_consumed_api_sats,
            "authority_npub": nostr_signer.npub,
            "nostr_certificate_enabled": True,
        }

        if s.dpyc_enforce_membership:
            result["dpyc_registry_enforcement"] = True

        result["vault_configured"] = bool(s.neon_database_url)
        result["vault_backend"] = "neon" if s.neon_database_url else "unconfigured"
        result["cache_health"] = cache.health()

        return result

    # ------------------------------------------------------------------
    # certify_credits — the revenue tool (ad valorem via paid_tool)
    # ------------------------------------------------------------------

    @tool
    @runtime.paid_tool(capability_uuid("certify_credits"))
    async def certify_credits(
        npub: Annotated[
            str,
            Field(description="The operator's DPYC npub; the request must prove it."),
        ] = "",
        amount_sats: Annotated[
            int,
            Field(description="The total purchase amount in satoshis. Must be positive."),
        ] = 0,
    ) -> dict[str, Any]:
        """Certify a purchase order: return a Schnorr-signed Nostr event certificate.

        The paid_tool decorator handles the ad valorem fee debit and stores
        the cost in runtime._last_debit_cost. No recomputation needed.

        Called by operator MCP servers (not end users) when a patron purchases credits.
        """
        if amount_sats <= 0:
            return {"success": False, "error": "amount_sats must be positive."}

        s = _get_settings()
        nostr_signer = _get_nostr_signer()
        replay = _get_replay_tracker()

        # Use the fee computed and debited by the paid_tool decorator — single
        # source of truth, no recomputation, no divergence risk.
        fee_sats = getattr(runtime, "_last_debit_cost", 0)
        net_sats = amount_sats - fee_sats

        # DPYC registry membership check (fail closed).
        # Membership is an expected lifecycle gate, not an exception.
        # Refund the certification fee and return a structured error
        # so the caller can route directly to the recovery flow.
        registry = _get_dpyc_registry()
        if registry is not None:
            try:
                await registry.check_membership(npub)
            except RegistryError as e:
                await runtime.rollback_debit(capability_uuid("certify_credits"), npub)
                return {
                    "success": False,
                    "error_code": "dpyc_membership_required",
                    "error": f"DPYC membership check failed: {e}",
                    "next_steps": [
                        "Confirm the operator npub is registered in dpyc-community members.json",
                        "If unregistered, register via the DPYC Oracle's registration flow",
                    ],
                }

        # Build claims and sign certificate
        jti = uuid.uuid4().hex
        expiration = int(time.time()) + s.certificate_ttl_seconds

        claims = {
            "sub": npub,
            "amount_sats": amount_sats,
            "fee_sats": fee_sats,
            "net_sats": net_sats,
            "dpyc_protocol": "dpyp-01-base-certificate",
        }

        replay.check_and_record(jti)

        nostr_event_json = nostr_signer.sign_certificate_event(
            claims=claims,
            jti=jti,
            operator_npub=npub,
            expiration=expiration,
        )

        # The fee debit is ALREADY durable: a non-zero charge goes through
        # `LedgerCache.mutate()`, which CAS-writes to Postgres before this tool
        # body ever runs. The explicit flush that used to sit here was a leftover
        # from the pre-write-through ledger, and it did harm — losing a CAS race
        # made it log "Failed to persist fee debit" about a fee that was, in
        # fact, persisted.

        return {
            "success": True,
            "certificate": nostr_event_json,
            "jti": jti,
            "amount_sats": amount_sats,
            "fee_sats": fee_sats,
            "net_sats": net_sats,
            "expires_at": expiration,
        }

    @tool
    async def audit_bootstrap_configs() -> dict[str, Any]:
        """Audit every operator's bootstrap config on the relays, and republish what needs it.

        Republishes a config when fewer than three relays hold it, or when it was
        last sent more than six days ago; then measures again. A GitHub Action
        runs this weekly. Free and unauthenticated: it reveals only relay counts
        anyone can take, and re-sending the Authority's own config is idempotent.
        """
        return await _audit_bootstrap_configs()

    # ------------------------------------------------------------------
    # DPYC membership diagnostic
    # ------------------------------------------------------------------

    @tool
    async def check_dpyc_membership(npub: str) -> dict[str, Any]:
        """Look up an npub in the DPYC community registry.

        Args:
            npub: The Nostr public key (npub1...) to look up.
        """
        s = _get_settings()
        registry = DPYCRegistry(
            url=DEFAULT_REGISTRY_URL,
            cache_ttl_seconds=s.dpyc_registry_cache_ttl_seconds,
        )
        try:
            member = await registry.check_membership(npub)
            return {"success": True, "member": member}
        except RegistryError as e:
            return {"success": False, "error": str(e)}
        finally:
            await registry.close()

    # ------------------------------------------------------------------
    # Deferred operator adoption (the courtship)
    # ------------------------------------------------------------------

    async def _notify_owner(text: str) -> None:
        """Best-effort "check your queue" DM to the Authority owner. The
        durable queue (``list_operator_requests``) is the source of truth."""
        await _dm_owner(runtime, text + " Review it with list_operator_requests in the Pricing Studio.")

    @tool
    async def receive_adoption_request(
        operator_npub: Annotated[
            str,
            Field(description="The operator's Nostr npub requesting adoption; the request must be signed by it."),
        ] = "",
        service_url: Annotated[str, Field(description="The operator's MCP endpoint URL.")] = "",
        display_name: Annotated[str, Field(description="The operator's name for the community roster.")] = "",
        note: Annotated[str, Field(description="An optional message for the Authority's owner.")] = "",
    ) -> dict[str, Any]:
        """Inbound: record an operator's request to be adopted by this Authority.

        Called MCP-to-MCP by the operator's ``request_adoption`` on a connection
        signed by the operator's key. Records a durable ``pending`` request and
        fires a best-effort owner-notification DM. Does NOT provision —
        provisioning waits for the owner's ``approve_operator_request``.
        """
        if not operator_npub:
            return {"success": False, "error": "operator_npub is required."}
        if not _signed_by(operator_npub):
            return {
                "success": False,
                "error_code": ErrorCode.PROOF_INVALID,
                "error": (
                    "An adoption request must be signed with operator_npub's key. "
                    "Call request_adoption on the operator."
                ),
            }
        try:
            vault = await runtime.vault()
            await operator_requests.ensure_schema(vault)
            await operator_requests.upsert_pending(
                vault, operator_requests.ADOPTION, operator_npub,
                service_url=service_url, display_name=display_name, note=note,
            )
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not record request: {exc}"}

        await _notify_owner(f"Operator {operator_npub[:16]}... ({service_url}) requests adoption.")
        return {
            "success": True,
            "status": operator_requests.PENDING,
            "operator_npub": operator_npub,
            "message": (
                "Adoption request received and pending the Authority owner's "
                "decision. Poll get_adoption_status for progress."
            ),
        }

    @tool
    async def list_operator_requests() -> dict[str, Any]:
        """Owner queue: pending operator requests, adoptions and registry
        updates alike, each naming its ``kind``.

        Restricted to the Authority's owner. This is the review-on-your-own-time
        surface the Pricing Studio renders.
        """
        if err := await _require_owner(runtime):
            return err
        try:
            vault = await runtime.vault()
            await operator_requests.ensure_schema(vault)
            await operator_requests.prune_expired(vault)
            pending = await operator_requests.list_pending(vault)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not list requests: {exc}"}
        return {"success": True, "count": len(pending), "requests": pending}

    # ------------------------------------------------------------------
    # Neon persistence health — the accounting store are the Authority's charge
    # ------------------------------------------------------------------

    async def _notify_owner_of_neon_402(operator_npub: str, detail: str) -> None:
        """Best-effort heads-up DM to the Authority owner that an operator's
        store is 402-locked. Never raises; the durable row is the record."""
        await _dm_owner(runtime, (
            f"⚠ Operator {operator_npub[:16]}... reports its Neon store is "
            "402-locked (compute/storage quota exhausted) — paid tools are "
            "down for its patrons. Review with network_persistence_health / "
            "list_neon_alerts and restore capacity (upgrade the plan or wait "
            "for the quota reset)."
        ))

    @tool
    async def receive_neon_402_alert(
        npub: Annotated[
            str,
            Field(description="The reporting operator's Nostr npub (the one whose store is locked)."),
        ] = "",
        detail: Annotated[
            str,
            Field(description="Short, credential-free error summary (the Neon 402 message)."),
        ] = "",
    ) -> dict[str, Any]:
        """Inbound: an operator reports its Neon store is 402-locked.

        Called MCP-to-MCP by the operator's runtime the instant it catches a
        Neon HTTP 402 on its own database, on a connection signed by the
        operator's key. Records a
        durable latest-state row, and fires a best-effort owner-notification
        DM. This is how the Authority learns the store is dark BEFORE a patron
        files a complaint.
        """
        if not npub:
            return {"success": False, "error": "npub is required."}
        if not _signed_by(npub):
            return {
                "success": False,
                "error_code": ErrorCode.PROOF_INVALID,
                "error": "The alert must be signed with npub's key.",
            }
        try:
            vault = await runtime.vault()
            await neon_alert_store.ensure_schema(vault)
            await neon_alert_store.record(vault, npub, detail)
        except Exception as exc:  # noqa: BLE001
            # The Authority's OWN store may be down too (shared project). Still
            # try to reach the human, and report honestly rather than 500.
            await _notify_owner_of_neon_402(npub, detail)
            return {"success": False, "error": f"Could not record alert: {exc}"}

        await _notify_owner_of_neon_402(npub, detail)
        return {
            "success": True,
            "operator_npub": npub,
            "message": (
                "Neon-402 alert recorded. The Authority owner has been notified "
                "to restore capacity."
            ),
        }

    @tool
    async def list_neon_alerts(
    ) -> dict[str, Any]:
        """Owner queue: operators that reported a Neon-402 (store locked).

        Restricted to the Authority owner. A companion to network_persistence_health:
        this is the reactive list (operators that already went dark); the health
        tool adds the proactive per-project compute posture.
        """
        err = await _require_owner(runtime)
        if err:
            return err
        try:
            vault = await runtime.vault()
            await neon_alert_store.ensure_schema(vault)
            alerts = await neon_alert_store.list_all(vault)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not list alerts: {exc}"}
        return {"success": True, "count": len(alerts), "alerts": alerts}

    @tool
    async def network_persistence_health(
    ) -> dict[str, Any]:
        """Owner view: the health of the DPYC economy's accounting store (Neon).

        Restricted to the Authority owner. Three layers, from most to least
        proactive:

        1. ``projects`` — if a Neon API key is configured (NEON_API_KEY), the
           per-project compute-quota posture across the org: hours used, %,
           reset date, and a status ladder (ok/warning/critical/exhausted) so a
           project can be topped up BEFORE it 402s. ``configured=false`` when no
           key is present (deliver one to enable the proactive watch).
        2. ``own_store`` — reactive self-detection: whether the Authority's OWN
           database answers, or is itself 402-locked. Always available.
        3. ``operator_alerts`` — operators that reported a 402 (from
           receive_neon_402_alert). Reactive, but immediate.
        """
        err = await _require_owner(runtime)
        if err:
            return err

        from tollbooth.vaults.neon import NeonQueryError

        # (2) Reactive self-probe of the Authority's own store.
        own_store: dict[str, Any]
        try:
            vault = await runtime.vault()
            await vault._execute("SELECT 1 AS one")
            own_store = {"status": "ok", "detail": ""}
        except NeonQueryError as exc:
            status = "quota_exceeded" if getattr(exc, "status", 0) == 402 else "error"
            own_store = {"status": status, "detail": str(exc)[:300]}
            vault = None
        except Exception as exc:  # noqa: BLE001
            own_store = {"status": "unreachable", "detail": str(exc)[:300]}
            vault = None

        # (3) Operator-reported alerts (best-effort — needs a live vault).
        operator_alerts: list[dict[str, Any]] = []
        if vault is not None:
            try:
                await neon_alert_store.ensure_schema(vault)
                operator_alerts = await neon_alert_store.list_all(vault)
            except Exception as exc:  # noqa: BLE001
                logger.info("Could not read operator Neon alerts: %s", exc)

        # (1) Proactive per-project compute posture. neon_api_key/neon_org_id are OPTIONAL
        # Authority secrets, courier-delivered and read from the vault — never env. Absent the
        # key, the watch degrades to the reactive self-probe + operator alerts above.
        neon_creds: dict[str, str] = {}
        try:
            neon_creds, neon_situation = await runtime._load_vault_creds(
                OPERATOR_CREDENTIAL_TEMPLATE.service,
            )
            if neon_situation:
                logger.info(
                    "Could not read Neon watch secrets from vault: %s", neon_situation,
                )
        except Exception as exc:  # noqa: BLE001
            logger.info("Could not load Neon watch secrets from vault: %s", exc)
        neon_api_key = (neon_creds.get("neon_api_key") or "").strip()
        neon_org_id = (neon_creds.get("neon_org_id") or "").strip()
        if not neon_api_key:
            neon_api: dict[str, Any] = {
                "configured": False,
                "hint": (
                    "Deliver NEON_API_KEY (org-scoped) via Secure Courier to enable proactive "
                    "per-project compute-quota monitoring, so a project can be topped up before "
                    "it 402s. Without it, only reactive detection is available."
                ),
                "projects": [],
            }
        else:
            from tollbooth.authority.neon_admin import NeonAdminClient
            try:
                # Narrow the watch to THIS Authority's own project by matching its
                # DSN host (from the vault it just probed) — no extra config; the
                # org key sees many projects but the owner cares about one.
                own_host = getattr(vault, "endpoint_host", "") if vault is not None else ""
                client = NeonAdminClient(neon_api_key, org_id=neon_org_id)
                usage = await client.project_usage(own_db_host=own_host)
                neon_api = {
                    "configured": True,
                    "projects": [u.to_dict() for u in usage],
                }
                # When compute reads 'unknown', say WHY — the consumption_history
                # probe's own reason — so it's never a mute blank.
                if client.last_usage_note:
                    neon_api["usage_note"] = client.last_usage_note
            except Exception as exc:  # noqa: BLE001
                neon_api = {
                    "configured": True,
                    "error": f"Neon API read failed: {str(exc)[:200]}",
                    "projects": [],
                }

        # Roll up a single worst-case posture for an at-a-glance dot.
        project_statuses = [p.get("status") for p in neon_api.get("projects", [])]
        ladder = ["exhausted", "critical", "warning", "ok"]
        overall = "ok"
        if own_store["status"] == "quota_exceeded" or operator_alerts:
            overall = "exhausted"
        else:
            for rung in ladder:
                if rung in project_statuses:
                    overall = rung
                    break
            if own_store["status"] in ("error", "unreachable"):
                overall = "critical" if overall == "ok" else overall

        return {
            "success": True,
            "overall_status": overall,
            "own_store": own_store,
            "operator_alerts": operator_alerts,
            "operator_alert_count": len(operator_alerts),
            "neon_api": neon_api,
        }

    _KIND = Annotated[
        str,
        Field(description="The request's kind, as list_operator_requests names it: 'adoption' or 'update'."),
    ]

    @tool
    async def approve_operator_request(
        operator_npub: Annotated[str, Field(description="The operator whose request to approve.")] = "",
        kind: _KIND = operator_requests.ADOPTION,
    ) -> dict[str, Any]:
        """Approve a pending operator request (owner only).

        An adoption provisions the operator: ledger row, isolated Neon tenant
        and registry entry. An update applies the requested service URL or
        display name to the registry entry.
        """
        if err := await _require_owner(runtime):
            return err
        try:
            vault = await runtime.vault()
            await operator_requests.ensure_schema(vault)
            row = await operator_requests.get(vault, kind, operator_npub)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not read request: {exc}"}
        if row is None:
            return {
                "success": False,
                "error_code": ErrorCode.ADOPTION_NOT_FOUND,
                "error": f"No {kind} request found for {operator_npub[:16]}...",
            }
        if row.get("status") != operator_requests.PENDING:
            return {
                "success": False,
                "error_code": ErrorCode.ADOPTION_ALREADY_PROVISIONED,
                "error": f"That {kind} request is already {row.get('status')}.",
            }

        if kind == operator_requests.ADOPTION:
            result = await _provision_operator(
                runtime, operator_npub, row.get("service_url", ""), row.get("display_name", ""),
            )
            await operator_requests.mark(vault, kind, operator_npub, operator_requests.PROVISIONED)
            return result

        try:
            commit_url = await _update_operator_via_oracle(
                operator_npub=operator_npub,
                service_url=row.get("service_url", ""),
                display_name=row.get("display_name", ""),
                authority_npub=_get_nostr_signer().npub,
            )
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Update failed: {exc}"}
        await operator_requests.mark(vault, kind, operator_npub, operator_requests.APPLIED)
        await _resend_bootstrap_dm(operator_npub)
        return {
            "success": True,
            "commit_url": commit_url,
            "message": f"Operator {operator_npub[:16]}... updated in community registry.",
        }

    @tool
    async def reject_operator_request(
        operator_npub: Annotated[str, Field(description="The operator whose request to reject.")] = "",
        kind: _KIND = operator_requests.ADOPTION,
        reason: Annotated[str, Field(description="Optional human-readable reason.")] = "",
    ) -> dict[str, Any]:
        """Reject a pending operator request (owner only)."""
        if err := await _require_owner(runtime):
            return err
        try:
            vault = await runtime.vault()
            await operator_requests.ensure_schema(vault)
            hit = await operator_requests.mark(vault, kind, operator_npub, operator_requests.REJECTED)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not reject request: {exc}"}
        if not hit:
            return {
                "success": False,
                "error_code": ErrorCode.ADOPTION_NOT_FOUND,
                "error": f"No {kind} request found for {operator_npub[:16]}...",
            }
        return {
            "success": True,
            "status": operator_requests.REJECTED,
            "operator_npub": operator_npub,
            "reason": reason,
        }

    @tool
    async def repair_operator_schema(
        operator_npub: Annotated[
            str,
            Field(description="The operator npub whose tenant-schema ownership to repair."),
        ] = "",
    ) -> dict[str, Any]:
        """Owner repair: reassign every table in an operator's tenant schema to
        the operator's own role, then re-grant DML.

        For tenants whose tables were created/owned by the provisioning role —
        the operator role then cannot CREATE INDEX on them ("must be owner"),
        which aborts the whole vault bootstrap. Unlike adoption this
        does NOT rotate the operator's DB password or re-send the bootstrap DM;
        it only fixes ownership + grants in place. Idempotent.
        """
        err = await _require_owner(runtime)
        if err:
            return err
        if not operator_npub.startswith("npub1"):
            return {"success": False, "error": "operator_npub must be a valid npub1..."}
        try:
            from tollbooth.authority.tenant_provisioner import (
                restore_operator_grants,
                schema_name_for_npub,
                transfer_schema_ownership,
            )
            vault = await runtime.vault()
            schema = schema_name_for_npub(operator_npub)
            # Ensure the Authority role can reassign ownership to the operator
            # role (membership may pre-date this repair); harmless if present.
            await vault._execute(f'GRANT "{schema}" TO CURRENT_USER')
            await transfer_schema_ownership(vault, schema)
            await restore_operator_grants(vault, schema)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Schema repair failed: {exc}"}
        return {
            "success": True,
            "operator_npub": operator_npub,
            "schema": schema,
            "message": (
                "Reassigned all table ownership in the operator's schema to its "
                "own role and re-granted DML. The operator bootstraps cleanly on "
                "its next call — no password change, no re-bootstrap."
            ),
        }

    @tool
    async def get_adoption_status(
        operator_npub: Annotated[
            str,
            Field(description="The operator npub whose request status to read."),
        ] = "",
    ) -> dict[str, Any]:
        """Read an operator's adoption-request status (free, no proof).

        Status (pending/rejected/provisioned/closed) isn't sensitive — it's
        the operator's own request — so the operator can poll it openly via
        its ``adoption_status`` tool.
        """
        try:
            vault = await runtime.vault()
            await operator_requests.ensure_schema(vault)
            row = await operator_requests.get(vault, operator_requests.ADOPTION, operator_npub)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not read status: {exc}"}
        if row is None:
            return {
                "success": False,
                "error_code": ErrorCode.ADOPTION_NOT_FOUND,
                "error": f"No adoption request found for {operator_npub[:16]}...",
            }
        return {"success": True, "status": row.get("status"), "operator_npub": operator_npub}

    # ------------------------------------------------------------------
    # Authority Onboarding (3-step Nostr DM challenge-response)
    # ------------------------------------------------------------------

    @tool
    async def register_authority_npub(
        candidate_npub: Annotated[
            str,
            Field(description="The Nostr npub of the candidate who wants to become the curator."),
        ],
    ) -> dict[str, Any]:
        """Step 1/3 of Authority onboarding — send a Nostr DM challenge to the candidate."""
        if not candidate_npub.startswith("npub1") or len(candidate_npub) < 60:
            return {"success": False, "error": "Invalid npub format."}

        existing = await _get_authority_npub()
        if existing:
            return {
                "success": False,
                "error": f"This Authority already has a curator ({existing[:16]}...).",
            }

        try:
            challenge = _onboarding.start_claim(candidate_npub)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        try:
            exchange = await _nostr_exchange()
            result = await exchange.open_channel(
                "authority_claim",
                greeting=(
                    "You are requesting to become the curator of this Authority. "
                    "Reply with: claim = @@@yes@@@ and include the nonce."
                ),
                recipient_npub=candidate_npub,
            )
            challenge.nonce = result.get("nonce", "")
        except Exception as exc:  # noqa: BLE001
            _onboarding.complete()
            return {"success": False, "error": f"Failed to send DM challenge: {exc}"}

        return {
            "success": True,
            "candidate_npub": candidate_npub,
            "phase": challenge.phase,
            "instructions": (
                f"A Nostr DM challenge has been sent to {candidate_npub[:16]}... "
                "Reply with: claim = @@@yes@@@ and the nonce. "
                "Then call confirm_authority_claim(candidate_npub)."
            ),
            "message": result.get("message", "DM sent."),
        }

    @tool
    async def confirm_authority_claim(
        candidate_npub: Annotated[
            str,
            Field(description="The Nostr npub of the candidate who replied to the DM challenge."),
        ],
    ) -> dict[str, Any]:
        """Step 2/3 of Authority onboarding — verify candidate DM, escalate to parent Authority.

        The parent Authority is resolved from THIS Authority's own entry in
        dpyc-community: whatever its ``upstream_authority_npub`` names. For
        Lonnie-Authority and NorthAmerica that's Prime; for NewEngland it's
        NorthAmerica; chain depth is transparent.
        """
        challenge = _onboarding.get()
        if challenge is None:
            return {"success": False, "error": "No active onboarding. Call register_authority_npub first."}
        if challenge.candidate_npub != candidate_npub:
            return {"success": False, "error": f"Active onboarding is for {challenge.candidate_npub[:16]}..."}
        if challenge.phase != "claim":
            return {"success": False, "error": f"Onboarding is in '{challenge.phase}' phase, not 'claim'."}

        try:
            exchange = await _nostr_exchange()
            await exchange.receive(
                sender_npub=candidate_npub, service="authority_claim", nonce=challenge.nonce,
            )
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"No valid claim DM received: {exc}"}

        try:
            signer = _get_nostr_signer()
            parent_npub = await resolve_my_parent_npub(signer.npub)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Failed to resolve parent Authority: {exc}"}

        try:
            _onboarding.promote_to_approval(parent_npub)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        try:
            opened = await exchange.open_channel(
                "authority_approval",
                greeting=(
                    f"{candidate_npub} requests to curate the Authority at "
                    f"npub {signer.npub[:16]}... "
                    "Reply with: approval = @@@yes@@@ and the nonce."
                ),
                recipient_npub=parent_npub,
            )
            challenge.nonce = opened.get("nonce", "")
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Failed to send approval request to parent Authority: {exc}"}

        return {
            "success": True,
            "candidate_npub": candidate_npub,
            "phase": "approval",
            "parent_npub": parent_npub,
            "message": (
                f"Candidate {candidate_npub[:16]}... verified. "
                f"Approval request sent to parent Authority ({parent_npub[:16]}...). "
                "Call check_authority_approval(candidate_npub) after parent responds."
            ),
        }

    @tool
    async def check_authority_approval(
        candidate_npub: Annotated[
            str,
            Field(description="The Nostr npub of the candidate awaiting parent Authority approval."),
        ],
    ) -> dict[str, Any]:
        """Step 3/3 of Authority onboarding — check parent approval, activate Authority."""
        challenge = _onboarding.get()
        if challenge is None:
            return {"success": False, "error": "No active onboarding."}
        if challenge.candidate_npub != candidate_npub:
            return {"success": False, "error": f"Active onboarding is for {challenge.candidate_npub[:16]}..."}
        if challenge.phase != "approval":
            return {"success": False, "error": f"Onboarding is in '{challenge.phase}' phase, not 'approval'."}

        parent_npub = challenge.parent_npub
        if not parent_npub:
            return {"success": False, "error": "Parent Authority npub not set."}

        try:
            exchange = await _nostr_exchange()
            await exchange.receive(
                sender_npub=parent_npub, service="authority_approval", nonce=challenge.nonce,
            )
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"No approval received from parent Authority: {exc}"}

        try:
            await _set_authority_npub(candidate_npub)
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error": (
                    f"Failed to persist authority npub to the vault: {exc}. "
                    "Activation aborted — retry once the vault is reachable."
                ),
            }

        commit_url = ""
        try:
            service_url = await _resolve_own_service_url()
            commit_url = await _register_via_oracle(
                authority_npub=candidate_npub,
                display_name=f"Authority ({candidate_npub[:16]}...)",
                service_url=service_url,
                upstream_authority_npub=parent_npub,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Oracle registration failed (Authority still activated): %s", exc)

        _onboarding.complete()

        result: dict[str, Any] = {
            "success": True,
            "candidate_npub": candidate_npub,
            "activated": True,
            "message": f"Authority curator set to {candidate_npub[:16]}... and activated.",
        }
        if commit_url:
            result["commit_url"] = commit_url
            result["message"] += f" Registered in DPYC community: {commit_url}"

        return result
