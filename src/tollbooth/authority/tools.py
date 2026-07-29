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

from tollbooth.authority import adoption_store, neon_alert_store
from tollbooth.authority.nostr_signing import AuthorityNostrSigner
from tollbooth.authority.onboarding import ONBOARDING_TEMPLATES, OnboardingState
from tollbooth.authority.replay import ReplayTracker
from tollbooth.authority.settings import AuthoritySettings
from tollbooth.constants import ErrorCode
from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.identity_proof import (
    ADOPTION_PROOF_TOOL,
    require_proof,
    verify_proof,
)
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
    ToolIdentity,
    capability_uuid,
)

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
REGISTER_OPERATOR_UUID        = "21de2356-1e49-52eb-8f28-6fed77c039d9"
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
LIST_ADOPTION_REQUESTS_UUID   = "d68a9b0b-dae6-5095-8d9a-1ed6f0b96edc"
APPROVE_ADOPTION_UUID         = "104c741c-4647-5e8c-bd6d-3ba49ea44be7"
REJECT_ADOPTION_UUID          = "7dbcfa85-f776-5306-a6dc-9032f1dcb9b3"
GET_ADOPTION_STATUS_UUID      = "45c1b7b4-93ac-52e3-ad5a-8fac1b05fad1"
# Tenant-schema ownership repair (owner consent)
REPAIR_OPERATOR_SCHEMA_UUID   = "b626ee48-dcd7-5ef2-93dd-0be12364acbb"
# Neon persistence health — 402 alerts (inbound from operators) + proactive watch
RECEIVE_NEON_402_ALERT_UUID   = "bdb94fb1-454a-5804-8885-42ba93f1446d"
LIST_NEON_ALERTS_UUID         = "b8079cbd-bc1e-579c-86fa-897a0a0641bd"
NETWORK_PERSISTENCE_HEALTH_UUID     = "07706702-fac9-50d0-b69d-d706a71b9101"


AUTHORITY_DOMAIN_TOOLS: list[ToolIdentity] = [
    ToolIdentity(
        tool_id=CERTIFY_CREDITS_UUID,
        capability="certify_credits",
        category="write",
        intent="Certify a purchase order with Schnorr-signed certificate.",
        pricing_hint_type="percent",
        pricing_hint_value=2,
        pricing_hint_param="amount_sats",
        pricing_hint_min=10,
    ),
    ToolIdentity(
        tool_id=REGISTER_OPERATOR_UUID,
        capability="register_operator",
        category="free",
        intent="Provision an operator in the Authority ledger.",
    ),
    ToolIdentity(
        tool_id=UPDATE_OPERATOR_UUID,
        capability="update_operator",
        category="free",
        intent="Update an operator's community registry entry.",
    ),
    ToolIdentity(
        tool_id=DEREGISTER_OPERATOR_UUID,
        capability="deregister_operator",
        category="free",
        intent="Remove an operator from the DPYC community registry.",
    ),
    ToolIdentity(
        tool_id=GET_OPERATOR_CONFIG_UUID,
        capability="get_operator_config",
        category="restricted",
        intent="Retrieve operator bootstrap configuration.",
    ),
    ToolIdentity(
        tool_id=OPERATOR_STATUS_UUID,
        capability="operator_status",
        category="free",
        intent="View registration status, balance, and Authority npub.",
    ),
    ToolIdentity(
        tool_id=CHECK_DPYC_MEMBERSHIP_UUID,
        capability="check_dpyc_membership",
        category="free",
        intent="Look up an npub in the DPYC community registry.",
    ),
    ToolIdentity(
        tool_id=REGISTER_AUTHORITY_NPUB_UUID,
        capability="register_authority_npub",
        category="free",
        intent="Step 1/3 of Authority onboarding — send DM challenge.",
    ),
    ToolIdentity(
        tool_id=CONFIRM_AUTHORITY_CLAIM_UUID,
        capability="confirm_authority_claim",
        category="free",
        intent="Step 2/3 of Authority onboarding — verify candidate DM.",
    ),
    ToolIdentity(
        tool_id=CHECK_AUTHORITY_APPROVAL_UUID,
        capability="check_authority_approval",
        category="free",
        intent="Step 3/3 of Authority onboarding — check parent approval.",
    ),
    # -- Deferred operator adoption --
    ToolIdentity(
        tool_id=RECEIVE_ADOPTION_REQUEST_UUID,
        capability="receive_adoption_request",
        category="free",
        intent="Inbound: record an operator's request to be adopted (operator-proof gated).",
    ),
    ToolIdentity(
        tool_id=LIST_ADOPTION_REQUESTS_UUID,
        capability="list_adoption_requests",
        category="restricted",
        intent="Operator-owner queue: list pending operator-adoption requests.",
    ),
    ToolIdentity(
        tool_id=APPROVE_ADOPTION_UUID,
        capability="approve_adoption",
        category="restricted",
        intent="Approve a pending operator-adoption request and provision the operator.",
    ),
    ToolIdentity(
        tool_id=REJECT_ADOPTION_UUID,
        capability="reject_adoption",
        category="restricted",
        intent="Reject a pending operator-adoption request.",
    ),
    ToolIdentity(
        tool_id=GET_ADOPTION_STATUS_UUID,
        capability="get_adoption_status",
        category="free",
        intent="Read the status of an operator's adoption request.",
    ),
    ToolIdentity(
        tool_id=REPAIR_OPERATOR_SCHEMA_UUID,
        capability="repair_operator_schema",
        category="restricted",
        intent="Owner repair: reassign all table ownership in an operator's tenant schema to its own role.",
    ),
    # -- Neon persistence health --
    ToolIdentity(
        tool_id=RECEIVE_NEON_402_ALERT_UUID,
        capability="receive_neon_402_alert",
        category="free",
        intent="Inbound: an operator reports its Neon store is 402-locked (operator-proof gated).",
    ),
    ToolIdentity(
        tool_id=LIST_NEON_ALERTS_UUID,
        capability="list_neon_alerts",
        category="restricted",
        intent="Owner queue: list operators that reported a Neon-402 (store locked).",
    ),
    ToolIdentity(
        tool_id=NETWORK_PERSISTENCE_HEALTH_UUID,
        capability="network_persistence_health",
        category="restricted",
        intent="Owner view: proactive per-project Neon compute-quota posture + reactive alerts.",
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


def _get_nostr_exchange() -> Any:
    from tollbooth.nostr_credentials import NostrCredentialExchange

    s = _get_settings()
    relays = _resolve_relays()
    return NostrCredentialExchange(
        nsec=s.tollbooth_nostr_operator_nsec,
        relays=relays,
        templates=ONBOARDING_TEMPLATES,
        credential_vault=None,
    )


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


async def _register_operator_via_oracle(
    operator_npub: str,
    display_name: str,
    service_url: str,
    authority_npub: str,
) -> str:
    from fastmcp import Client

    from tollbooth.registry import resolve_oracle_service

    signer = _get_nostr_signer()
    oracle_info = await resolve_oracle_service(signer.npub)

    async with Client(oracle_info["url"]) as client:
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
    from fastmcp import Client

    from tollbooth.registry import resolve_oracle_service

    signer = _get_nostr_signer()
    oracle_info = await resolve_oracle_service(signer.npub)

    args: dict = {"operator_npub": operator_npub, "authority_npub": authority_npub}
    if service_url:
        args["service_url"] = service_url
    if display_name:
        args["display_name"] = display_name

    async with Client(oracle_info["url"]) as client:
        result = await client.call_tool("update_operator", args)
        return _parse_oracle_commit_url(result)


async def _deregister_operator_via_oracle(
    operator_npub: str,
    authority_npub: str,
) -> str:
    from fastmcp import Client

    from tollbooth.registry import resolve_oracle_service

    signer = _get_nostr_signer()
    oracle_info = await resolve_oracle_service(signer.npub)

    async with Client(oracle_info["url"]) as client:
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
    from fastmcp import Client

    from tollbooth.registry import resolve_oracle_service

    signer = _get_nostr_signer()
    oracle_info = await resolve_oracle_service(signer.npub)

    async with Client(oracle_info["url"]) as client:
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


async def _require_authority_consent(
    runtime: OperatorRuntime,
    authority_proof: str,
    tool_name: str,
) -> dict[str, Any] | None:
    """Verify the caller holds the Authority's own nsec.

    The Authority's adoption / modification of an Operator entry is a
    discretionary act. Only someone with the Authority's nsec on hand can
    produce a valid Schnorr proof signed by the Authority's npub — the
    cryptographic witness that the human who controls this Authority has
    authorized this action. An app (e.g. the Pricing Studio) produces
    this proof automatically when its user clicks "adopt", provided the
    Authority's nsec is in the app's secure store.

    Returns ``None`` on success (caller proceeds) or a structured error
    dict the caller should return verbatim. The error code is
    ``AUTHORITY_CONSENT_REQUIRED`` regardless of whether the proof was
    missing, expired, or signature-invalid — the patron-actionable
    remedy is the same: present a fresh valid Authority proof.
    """
    from tollbooth.constants import ErrorCode as _EC

    authority_npub = runtime.operator_npub()
    err = await require_proof(
        authority_npub,
        authority_proof,
        tool_name,
        proven_cache=await runtime.proven_npub_cache(),
    )
    if err is None:
        return None
    return {
        "success": False,
        "error_code": _EC.AUTHORITY_CONSENT_REQUIRED,
        "error": (
            f"Authority consent required: caller must supply a valid identity "
            f"proof signed by the Authority's npub ({authority_npub[:16]}...). "
            f"Apps holding the Authority's nsec produce this proof inline; "
            f"otherwise call request_npub_proof / receive_npub_proof against "
            f"this MCP using the Authority's npub to cache one."
        ),
        "authority_npub": authority_npub,
    }


# Bootstrap DMs are kind-4 events on free public relays, which purge them
# on their own retention schedules (empirically < 69 days on damus/primal,
# 2026-06-06). A one-shot publication therefore guarantees an eventual
# cold-start outage. Re-send whenever the last publication is older than
# this — opportunistically, on Authority tool traffic (the OTS pattern).
_BOOTSTRAP_DM_REFRESH_SECONDS = 7 * 24 * 3600
# In-process throttle so high-traffic tools don't re-check the vault stamp
# on every call.
_BOOTSTRAP_DM_CHECK_INTERVAL = 3600.0
_bootstrap_dm_last_check: dict[str, float] = {}


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
        # The publish is blocking websocket I/O (up to ~10s per relay) —
        # keep it off the event loop.
        sent = await asyncio.to_thread(
            send_bootstrap_config,
            authority_nsec=signer.nsec,
            operator_npub=npub,
            config={"neon_database_url": neon_url, "schema": schema},
        )
        if sent:
            await store_operator_config(
                vault, npub, "bootstrap_dm_sent_at", str(int(time.time())),
            )
            logger.info("Bootstrap config DM (re)sent to operator %s", npub[:16])
        return sent
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bootstrap DM resend failed for %s: %s", npub[:16], exc)
        return False


async def _maybe_refresh_bootstrap_dm(npub: str) -> None:
    """Re-publish the operator's bootstrap DM if the last send has aged.

    Called opportunistically from Authority tool traffic. Cheap by
    design: an in-process throttle gates the vault read, and the
    publish itself runs only when the stored stamp is older than
    ``_BOOTSTRAP_DM_REFRESH_SECONDS``. Never raises.
    """
    now = time.monotonic()
    last_check = _bootstrap_dm_last_check.get(npub, 0.0)
    if now - last_check < _BOOTSTRAP_DM_CHECK_INTERVAL:
        return
    _bootstrap_dm_last_check[npub] = now
    try:
        vault = await _get_runtime().vault()
        from tollbooth.authority.tenant_provisioner import get_operator_config_value
        stamp = await get_operator_config_value(vault, npub, "bootstrap_dm_sent_at")
        sent_at = int(stamp) if stamp else 0
        if time.time() - sent_at >= _BOOTSTRAP_DM_REFRESH_SECONDS:
            await _resend_bootstrap_dm(npub)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Bootstrap DM refresh check skipped for %s: %s", npub[:16], exc)


async def _provision_operator(
    runtime: OperatorRuntime,
    npub: str,
    service_url: str,
    display_name: str = "",
) -> dict[str, Any]:
    """Provision an operator: ledger row + isolated Neon tenant + registry.

    The single source of truth for *what adoption does*. Both the inline-
    consent path (``register_operator``, consent supplied at call time) and
    the deferred-courtship path (``approve_adoption``, consent supplied
    later via the owner's review) call this, so the two paths produce a
    byte-identical effect. Proof/consent gating lives in the callers, never
    here.
    """
    cache = await runtime.ledger_cache()
    ledger = await cache.get(npub)
    cache.mark_dirty(npub)
    await cache.flush_user(npub)

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

    tool = make_slug_tool(mcp, slug)

    # ------------------------------------------------------------------
    # Operator lifecycle tools
    # ------------------------------------------------------------------

    @tool
    async def register_operator(
        npub: Annotated[
            str,
            Field(description="Your Nostr npub (bech32). Get one from the dpyc-oracle's how_to_join() tool."),
        ] = "",
        dpop_token: str = "",
        service_url: Annotated[
            str,
            Field(description="Your MCP endpoint URL (e.g. 'https://my-service.fastmcp.app/mcp')."),
        ] = "",
        display_name: Annotated[
            str,
            Field(description=(
                "Human-readable name for the Operator service, shown in the community "
                "roster (e.g. 'my-service'). If empty, the roster falls back to a "
                "truncated npub."
            )),
        ] = "",
        authority_proof: Annotated[
            str,
            Field(description=(
                "Identity proof signed by the Authority's OWN npub — the "
                "Authority's discretionary consent to adopt this Operator. "
                "Apps with the Authority's nsec in their keystore (e.g. the "
                "Pricing Studio) produce this proof automatically when the "
                "user clicks 'adopt'."
            )),
        ] = "",
    ) -> dict[str, Any]:
        """Provision an operator in the Authority ledger.

        Creates a ledger entry so the operator can purchase credits and
        certify purchase orders. Idempotent — safe to call again.

        Requires TWO independent identity proofs:

        1. ``proof`` — Schnorr proof signed by the candidate operator's
           ``npub``. Proves the requester really controls that npub. The
           operator typically calls ``request_npub_proof`` /
           ``receive_npub_proof`` against this Authority first to mint
           a cached dpop_token.
        2. ``authority_proof`` — Schnorr proof signed by the Authority's
           own npub. This is the Authority's human consent — only an
           agent with the Authority's nsec on hand can produce it. Apps
           generate this inline when the human admin clicks 'adopt';
           otherwise an Authority-side proof can be minted the same way
           an operator-side one is.

        Next step: Call purchase_credits to fund your credit balance.
        """
        err = await require_proof(npub, dpop_token, runtime.runtime_name("register_operator"), proven_cache=await runtime.proven_npub_cache())
        if err:
            return err
        err = await _require_authority_consent(
            runtime, authority_proof, runtime.runtime_name("register_operator"),
        )
        if err:
            return err

        # Inline-consent path: provisioning is the shared effect.
        return await _provision_operator(runtime, npub, service_url, display_name)

    @tool
    async def update_operator(
        npub: Annotated[str, Field(description="Nostr npub of the Operator to update.")] = "",
        dpop_token: str = "",
        service_url: Annotated[str, Field(description="New MCP endpoint URL (leave empty to keep current).")] = "",
        display_name: Annotated[str, Field(description="New display name (leave empty to keep current).")] = "",
        authority_proof: Annotated[
            str,
            Field(description=(
                "Identity proof signed by the Authority's OWN npub — the "
                "Authority's consent to modify this Operator's registry entry."
            )),
        ] = "",
    ) -> dict[str, Any]:
        """Update an existing Operator's community registry entry.

        Requires the same two proofs as ``register_operator``:

        - ``proof`` proves the caller controls the Operator's ``npub``.
        - ``authority_proof`` proves the Authority's human admin consents
          to the change. Without the Authority proof, anyone with the
          Operator's nsec could redirect their own ``service_url`` under
          this Authority's signature without the Authority's awareness.
        """
        err = await require_proof(npub, dpop_token, runtime.runtime_name("update_operator"), proven_cache=await runtime.proven_npub_cache())
        if err:
            return err
        err = await _require_authority_consent(
            runtime, authority_proof, runtime.runtime_name("update_operator"),
        )
        if err:
            return err
        if not service_url and not display_name:
            return {"success": False, "error": "Nothing to update. Provide service_url and/or display_name."}

        try:
            signer = _get_nostr_signer()
            commit_url = await _update_operator_via_oracle(
                operator_npub=npub,
                service_url=service_url,
                display_name=display_name,
                authority_npub=signer.npub,
            )
            await _resend_bootstrap_dm(npub)
            return {
                "success": True,
                "commit_url": commit_url,
                "message": f"Operator {npub[:16]}... updated in community registry.",
            }
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Update failed: {exc}"}

    @tool
    async def deregister_operator(
        npub: Annotated[str, Field(description="Nostr npub of the Operator to deregister.")] = "",
        dpop_token: str = "",
        authority_proof: Annotated[
            str,
            Field(description=(
                "Identity proof signed by the Authority's OWN npub — the "
                "Authority's consent to remove this Operator from the community "
                "registry under its signature."
            )),
        ] = "",
    ) -> dict[str, Any]:
        """Remove an Operator from the DPYC community registry.

        Requires the same two proofs as ``register_operator``:

        - ``proof`` proves the caller controls the Operator's ``npub``.
        - ``authority_proof`` proves the Authority's human admin consents
          to the removal. Without the Authority proof, anyone who knew an
          Operator's public npub and held its nsec could remove themselves
          from this Authority's roster without the Authority noticing.
        """
        err = await require_proof(npub, dpop_token, runtime.runtime_name("deregister_operator"), proven_cache=await runtime.proven_npub_cache())
        if err:
            return err
        err = await _require_authority_consent(
            runtime, authority_proof, runtime.runtime_name("deregister_operator"),
        )
        if err:
            return err

        try:
            signer = _get_nostr_signer()
            commit_url = await _deregister_operator_via_oracle(
                operator_npub=npub,
                authority_npub=signer.npub,
            )
            return {
                "success": True,
                "commit_url": commit_url,
                "message": f"Operator {npub[:16]}... removed from community registry.",
            }
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Deregistration failed: {exc}"}

    @tool
    async def get_operator_config(
        npub: Annotated[str, Field(description="Your Nostr npub (bech32).")] = "",
        dpop_token: str = "",
    ) -> dict[str, Any]:
        """Retrieve operator bootstrap configuration (Neon URL, schema).

        Gated by Schnorr signature proving ownership of the requested npub.
        """
        err = await require_proof(npub, dpop_token, runtime.runtime_name("get_operator_config"), proven_cache=await runtime.proven_npub_cache())
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
        dpop_token: str = "",
    ) -> dict[str, Any]:
        """View registration status, balance summary, and the Authority's Nostr npub.

        When an explicit ``npub`` is provided, requires a Schnorr proof of
        ownership — without it, anyone could enumerate balances by walking
        the community registry. When ``npub`` is empty, falls back to the
        Authority's own operator identity and skips the proof check (self-
        inspection is always allowed).
        """
        if npub:
            err = await require_proof(npub, dpop_token, runtime.runtime_name("operator_status"), proven_cache=await runtime.proven_npub_cache())
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

        # Opportunistic bootstrap DM refresh for the inspected operator.
        if npub:
            asyncio.create_task(_maybe_refresh_bootstrap_dm(user_id))

        return result

    # ------------------------------------------------------------------
    # certify_credits — the revenue tool (ad valorem via paid_tool)
    # ------------------------------------------------------------------

    @tool
    @runtime.paid_tool(capability_uuid("certify_credits"))
    async def certify_credits(
        npub: Annotated[
            str,
            Field(description="The operator's DPYC npub (from register_operator response)."),
        ] = "",
        dpop_token: str = "",
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

        # Flush immediately (credit-critical)
        cache = await runtime.ledger_cache()
        if not await cache.flush_user(npub):
            logger.error("Failed to persist fee debit for %s", npub)

        # Opportunistic bootstrap DM refresh (the OTS pattern): relays
        # purge kind-4 events, so keep this operator's bootstrap config
        # alive on the back of its own certification traffic. Fire and
        # forget — certification latency must not pay for relay I/O.
        asyncio.create_task(_maybe_refresh_bootstrap_dm(npub))

        return {
            "success": True,
            "certificate": nostr_event_json,
            "jti": jti,
            "amount_sats": amount_sats,
            "fee_sats": fee_sats,
            "net_sats": net_sats,
            "expires_at": expiration,
        }

    # ------------------------------------------------------------------
    # DPYC membership diagnostic
    # ------------------------------------------------------------------

    @tool
    async def check_dpyc_membership(npub: str) -> dict[str, Any]:
        """Look up an npub in the DPYC community registry."""
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

    async def _notify_owner_of_request(operator_npub: str, service_url: str) -> None:
        """Best-effort heads-up DM to the Authority owner. Never raises.

        The durable queue (``list_adoption_requests``) is the source of
        truth; this DM is only a "check your queue" ping. Fire-and-forget so
        relay I/O never blocks the inbound request.
        """
        try:
            exchange = _get_nostr_exchange()
            await exchange.open_channel(
                "operator_adoption",
                greeting=(
                    f"Operator {operator_npub[:16]}... ({service_url}) requests "
                    "adoption. Review pending requests with list_adoption_requests "
                    "and approve/reject in the Pricing Studio."
                ),
                recipient_npub=runtime.operator_npub(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Owner-notification DM skipped for %s: %s", operator_npub[:16], exc)

    @tool
    async def receive_adoption_request(
        operator_npub: Annotated[
            str,
            Field(description="The operator's Nostr npub requesting adoption."),
        ] = "",
        dpop_token: Annotated[
            str,
            Field(description=(
                "Inline kind-27235 proof signed by the operator's nsec, bound "
                "to the canonical adoption sentinel. request_adoption mints "
                "this automatically."
            )),
        ] = "",
        service_url: Annotated[
            str,
            Field(description="The operator's MCP endpoint URL."),
        ] = "",
    ) -> dict[str, Any]:
        """Inbound: record an operator's request to be adopted by this Authority.

        Called MCP-to-MCP by the operator's ``request_adoption``. Verifies the
        operator controls ``operator_npub`` (inline Schnorr bound to the
        adoption sentinel — no relay round-trip), records a durable ``pending``
        row, and fires a best-effort owner-notification DM. Does NOT provision —
        provisioning waits for the owner's ``approve_adoption``.
        """
        if not operator_npub:
            return {"success": False, "error": "operator_npub is required."}
        # Verify the operator owns the npub (inline proof bound to the sentinel).
        if not verify_proof(dpop_token, operator_npub, ADOPTION_PROOF_TOOL):
            return {
                "success": False,
                "error_code": ErrorCode.PROOF_INVALID,
                "error": (
                    "Invalid operator adoption proof. The request must carry a "
                    "fresh kind-27235 proof signed by operator_npub. Call "
                    "request_adoption on the operator (it mints this inline)."
                ),
            }
        try:
            vault = await runtime.vault()
            await adoption_store.ensure_schema(vault)
            await adoption_store.upsert_pending(vault, operator_npub, service_url)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not record request: {exc}"}

        await _notify_owner_of_request(operator_npub, service_url)
        return {
            "success": True,
            "status": adoption_store.PENDING,
            "operator_npub": operator_npub,
            "message": (
                "Adoption request received and pending the Authority owner's "
                "decision. Poll get_adoption_status for progress."
            ),
        }

    @tool
    async def list_adoption_requests(
        authority_proof: Annotated[
            str,
            Field(description="Proof signed by the Authority's OWN npub (owner consent)."),
        ] = "",
    ) -> dict[str, Any]:
        """Owner queue: list pending operator-adoption requests.

        Restricted to the Authority owner (consent proof). This is the
        review-on-your-own-time surface the Pricing Studio renders.
        """
        err = await _require_authority_consent(
            runtime, authority_proof, runtime.runtime_name("list_adoption_requests"),
        )
        if err:
            return err
        try:
            vault = await runtime.vault()
            await adoption_store.ensure_schema(vault)
            await adoption_store.prune_expired(vault)
            pending = await adoption_store.list_pending(vault)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not list requests: {exc}"}
        return {"success": True, "count": len(pending), "requests": pending}

    # ------------------------------------------------------------------
    # Neon persistence health — the accounting store are the Authority's charge
    # ------------------------------------------------------------------

    async def _notify_owner_of_neon_402(operator_npub: str, detail: str) -> None:
        """Best-effort heads-up DM to the Authority owner that an operator's
        store is 402-locked. Never raises; the durable row is the record."""
        try:
            exchange = _get_nostr_exchange()
            await exchange.open_channel(
                "neon_persistence_alert",
                greeting=(
                    f"⚠ Operator {operator_npub[:16]}... reports its Neon store is "
                    "402-locked (compute/storage quota exhausted) — paid tools are "
                    "down for its patrons. Review with network_persistence_health / "
                    "list_neon_alerts and restore capacity (upgrade the plan or wait "
                    "for the quota reset)."
                ),
                recipient_npub=runtime.operator_npub(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Neon-402 owner DM skipped for %s: %s", operator_npub[:16], exc)

    @tool
    async def receive_neon_402_alert(
        npub: Annotated[
            str,
            Field(description="The reporting operator's Nostr npub (the one whose store is locked)."),
        ] = "",
        dpop_token: Annotated[
            str,
            Field(description=(
                "Inline kind-27235 proof signed by the operator's nsec, bound to "
                "this tool's wire name. The operator's runtime mints and sends this "
                "automatically when it catches a Neon 402."
            )),
        ] = "",
        detail: Annotated[
            str,
            Field(description="Short, credential-free error summary (the Neon 402 message)."),
        ] = "",
    ) -> dict[str, Any]:
        """Inbound: an operator reports its Neon store is 402-locked.

        Called MCP-to-MCP by the operator's runtime the instant it catches a
        Neon HTTP 402 on its own database. Verifies the operator controls
        ``npub`` (inline Schnorr bound to this tool's wire name), records a
        durable latest-state row, and fires a best-effort owner-notification
        DM. This is how the Authority learns the store is dark BEFORE a patron
        files a complaint.
        """
        if not npub:
            return {"success": False, "error": "npub is required."}
        if not verify_proof(dpop_token, npub, runtime.runtime_name("receive_neon_402_alert")):
            return {
                "success": False,
                "error_code": ErrorCode.PROOF_INVALID,
                "error": (
                    "Invalid operator proof. The alert must carry a fresh "
                    "kind-27235 proof signed by npub, bound to this tool."
                ),
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
        authority_proof: Annotated[
            str,
            Field(description="Proof signed by the Authority's OWN npub (owner consent)."),
        ] = "",
    ) -> dict[str, Any]:
        """Owner queue: operators that reported a Neon-402 (store locked).

        Restricted to the Authority owner. A companion to network_persistence_health:
        this is the reactive list (operators that already went dark); the health
        tool adds the proactive per-project compute posture.
        """
        err = await _require_authority_consent(
            runtime, authority_proof, runtime.runtime_name("list_neon_alerts"),
        )
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
        authority_proof: Annotated[
            str,
            Field(description="Proof signed by the Authority's OWN npub (owner consent)."),
        ] = "",
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
        err = await _require_authority_consent(
            runtime, authority_proof, runtime.runtime_name("network_persistence_health"),
        )
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

    @tool
    async def approve_adoption(
        operator_npub: Annotated[
            str,
            Field(description="The operator npub to approve and provision."),
        ] = "",
        authority_proof: Annotated[
            str,
            Field(description="Proof signed by the Authority's OWN npub (owner consent)."),
        ] = "",
    ) -> dict[str, Any]:
        """Approve a pending request and provision the operator.

        The deferred-courtship counterpart to register_operator: same
        ``authority_proof`` consent, same provisioning effect
        (``_provision_operator``) — just supplied later, after review.
        """
        err = await _require_authority_consent(
            runtime, authority_proof, runtime.runtime_name("approve_adoption"),
        )
        if err:
            return err
        try:
            vault = await runtime.vault()
            await adoption_store.ensure_schema(vault)
            row = await adoption_store.get(vault, operator_npub)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not read request: {exc}"}
        if row is None:
            return {
                "success": False,
                "error_code": ErrorCode.ADOPTION_NOT_FOUND,
                "error": f"No adoption request found for {operator_npub[:16]}...",
            }
        if row.get("status") == adoption_store.PROVISIONED:
            return {
                "success": False,
                "error_code": ErrorCode.ADOPTION_ALREADY_PROVISIONED,
                "error": f"Operator {operator_npub[:16]}... is already provisioned.",
            }

        # Deferred path carries the name if the request captured one; today the adoption
        # store has no display_name column, so this is "" and falls back to a short npub.
        # The inline register_operator path (the app's "adopt") is where the name flows.
        result = await _provision_operator(
            runtime, operator_npub, row.get("service_url", ""), row.get("display_name", ""),
        )
        await adoption_store.mark(vault, operator_npub, adoption_store.PROVISIONED)
        result["adoption"] = "approved"
        return result

    @tool
    async def reject_adoption(
        operator_npub: Annotated[
            str,
            Field(description="The operator npub to reject."),
        ] = "",
        authority_proof: Annotated[
            str,
            Field(description="Proof signed by the Authority's OWN npub (owner consent)."),
        ] = "",
        reason: Annotated[str, Field(description="Optional human-readable reason.")] = "",
    ) -> dict[str, Any]:
        """Reject a pending operator-adoption request (owner consent)."""
        err = await _require_authority_consent(
            runtime, authority_proof, runtime.runtime_name("reject_adoption"),
        )
        if err:
            return err
        try:
            vault = await runtime.vault()
            await adoption_store.ensure_schema(vault)
            hit = await adoption_store.mark(vault, operator_npub, adoption_store.REJECTED)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Could not reject request: {exc}"}
        if not hit:
            return {
                "success": False,
                "error_code": ErrorCode.ADOPTION_NOT_FOUND,
                "error": f"No adoption request found for {operator_npub[:16]}...",
            }
        return {
            "success": True,
            "status": adoption_store.REJECTED,
            "operator_npub": operator_npub,
            "reason": reason,
        }

    @tool
    async def repair_operator_schema(
        operator_npub: Annotated[
            str,
            Field(description="The operator npub whose tenant-schema ownership to repair."),
        ] = "",
        authority_proof: Annotated[
            str,
            Field(description="Proof signed by the Authority's OWN npub (owner consent)."),
        ] = "",
    ) -> dict[str, Any]:
        """Owner repair: reassign every table in an operator's tenant schema to
        the operator's own role, then re-grant DML.

        For tenants whose tables were created/owned by the provisioning role —
        the operator role then cannot CREATE INDEX on them ("must be owner"),
        which aborts the whole vault bootstrap. Unlike register_operator this
        does NOT rotate the operator's DB password or re-send the bootstrap DM;
        it only fixes ownership + grants in place. Idempotent.
        """
        err = await _require_authority_consent(
            runtime, authority_proof, runtime.runtime_name("repair_operator_schema"),
        )
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

        Status (pending/approved/rejected/provisioned) isn't sensitive — it's
        the operator's own request — so the operator can poll it openly via
        its ``adoption_status`` tool.
        """
        try:
            vault = await runtime.vault()
            await adoption_store.ensure_schema(vault)
            row = await adoption_store.get(vault, operator_npub)
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
            exchange = _get_nostr_exchange()
            result = await exchange.open_channel(
                "authority_claim",
                greeting=(
                    "You are requesting to become the curator of this Authority. "
                    "Reply with: claim = @@@yes@@@ and include the dpop_token slug."
                ),
                recipient_npub=candidate_npub,
            )
        except Exception as exc:  # noqa: BLE001
            _onboarding.complete()
            return {"success": False, "error": f"Failed to send DM challenge: {exc}"}

        return {
            "success": True,
            "candidate_npub": candidate_npub,
            "phase": challenge.phase,
            "instructions": (
                f"A Nostr DM challenge has been sent to {candidate_npub[:16]}... "
                "Reply with: claim = @@@yes@@@ and the dpop_token slug. "
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
            exchange = _get_nostr_exchange()
            await exchange.receive(sender_npub=candidate_npub, service="authority_claim")
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
            exchange2 = _get_nostr_exchange()
            await exchange2.open_channel(
                "authority_approval",
                greeting=(
                    f"{candidate_npub} requests to curate the Authority at "
                    f"npub {signer.npub[:16]}... "
                    "Reply with: approval = @@@yes@@@ and the dpop_token slug."
                ),
                recipient_npub=parent_npub,
            )
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
            exchange = _get_nostr_exchange()
            await exchange.receive(sender_npub=parent_npub, service="authority_approval")
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
