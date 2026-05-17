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
        purchase_mode="direct",
        service_name="My Authority",
        ots_enabled=True,
        operator_credential_template=OPERATOR_CREDENTIAL_TEMPLATE,
    )

    register_standard_tools(mcp, "authority", runtime, ...)
    register_authority_tools(mcp, runtime)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Annotated, Any

from pydantic import Field

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.identity_proof import require_proof
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
    STANDARD_IDENTITIES,
    ToolIdentity,
    capability_uuid,
)

from tollbooth.authority.nostr_signing import AuthorityNostrSigner
from tollbooth.authority.onboarding import ONBOARDING_TEMPLATES, OnboardingState
from tollbooth.authority.replay import ReplayTracker
from tollbooth.authority.settings import AuthoritySettings

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
    },
)


# ======================================================================
# Authority domain tool registry
# ======================================================================

AUTHORITY_DOMAIN_TOOLS: list[ToolIdentity] = [
    ToolIdentity(
        capability="certify_credits",
        category="write",
        intent="Certify a purchase order with Schnorr-signed certificate.",
        pricing_hint_type="percent",
        pricing_hint_value=2,
        pricing_hint_param="amount_sats",
        pricing_hint_min=10,
    ),
    ToolIdentity(
        capability="register_operator",
        category="free",
        intent="Provision an operator in the Authority ledger.",
    ),
    ToolIdentity(
        capability="update_operator",
        category="free",
        intent="Update an operator's community registry entry.",
    ),
    ToolIdentity(
        capability="deregister_operator",
        category="free",
        intent="Remove an operator from the DPYC community registry.",
    ),
    ToolIdentity(
        capability="get_operator_config",
        category="restricted",
        intent="Retrieve operator bootstrap configuration.",
    ),
    ToolIdentity(
        capability="operator_status",
        category="free",
        intent="View registration status, balance, and Authority npub.",
    ),
    ToolIdentity(
        capability="check_dpyc_membership",
        category="free",
        intent="Look up an npub in the DPYC community registry.",
    ),
    ToolIdentity(
        capability="register_authority_npub",
        category="free",
        intent="Step 1/3 of Authority onboarding — send DM challenge.",
    ),
    ToolIdentity(
        capability="confirm_authority_claim",
        category="free",
        intent="Step 2/3 of Authority onboarding — verify candidate DM.",
    ),
    ToolIdentity(
        capability="check_authority_approval",
        category="free",
        intent="Step 3/3 of Authority onboarding — check parent approval.",
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
    relays = _resolve_relays(s.tollbooth_nostr_relays or None)
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
        pass
    return None


async def _set_authority_npub(npub: str) -> None:
    global _cached_authority_npub
    try:
        vault = await _get_runtime().vault()
        await vault.set_config("authority_npub", npub)
    except Exception:
        pass
    _cached_authority_npub = npub


# ======================================================================
# Oracle MCP-to-MCP helpers
# ======================================================================


async def _register_operator_via_oracle(
    operator_npub: str,
    display_name: str,
    service_url: str,
    authority_npub: str,
) -> str:
    from tollbooth.registry import resolve_oracle_service
    from fastmcp import Client

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
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "text"):
                    import json
                    try:
                        return json.loads(block.text).get("commit_url", block.text)
                    except (json.JSONDecodeError, TypeError):
                        return block.text
        return str(result)


async def _update_operator_via_oracle(
    operator_npub: str,
    service_url: str,
    display_name: str,
    authority_npub: str,
) -> str:
    from tollbooth.registry import resolve_oracle_service
    from fastmcp import Client

    signer = _get_nostr_signer()
    oracle_info = await resolve_oracle_service(signer.npub)

    args: dict = {"operator_npub": operator_npub, "authority_npub": authority_npub}
    if service_url:
        args["service_url"] = service_url
    if display_name:
        args["display_name"] = display_name

    async with Client(oracle_info["url"]) as client:
        result = await client.call_tool("update_operator", args)
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "text"):
                    import json
                    try:
                        return json.loads(block.text).get("commit_url", block.text)
                    except (json.JSONDecodeError, TypeError):
                        return block.text
        return str(result)


async def _deregister_operator_via_oracle(
    operator_npub: str,
    authority_npub: str,
) -> str:
    from tollbooth.registry import resolve_oracle_service
    from fastmcp import Client

    signer = _get_nostr_signer()
    oracle_info = await resolve_oracle_service(signer.npub)

    async with Client(oracle_info["url"]) as client:
        result = await client.call_tool(
            "deregister_operator",
            {"operator_npub": operator_npub, "authority_npub": authority_npub},
        )
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "text"):
                    import json
                    try:
                        return json.loads(block.text).get("commit_url", block.text)
                    except (json.JSONDecodeError, TypeError):
                        return block.text
        return str(result)


async def _register_via_oracle(
    authority_npub: str,
    display_name: str,
    service_url: str,
    upstream_authority_npub: str,
) -> str:
    from tollbooth.registry import resolve_oracle_service
    from fastmcp import Client

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
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "text"):
                    import json
                    try:
                        return json.loads(block.text).get("commit_url", "")
                    except (json.JSONDecodeError, TypeError):
                        return block.text
        return str(result)


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


async def _resend_bootstrap_dm(npub: str) -> bool:
    try:
        vault = await _get_runtime().vault()
        from tollbooth.authority.tenant_provisioner import get_all_operator_config
        config = await get_all_operator_config(vault, npub)
        neon_url = config.get("neon_database_url")
        schema = config.get("schema", "")
        if not neon_url:
            return False
        from tollbooth.bootstrap_relay import send_bootstrap_config
        signer = _get_nostr_signer()
        sent = send_bootstrap_config(
            authority_nsec=signer.nsec,
            operator_npub=npub,
            config={"neon_database_url": neon_url, "schema": schema},
        )
        if sent:
            logger.info("Bootstrap config DM (re)sent to operator %s", npub[:16])
        return sent
    except Exception as exc:
        logger.warning("Bootstrap DM resend failed for %s: %s", npub[:16], exc)
        return False


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
            with ``purchase_mode="direct"``, ``ots_enabled=True``,
            and the Authority's tool_registry (``STANDARD_IDENTITIES``
            merged with ``AUTHORITY_TOOL_REGISTRY``).
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
        proof: str = "",
        service_url: Annotated[
            str,
            Field(description="Your MCP endpoint URL (e.g. 'https://my-service.fastmcp.app/mcp')."),
        ] = "",
    ) -> dict[str, Any]:
        """Provision an operator in the Authority ledger.

        Creates a ledger entry so the operator can purchase credits and
        certify purchase orders. Idempotent — safe to call again. Requires
        a Schnorr proof of npub ownership; the candidate operator should
        call ``request_npub_proof`` + ``receive_npub_proof`` on this
        Authority first, then pass the resulting token here.

        Next step: Call purchase_credits to fund your credit balance.
        """
        err = require_proof(npub, proof, "register_operator")
        if err:
            return err

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
                provision_operator_schema,
                store_operator_config,
                neon_url_for_operator,
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
        except Exception as exc:
            logger.warning("Neon tenant provisioning failed (non-fatal): %s", exc)

        # Register in community registry via Oracle
        commit_url = ""
        try:
            signer = _get_nostr_signer()
            commit_url = await _register_operator_via_oracle(
                operator_npub=npub,
                display_name=npub[:16] + "...",
                service_url=service_url,
                authority_npub=signer.npub,
            )
        except Exception as exc:
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

    @tool
    async def update_operator(
        npub: Annotated[str, Field(description="Nostr npub of the Operator to update.")] = "",
        proof: str = "",
        service_url: Annotated[str, Field(description="New MCP endpoint URL (leave empty to keep current).")] = "",
        display_name: Annotated[str, Field(description="New display name (leave empty to keep current).")] = "",
    ) -> dict[str, Any]:
        """Update an existing Operator's community registry entry.

        Requires a Schnorr proof of npub ownership — without it, an attacker
        who knew a victim Operator's public npub could rewrite their
        ``service_url`` to point at an attacker-controlled MCP endpoint.
        """
        err = require_proof(npub, proof, "update_operator")
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
        except Exception as exc:
            return {"success": False, "error": f"Update failed: {exc}"}

    @tool
    async def deregister_operator(
        npub: Annotated[str, Field(description="Nostr npub of the Operator to deregister.")] = "",
        proof: str = "",
    ) -> dict[str, Any]:
        """Remove an Operator from the DPYC community registry.

        Requires a Schnorr proof of npub ownership — without it, anyone who
        knew a victim Operator's public npub could remove them from the
        community registry under this Authority's signature.
        """
        err = require_proof(npub, proof, "deregister_operator")
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
        except Exception as exc:
            return {"success": False, "error": f"Deregistration failed: {exc}"}

    @tool
    async def get_operator_config(
        npub: Annotated[str, Field(description="Your Nostr npub (bech32).")] = "",
        proof: str = "",
    ) -> dict[str, Any]:
        """Retrieve operator bootstrap configuration (Neon URL, schema).

        Gated by Schnorr signature proving ownership of the requested npub.
        """
        err = require_proof(npub, proof, "get_operator_config")
        if err:
            return err

        try:
            vault = await runtime.vault()
            from tollbooth.authority.tenant_provisioner import get_all_operator_config
            config = await get_all_operator_config(vault, npub)
        except Exception as exc:
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
        proof: str = "",
    ) -> dict[str, Any]:
        """View registration status, balance summary, and the Authority's Nostr npub.

        When an explicit ``npub`` is provided, requires a Schnorr proof of
        ownership — without it, anyone could enumerate balances by walking
        the community registry. When ``npub`` is empty, falls back to the
        Authority's own operator identity and skips the proof check (self-
        inspection is always allowed).
        """
        if npub:
            err = require_proof(npub, proof, "operator_status")
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
            Field(description="The operator's DPYC npub (from register_operator response)."),
        ] = "",
        proof: str = "",
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
            "dpyc_protocol": "tollbooth-cert-v1",
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
                    "Reply with: claim = @@@yes@@@ and include the poison slug."
                ),
                recipient_npub=candidate_npub,
            )
        except Exception as exc:
            _onboarding.complete()
            return {"success": False, "error": f"Failed to send DM challenge: {exc}"}

        return {
            "success": True,
            "candidate_npub": candidate_npub,
            "phase": challenge.phase,
            "instructions": (
                f"A Nostr DM challenge has been sent to {candidate_npub[:16]}... "
                "Reply with: claim = @@@yes@@@ and the poison slug. "
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
        except Exception as exc:
            return {"success": False, "error": f"No valid claim DM received: {exc}"}

        try:
            signer = _get_nostr_signer()
            parent_npub = await resolve_my_parent_npub(signer.npub)
        except Exception as exc:
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
                    "Reply with: approval = @@@yes@@@ and the poison slug."
                ),
                recipient_npub=parent_npub,
            )
        except Exception as exc:
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
        except Exception as exc:
            return {"success": False, "error": f"No approval received from parent Authority: {exc}"}

        await _set_authority_npub(candidate_npub)

        commit_url = ""
        try:
            service_url = await _resolve_own_service_url()
            commit_url = await _register_via_oracle(
                authority_npub=candidate_npub,
                display_name=f"Authority ({candidate_npub[:16]}...)",
                service_url=service_url,
                upstream_authority_npub=parent_npub,
            )
        except Exception as exc:
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
