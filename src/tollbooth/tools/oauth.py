"""OAuth2 browser-dance tool bodies (audit M2.1h Phase 2).

Extracted verbatim from the begin_oauth / check_oauth_status closures (which
register only when an OAuthProviderConfig is set). check_oauth_status performs
the code→token exchange and persists OAuth tokens to the vault — §2-sensitive,
so the move is behavior-preserving: the ``_opc`` closure local became
``opc = rt._oauth_provider``, ``_OAUTH_SERVICE`` became ``oauth_service``, and
``resolve_npub`` became ``rt.resolve_npub``. Pinned by
tests/test_oauth_tools_characterization.py; the underlying oauth2_collector
functions are tested in test_oauth2_collector.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tollbooth.oauth_situation import OAuthSituation

logger = logging.getLogger(__name__)

_CREDS_NOT_DELIVERED = (
    "Operator API credentials ({field}) have not been delivered yet. "
    "This is not an error — the operator needs to deliver credentials "
    "via Secure Courier (request_credential_channel / receive_credentials) "
    "before OAuth can start."
)


def _exchange_denied_situation(exc: Any) -> OAuthSituation:
    """Map a classified exchange refusal to the situation that names its culprit.

    Mirrors ``OperatorRuntime._denied_situation`` for the refresh path, but the
    grant case is not ``token_expired`` — no session ever existed. Only the
    code was refused, so the recovery is "start the browser dance again" with
    the provider's words attached, not "your existing session aged out".
    """
    detail, status, oauth_error = exc.detail, exc.status_code, exc.oauth_error
    if exc.fault == "client":
        return OAuthSituation(
            "operator_app_credentials_rejected",
            detail=detail, status_code=status, oauth_error=oauth_error,
        )
    if exc.fault == "request":
        return OAuthSituation(
            "exchange_request_malformed",
            detail=detail, status_code=status, oauth_error=oauth_error,
        )
    return OAuthSituation(
        "exchange_grant_rejected",
        detail=detail, status_code=status, oauth_error=oauth_error,
    )


async def begin_oauth_tool(rt: Any, npub: str, dpop_token: str) -> dict[str, Any]:
    """Start the OAuth2 authorization flow; return an authorize URL."""
    opc = rt._oauth_provider
    oauth_service = opc.service_name

    if err := await rt.require_caller_proof(npub, dpop_token, "begin_oauth"):
        return err
    resolved = rt.resolve_npub(npub)

    # Load operator credentials using vendor field names
    _id_field = opc.client_id_field
    _secret_field = opc.client_secret_field
    try:
        creds = await rt.load_credentials(
            [_id_field, _secret_field],
        )
    except Exception:  # noqa: BLE001
        return {
            "success": False,
            "error": _CREDS_NOT_DELIVERED.format(field=_id_field),
        }

    client_id = creds.get(_id_field, "")
    if not client_id:
        return {
            "success": False,
            "error": _CREDS_NOT_DELIVERED.format(field=_id_field),
        }

    # Resolve collector redirect URI via the Oracle (operators never read GitHub).
    try:
        from tollbooth.oracle_client import default_oracle_client
        svc = await default_oracle_client().resolve_service(
            name="tollbooth-oauth2-callback"
        )
        if not svc or not svc.get("url"):
            return {"success": False, "error": "OAuth2 collector not found in registry"}
        redirect_uri = svc["url"].rstrip("/")
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"OAuth2 collector not found: {e}"}

    from tollbooth.oauth2_collector import (
        build_authorize_url,
        generate_pkce_pair,
        pack_oauth_state,
    )

    extra_params: dict[str, str] = {}
    verifier = ""
    if opc.pkce:
        verifier, challenge = generate_pkce_pair()
        extra_params["code_challenge"] = challenge
        extra_params["code_challenge_method"] = "S256"

    # State carries BOTH npubs: the patron half is the collector's lookup key
    # (and the retrieve key, unchanged); the operator half is the PUBLIC key the
    # collector seals the code to, so only this operator's nsec can open it
    # (check_oauth_status passes rt._get_nsec()). Both are public — safe in a URL.
    authorize_url = build_authorize_url(
        opc.authorize_url,
        client_id,
        redirect_uri,
        pack_oauth_state(resolved, rt.operator_npub()),
        scope=opc.scopes or None,
        extra_params=extra_params or None,
    )

    # Store PKCE verifier and redirect_uri for check_oauth_status
    vault_data: dict[str, str] = {"redirect_uri": redirect_uri}
    if verifier:
        vault_data["pkce_verifier"] = verifier
    await rt.store_patron_session(
        resolved, vault_data, service=f"_oauth_pending_{oauth_service}",
    )

    # Try to shorten the URL
    short_url = None
    try:
        from tollbooth.shortlinks import create_shortlink
        short_url = await create_shortlink(authorize_url)
    except Exception:
        logger.debug("OAuth URL shortening failed; using full URL", exc_info=True)

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


async def check_oauth_status_tool(rt: Any, npub: str, dpop_token: str) -> dict[str, Any]:
    """Poll the collector for the auth code, exchange it, persist tokens."""
    opc = rt._oauth_provider
    oauth_service = opc.service_name

    if err := await rt.require_caller_proof(npub, dpop_token, "check_oauth_status"):
        return err
    resolved = rt.resolve_npub(npub)

    # Load pending state (PKCE verifier, redirect_uri)
    pending, pending_situation = await rt.load_patron_session(
        resolved, service=f"_oauth_pending_{oauth_service}",
    )
    if pending_situation:
        # Mid-dance: the patron has already clicked Allow. Telling them to start
        # over because the vault was cold would discard a completed grant.
        return rt.oauth_situation_response(OAuthSituation(pending_situation))
    if not pending or "redirect_uri" not in pending:
        return {
            "success": False,
            "error": "No pending OAuth flow. Call begin_oauth first.",
        }
    redirect_uri = pending["redirect_uri"]
    verifier = pending.get("pkce_verifier")

    # Load operator credentials (vendor field names → OAuth protocol names)
    _cid_field = opc.client_id_field
    _csec_field = opc.client_secret_field
    try:
        creds, op_situation = await rt._load_vault_creds(rt.operator_credential_service)
    except Exception:
        logger.exception("Failed to load operator credentials")
        return {"success": False, "error": "Operator credentials could not be loaded. Check operator logs."}

    if op_situation:
        # The patron has already clicked Allow; the authorization code is real
        # and waiting. Exchanging it with blank client credentials would burn a
        # single-use code and fail at the provider with a message that names
        # nothing. Say what actually happened instead.
        return rt.oauth_situation_response(OAuthSituation(op_situation))

    client_id = creds.get(_cid_field, "")
    client_secret = creds.get(_csec_field, "")

    # Poll collector for the authorization code — URL via the Oracle, not GitHub.
    try:
        from tollbooth.oracle_client import default_oracle_client
        svc = await default_oracle_client().resolve_service(
            name="tollbooth-oauth2-collector"
        )
        if not svc or not svc.get("url"):
            return {"success": False, "error": "OAuth2 collector not found in registry"}
        collector_url = svc["url"]
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"OAuth2 collector: {e}"}

    from tollbooth.oauth2_collector import (
        OAuthRefreshDenied,
        OAuthRefreshUnavailable,
        exchange_code_for_token,
        retrieve_code_from_collector,
    )

    # NIP-44 seal opens only with the operator nsec — state is lookup only (#228).
    code = await retrieve_code_from_collector(
        collector_url, resolved, rt._get_nsec(),
    )
    if code is None:
        return {
            "success": True,
            "status": "pending",
            "message": (
                "Waiting for browser authorization. "
                "Open the URL from begin_oauth."
            ),
        }

    # Exchange code for tokens. The exchange path classifies the same way the
    # refresh path does (#170): a 400 carries fault/oauth_error/the provider's
    # words. A bare Exception used to swallow all of that into "Token exchange
    # failed. Check operator logs." — leaving both patron and operator blind
    # while X's reason sat only in a raise_for_status stack trace.
    try:
        token = await exchange_code_for_token(
            code, client_id, client_secret, redirect_uri,
            opc.token_url,
            code_verifier=verifier,
        )
    except OAuthRefreshDenied as exc:
        logger.warning(
            "Token exchange refused for %s (fault=%s, oauth_error=%s): %s",
            resolved[:16], exc.fault, exc.oauth_error, exc.detail,
        )
        return rt.oauth_situation_response(_exchange_denied_situation(exc))
    except OAuthRefreshUnavailable as exc:
        logger.warning(
            "Token exchange unavailable for %s (status=%s): %s",
            resolved[:16], exc.status_code, exc.detail,
        )
        return rt.oauth_situation_response(OAuthSituation(
            "exchange_unavailable",
            detail=exc.detail,
            status_code=exc.status_code,
        ))
    except Exception as exc:
        logger.exception("Token exchange failed for %s", resolved[:16])
        return rt.oauth_situation_response(OAuthSituation(
            "exchange_failed_unclassified",
            detail=f"{type(exc).__name__}: {exc}",
        ))

    # Build vault data from token.
    # Store both the raw token_json (for operators that expect the
    # full blob) and individual fields (for direct access).
    #
    # Built from scratch, not merged over the prior blob — and the vault write
    # REPLACES rather than merges, so a fresh authorization silently drops
    # `refresh_lost_at` (see OperatorRuntime._mark_refresh_lost). That is
    # exactly right: this grant is new, so no earlier lost renewal can be
    # blamed for its death. If this is ever changed to merge, clear that field
    # explicitly, or a stale marker will misattribute a future expiry.
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
    if opc.on_token_received is not None:
        try:
            extra = await opc.on_token_received(resolved, token)
            if extra:
                vault_data.update({k: str(v) for k, v in extra.items()})
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": f"Post-token callback: {e}"}

    # Persist tokens to vault
    stored = await rt.store_patron_session(
        resolved, vault_data, service=oauth_service,
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
