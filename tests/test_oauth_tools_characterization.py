"""Characterization net for the OAuth tools (audit M2.1h Phase 1).

begin_oauth and check_oauth_status orchestrate the browser OAuth2 dance —
PKCE, collector resolution, authorize-URL build, and (in check_oauth_status)
the code→token exchange and token vault storage. §2-sensitive. The underlying
oauth2_collector functions are tested in test_oauth2_collector.py; this pins
the tool-closure orchestration before the Phase 2 move to tools/oauth.py.

Heavy patching: the tools import resolve_service_by_name / generate_pkce_pair /
build_authorize_url / create_shortlink / retrieve_code_from_collector /
exchange_code_for_token at call time, so we patch them at their source module.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.oauth_config import OAuthProviderConfig
from tollbooth.runtime import OperatorRuntime, register_standard_tools

os.environ.setdefault(
    "TOLLBOOTH_NOSTR_OPERATOR_NSEC",
    "nsec1test000000000000000000000000000000000000000000000000000000",
)

# A real npub so resolve_npub accepts it.
PATRON = "npub1y20qa7d3ddmh6730hdr0u0r08zys4p7pyk30uhur9edx4d88q4zqnr3q2h"
# The operator npub the collector seals TO — begin_oauth packs it into the state.
OPERATOR = "npub1ymgfh46ace33zgld5zdc7gyhc5keyu42v36td0q7c44ks45d79eslwe2q2"

OPC = OAuthProviderConfig(
    service_name="testoauth",
    authorize_url="https://provider.example/authorize",
    token_url="https://provider.example/token",
    client_id_field="app_key",
    client_secret_field="secret",
    pkce=True,
    scopes="readonly",
)


def _runtime(*, opc=OPC, on_token=None):
    if on_token is not None:
        opc = OAuthProviderConfig(
            service_name=opc.service_name, authorize_url=opc.authorize_url,
            token_url=opc.token_url, client_id_field=opc.client_id_field,
            client_secret_field=opc.client_secret_field, pkce=opc.pkce,
            scopes=opc.scopes, on_token_received=on_token,
        )
    rt = OperatorRuntime(tool_registry={}, service_name="Test", oauth_provider=opc)
    rt.require_caller_proof = AsyncMock(return_value=None)
    rt.load_credentials = AsyncMock(return_value={"app_key": "CID", "secret": "CSEC"})
    rt._load_vault_creds = AsyncMock(
        return_value=({"app_key": "CID", "secret": "CSEC"}, ""),
    )
    rt.store_patron_session = AsyncMock(return_value=True)
    rt.load_patron_session = AsyncMock(return_value=(None, ""))
    # begin_oauth seals to the operator: it packs rt.operator_npub() into the state.
    rt.operator_npub = MagicMock(return_value=OPERATOR)
    return rt


def _register(rt):
    tools: dict = {}

    def fake_slug_tool(_mcp, _slug):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        register_standard_tools(MagicMock(), "test", rt, service_name="test")
    return tools


# ── begin_oauth ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_begin_oauth_happy_builds_url_and_stores_verifier():
    rt = _runtime()
    tools = _register(rt)
    with patch("tollbooth.registry.resolve_service_by_name",
               new=AsyncMock(return_value={"url": "https://collector.example/"})), \
         patch("tollbooth.oauth2_collector.generate_pkce_pair",
               return_value=("VERIFIER", "CHALLENGE")), \
         patch("tollbooth.oauth2_collector.build_authorize_url",
               return_value="https://provider.example/authorize?x=1") as mk_url, \
         patch("tollbooth.shortlinks.create_shortlink",
               new=AsyncMock(return_value="https://s.example/abc")):
        r = await tools["begin_oauth"](npub=PATRON, dpop_token="ok")

    assert r["success"] is True and r["status"] == "pending"
    assert r["authorize_url"] == "https://provider.example/authorize?x=1"
    assert r["authorize_url_short"] == "https://s.example/abc"
    # PKCE verifier + redirect_uri persisted for check_oauth_status
    stored = rt.store_patron_session.await_args.args[1]
    assert stored["pkce_verifier"] == "VERIFIER"
    assert stored["redirect_uri"] == "https://collector.example"  # trailing / stripped
    # client_id from operator creds; state packs patron npub (lookup key) plus the
    # operator npub the collector seals the code to.
    assert mk_url.call_args.args[1] == "CID"
    assert mk_url.call_args.args[3] == f"{PATRON}.{OPERATOR}"


@pytest.mark.asyncio
async def test_begin_oauth_creds_not_delivered():
    rt = _runtime()
    rt.load_credentials = AsyncMock(side_effect=RuntimeError("no creds"))
    tools = _register(rt)
    r = await tools["begin_oauth"](npub=PATRON, dpop_token="ok")
    assert r["success"] is False and "have not been delivered yet" in r["error"]


@pytest.mark.asyncio
async def test_begin_oauth_empty_client_id():
    rt = _runtime()
    rt.load_credentials = AsyncMock(return_value={"app_key": "", "secret": "x"})
    tools = _register(rt)
    r = await tools["begin_oauth"](npub=PATRON, dpop_token="ok")
    assert r["success"] is False and "have not been delivered yet" in r["error"]


@pytest.mark.asyncio
async def test_begin_oauth_collector_not_found():
    rt = _runtime()
    tools = _register(rt)
    with patch("tollbooth.registry.resolve_service_by_name",
               new=AsyncMock(side_effect=RuntimeError("404"))):
        r = await tools["begin_oauth"](npub=PATRON, dpop_token="ok")
    assert r["success"] is False and "OAuth2 collector not found" in r["error"]


# ── check_oauth_status ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_status_no_pending_flow():
    rt = _runtime()
    rt.load_patron_session = AsyncMock(return_value=(None, ""))
    tools = _register(rt)
    r = await tools["check_oauth_status"](npub=PATRON, dpop_token="ok")
    assert r["success"] is False and "No pending OAuth flow" in r["error"]


@pytest.mark.asyncio
async def test_check_status_code_pending():
    rt = _runtime()
    rt.load_patron_session = AsyncMock(return_value=({"redirect_uri": "https://c.example", "pkce_verifier": "V"}, ""))
    tools = _register(rt)
    with patch("tollbooth.registry.resolve_service_by_name",
               new=AsyncMock(return_value={"url": "https://collector.example"})), \
         patch("tollbooth.oauth2_collector.retrieve_code_from_collector",
               new=AsyncMock(return_value=None)):
        r = await tools["check_oauth_status"](npub=PATRON, dpop_token="ok")
    assert r["success"] is True and r["status"] == "pending"


@pytest.mark.asyncio
async def test_check_status_completes_and_persists_tokens():
    rt = _runtime()
    rt.load_patron_session = AsyncMock(return_value=({"redirect_uri": "https://c.example", "pkce_verifier": "V"}, ""))
    tools = _register(rt)
    token = {"access_token": "AT", "token_type": "Bearer", "refresh_token": "RT", "expires_at": 1999}
    with patch("tollbooth.registry.resolve_service_by_name",
               new=AsyncMock(return_value={"url": "https://collector.example"})), \
         patch("tollbooth.oauth2_collector.retrieve_code_from_collector",
               new=AsyncMock(return_value="AUTHCODE")), \
         patch("tollbooth.oauth2_collector.exchange_code_for_token",
               new=AsyncMock(return_value=token)) as mk_exch:
        r = await tools["check_oauth_status"](npub=PATRON, dpop_token="ok")

    assert r["success"] is True and r["status"] == "completed"
    # code exchanged with the operator creds + PKCE verifier
    assert mk_exch.await_args.args[0] == "AUTHCODE"
    assert mk_exch.await_args.kwargs["code_verifier"] == "V"
    # tokens persisted to the oauth service vault
    stored = rt.store_patron_session.await_args.args[1]
    assert stored["access_token"] == "AT" and stored["refresh_token"] == "RT"
    assert rt.store_patron_session.await_args.kwargs["service"] == "testoauth"


@pytest.mark.asyncio
async def test_check_status_exchange_failure_unclassified():
    """An unexpected exception still surfaces as a structured situation, not
    the old blind "Token exchange failed. Check operator logs." """
    rt = _runtime()
    rt.load_patron_session = AsyncMock(return_value=({"redirect_uri": "https://c.example", "pkce_verifier": "V"}, ""))
    tools = _register(rt)
    with patch("tollbooth.registry.resolve_service_by_name",
               new=AsyncMock(return_value={"url": "https://collector.example"})), \
         patch("tollbooth.oauth2_collector.retrieve_code_from_collector",
               new=AsyncMock(return_value="AUTHCODE")), \
         patch("tollbooth.oauth2_collector.exchange_code_for_token",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        r = await tools["check_oauth_status"](npub=PATRON, dpop_token="ok")
    assert r["success"] is False
    assert r["error_code"] == "oauth_exchange_failed_unclassified"
    assert "RuntimeError" in r["error"]
    assert "boom" in r["detail"]


@pytest.mark.asyncio
async def test_check_status_exchange_surfaces_provider_refusal():
    """A classified exchange 400 carries fault/oauth_error/X's words — the
    defect that left both patron and operator blind behind raise_for_status."""
    from tollbooth.oauth2_collector import OAuthRefreshDenied

    rt = _runtime()
    rt.load_patron_session = AsyncMock(return_value=({"redirect_uri": "https://c.example", "pkce_verifier": "V"}, ""))
    tools = _register(rt)
    denied = OAuthRefreshDenied(
        "400 invalid_client: Client authentication failed due to unknown client",
        status_code=400,
        oauth_error="invalid_client",
        fault="client",
    )
    with patch("tollbooth.registry.resolve_service_by_name",
               new=AsyncMock(return_value={"url": "https://collector.example"})), \
         patch("tollbooth.oauth2_collector.retrieve_code_from_collector",
               new=AsyncMock(return_value="AUTHCODE")), \
         patch("tollbooth.oauth2_collector.exchange_code_for_token",
               new=AsyncMock(side_effect=denied)):
        r = await tools["check_oauth_status"](npub=PATRON, dpop_token="ok")

    assert r["success"] is False
    assert r["error_code"] == "operator_app_credentials_rejected"
    assert r["upstream_oauth_error"] == "invalid_client"
    assert r["upstream_status"] == 400
    assert "unknown client" in r["detail"]
    assert "begin_oauth" not in " ".join(r["next_steps"]).lower()


@pytest.mark.asyncio
async def test_check_status_exchange_invalid_grant_routes_to_restart():
    from tollbooth.oauth2_collector import OAuthRefreshDenied

    rt = _runtime()
    rt.load_patron_session = AsyncMock(return_value=({"redirect_uri": "https://c.example", "pkce_verifier": "V"}, ""))
    tools = _register(rt)
    denied = OAuthRefreshDenied(
        "400 invalid_grant: code has already been used",
        status_code=400,
        oauth_error="invalid_grant",
        fault="grant",
    )
    with patch("tollbooth.registry.resolve_service_by_name",
               new=AsyncMock(return_value={"url": "https://collector.example"})), \
         patch("tollbooth.oauth2_collector.retrieve_code_from_collector",
               new=AsyncMock(return_value="AUTHCODE")), \
         patch("tollbooth.oauth2_collector.exchange_code_for_token",
               new=AsyncMock(side_effect=denied)):
        r = await tools["check_oauth_status"](npub=PATRON, dpop_token="ok")

    assert r["success"] is False
    assert r["error_code"] == "oauth_exchange_grant_rejected"
    assert r["upstream_oauth_error"] == "invalid_grant"
    assert "code has already been used" in r["detail"]
    assert "begin_oauth" in " ".join(r["next_steps"]).lower()


@pytest.mark.asyncio
async def test_check_status_on_token_received_merges_extra():
    rt = _runtime(on_token=AsyncMock(return_value={"account_hash": "H123"}))
    rt.load_patron_session = AsyncMock(return_value=({"redirect_uri": "https://c.example"}, ""))
    tools = _register(rt)
    token = {"access_token": "AT", "token_type": "Bearer"}
    with patch("tollbooth.registry.resolve_service_by_name",
               new=AsyncMock(return_value={"url": "https://collector.example"})), \
         patch("tollbooth.oauth2_collector.retrieve_code_from_collector",
               new=AsyncMock(return_value="AUTHCODE")), \
         patch("tollbooth.oauth2_collector.exchange_code_for_token",
               new=AsyncMock(return_value=token)):
        r = await tools["check_oauth_status"](npub=PATRON, dpop_token="ok")

    assert r["success"] is True
    stored = rt.store_patron_session.await_args.args[1]
    assert stored["account_hash"] == "H123"  # callback extra merged into vault data
