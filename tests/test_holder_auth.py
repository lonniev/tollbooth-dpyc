"""Holder bearer: a connection holds proof grants so tool calls name only the npub.

Pins:
  - the OAuth 2.1 flow a client like Claude.ai runs at connect (DCR, PKCE code,
    rotating refresh tokens, reuse disconnects the holder);
  - every bearer that is not ours is an anonymous caller, proving nobody;
  - a held grant is checked by require_held_grant, and only for the npub it
    was proven for;
  - receive_npub_proof puts the grant in the wallet and keeps it out of the reply;
  - forget_credentials drops the npub from every wallet.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import secrets
import socket
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.tests import run_server_async
from mcp.server.auth.provider import AuthorizationParams, TokenError
from mcp.shared.auth import OAuthClientInformationFull
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.constants import ErrorCode
from tollbooth.holder_auth import ANONYMOUS, HolderAuthProvider
from tollbooth.identity_credential import sign_proof_grant
from tollbooth.mcp_client import service_client
from tollbooth.runtime import OperatorRuntime, register_standard_tools
from tollbooth.tools.courier import forget_credentials_tool
from tollbooth.vaults.holder_store import HolderStore, TokenKind, TokenRow, token_hash

NONCE = "bold-hawk-42"


class MemoryHolderStore:
    """HolderStore's contract, kept in memory."""

    def __init__(self) -> None:
        self.clients: dict[str, str] = {}
        self.tokens: dict[tuple[str, str], TokenRow] = {}
        self.wallet: dict[tuple[str, str], tuple[str, int]] = {}

    async def save_client(self, client_id: str, info_json: str) -> None:
        self.clients[client_id] = info_json

    async def client(self, client_id: str) -> str | None:
        return self.clients.get(client_id)

    async def put_token(
        self, token: str, kind: TokenKind, *, holder_id: str, client_id: str,
        expires_at: int, meta: dict[str, Any],
    ) -> None:
        self.tokens[(token_hash(token), kind)] = TokenRow(holder_id, client_id, expires_at, False, meta)

    async def token(self, token: str, kind: TokenKind) -> TokenRow | None:
        return self.tokens.get((token_hash(token), kind))

    async def spend_token(self, token: str, kind: TokenKind) -> TokenRow | None:
        row = await self.token(token, kind)
        if row is None or not row.live:
            return None
        spent = dataclasses.replace(row, spent=True)
        self.tokens[(token_hash(token), kind)] = spent
        return spent

    async def purge_expired(self) -> None:
        now = time.time()
        self.tokens = {k: r for k, r in self.tokens.items() if r.expires_at >= now}

    async def hold(self, holder_id: str, npub: str, envelope: str, expires_at: int) -> None:
        self.wallet[(holder_id, npub)] = (envelope, expires_at)

    async def envelope(self, holder_id: str, npub: str) -> str | None:
        held = self.wallet.get((holder_id, npub))
        return held[0] if held and held[1] > time.time() else None

    async def holdings(self, holder_id: str) -> list[dict[str, Any]]:
        return [
            {"npub": n, "expires_at": e}
            for (h, n), (_, e) in sorted(self.wallet.items()) if h == holder_id
        ]

    async def release(self, npub: str) -> None:
        self.wallet = {k: v for k, v in self.wallet.items() if k[1] != npub}

    async def forget_holder(self, holder_id: str) -> None:
        self.tokens = {k: r for k, r in self.tokens.items() if r.holder_id != holder_id}
        self.wallet = {k: v for k, v in self.wallet.items() if k[0] != holder_id}

    def holder_of(self, access_token: str) -> str:
        return self.tokens[(token_hash(access_token), "access")].holder_id


def test_memory_store_keeps_holder_store_contract():
    public = {n for n in vars(HolderStore) if not n.startswith("_")}
    assert public <= set(vars(MemoryHolderStore))


def _envelope(op_nsec: str, npub: str, ttl: int = 3600) -> str:
    grant = sign_proof_grant(
        patron_npub=npub,
        operator_nsec=op_nsec,
        challenge_hash=hashlib.sha256(NONCE.encode()).hexdigest(),
        patron_sig="ab" * 64,
        patron_event_id="seal-1",
        ttl_seconds=ttl,
    )
    return json.dumps({"grant": json.loads(grant), "nonce": NONCE}, separators=(",", ":"))


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    return verifier, challenge.decode()


def _provider(store: MemoryHolderStore) -> HolderAuthProvider:
    return HolderAuthProvider(AsyncMock(return_value=store), base_url="https://op.example")


CLIENT = OAuthClientInformationFull(client_id="c1", redirect_uris=["http://localhost/cb"])


async def _connect(provider: HolderAuthProvider, resource: str | None = None):
    """Authorize and redeem the code, as a client does at connect."""
    url = await provider.authorize(CLIENT, AuthorizationParams(
        state="s", scopes=[], code_challenge="x", redirect_uri="http://localhost/cb",
        redirect_uri_provided_explicitly=True, resource=resource,
    ))
    code = httpx.URL(url).params["code"]
    loaded = await provider.load_authorization_code(CLIENT, code)
    assert loaded is not None
    return await provider.exchange_authorization_code(CLIENT, loaded)


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_bearer_that_is_not_ours_is_an_anonymous_caller():
    token = await _provider(MemoryHolderStore()).verify_token("anything-at-all")
    assert token is not None and token.client_id == ANONYMOUS and not token.claims


@pytest.mark.asyncio
async def test_an_unknown_holder_token_is_refused_so_the_client_refreshes():
    assert await _provider(MemoryHolderStore()).verify_token("dpyc_ha_forged") is None


@pytest.mark.asyncio
async def test_a_code_is_redeemed_once():
    store = MemoryHolderStore()
    provider = _provider(store)
    url = await provider.authorize(CLIENT, AuthorizationParams(
        state=None, scopes=[], code_challenge="x", redirect_uri="http://localhost/cb",
        redirect_uri_provided_explicitly=True,
    ))
    code = await provider.load_authorization_code(CLIENT, httpx.URL(url).params["code"])
    assert code is not None
    await provider.exchange_authorization_code(CLIENT, code)
    with pytest.raises(TokenError):
        await provider.exchange_authorization_code(CLIENT, code)


@pytest.mark.asyncio
async def test_refresh_rotates_and_a_replayed_refresh_disconnects_the_holder():
    store = MemoryHolderStore()
    provider = _provider(store)
    first = await _connect(provider)
    holder = store.holder_of(first.access_token)
    await store.hold(holder, "npub1x", "envelope", int(time.time()) + 60)

    old = await provider.load_refresh_token(CLIENT, first.refresh_token or "")
    assert old is not None
    second = await provider.exchange_refresh_token(CLIENT, old, [])
    assert (await provider.verify_token(second.access_token)).claims["holder"] == holder  # type: ignore[union-attr]

    # The rotated token comes back: someone else has it. Everything goes.
    assert await provider.load_refresh_token(CLIENT, first.refresh_token or "") is None
    assert await provider.verify_token(second.access_token) is None
    assert await store.holdings(holder) == []


@pytest.mark.asyncio
async def test_a_token_minted_for_another_resource_is_refused_here():
    provider = _provider(MemoryHolderStore())
    provider.set_mcp_path("/mcp")
    elsewhere = await _connect(provider, resource="https://other.example/mcp")
    here = await _connect(provider, resource="https://op.example/mcp")
    assert await provider.verify_token(elsewhere.access_token) is None
    assert await provider.verify_token(here.access_token) is not None


# ---------------------------------------------------------------------------
# Over HTTP, the way a connector meets it
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.asyncio
async def test_connect_then_act_as_held_npubs_over_http():
    op = PrivateKey()
    alice, bob = PrivateKey().public_key.bech32(), PrivateKey().public_key.bech32()
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    store = MemoryHolderStore()

    mcp = FastMCP("holder-test")
    rt = OperatorRuntime(nsec_env_var="__UNUSED__", holder_auth_url=base)
    rt._nsec = op.bech32()
    rt.holder_store = AsyncMock(return_value=store)  # type: ignore[method-assign]
    rt.proof_grant_revocations = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(is_revoked=AsyncMock(return_value=False)),
    )
    register_standard_tools(mcp, "test", rt, service_name="test")

    @mcp.tool
    async def act_as(npub: str) -> dict[str, Any]:
        return await rt.require_caller_proof(npub) or {"acted_as": npub}

    async def call(bearer: str | None, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
        async with Client(StreamableHttpTransport(f"{base}/mcp", headers=headers)) as client:
            return (await client.call_tool(tool, args)).structured_content or {}

    async with run_server_async(mcp, port=port), httpx.AsyncClient(base_url=base) as http:
        # No header: 401, which is what starts a connector's Authorization Code connect.
        assert (await http.post("/mcp", json={})).status_code == 401
        # Another DPYC service gets in with the anonymous bearer, and no OAuth.
        async with service_client(f"{base}/mcp") as peer:
            assert any(t.name == "test_held_identities" for t in await peer.list_tools())

        # Connect: register, authorize (no consent page), redeem with PKCE.
        reg = (await http.post("/register", json={
            "redirect_uris": ["http://localhost/cb"],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        })).json()
        verifier, challenge = _pkce()
        auth = await http.get("/authorize", params={
            "response_type": "code", "client_id": reg["client_id"],
            "redirect_uri": "http://localhost/cb", "state": "s",
            "code_challenge": challenge, "code_challenge_method": "S256",
            "resource": f"{base}/mcp",
        })
        assert auth.status_code == 302

        # A web page asks the same by fetch, and reads the code without leaving itself.
        fetched = await http.get("/authorize", headers={"Accept": "application/json"}, params={
            "response_type": "code", "client_id": reg["client_id"],
            "redirect_uri": "http://localhost/cb", "state": "f",
            "code_challenge": challenge, "code_challenge_method": "S256",
            "resource": f"{base}/mcp",
        })
        assert fetched.status_code == 200
        assert fetched.headers["access-control-allow-origin"] == "*"
        answered = httpx.URL(fetched.json()["redirect_uri"])
        assert answered.params["state"] == "f" and answered.params["code"].startswith("dpyc_hc_")

        tokens = (await http.post("/token", data={
            "grant_type": "authorization_code", "client_id": reg["client_id"],
            "code": httpx.URL(auth.headers["location"]).params["code"],
            "redirect_uri": "http://localhost/cb", "code_verifier": verifier,
            "resource": f"{base}/mcp",
        })).json()
        access = tokens["access_token"]

        # A new holder holds nothing; proving alice puts her grant in its wallet.
        assert (await call(access, "test_held_identities", {}))["identities"] == []
        await store.hold(store.holder_of(access), alice, _envelope(op.bech32(), alice), int(time.time()) + 3600)
        held = await call(access, "test_held_identities", {})
        assert [i["npub"] for i in held["identities"]] == [alice]

        # The holder acts as alice by naming her — and only her.
        assert await call(access, "act_as", {"npub": alice}) == {"acted_as": alice}
        refused = await call(access, "act_as", {"npub": bob})
        assert refused["error_code"] == ErrorCode.PROOF_REQUIRED

        # Any other bearer is anonymous: it proves nobody.
        assert (await call("anonymous", "act_as", {"npub": alice}))["error_code"] == ErrorCode.PROOF_REQUIRED


# ---------------------------------------------------------------------------
# The Courier hand-off
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_holder_receives_its_proof_into_the_wallet_never_the_reply():
    from tests.test_proof_tools_characterization import (
        PIN,
        SIGNED,
        FakeExchange,
        _delimited,
        _register,
        _runtime_with_exchange,
    )

    patron = PrivateKey().public_key.bech32()
    ex = FakeExchange(candidates=[{
        "id": "e1", "_relay": PIN, **SIGNED, "_plaintext": _delimited(nonce=NONCE),
    }])
    rt, _ = _runtime_with_exchange(ex)
    rt._holder_auth_url = "https://op.example"
    store = MemoryHolderStore()
    rt.holder_store = AsyncMock(return_value=store)  # type: ignore[method-assign]
    tools = _register(rt)

    with patch("tollbooth.holder_auth.current_holder", return_value="h1"):
        reply = await tools["receive_npub_proof"](patron_npub=patron, nonce=NONCE)
        assert await rt.require_caller_proof(patron) is None

    assert reply["success"] is True and reply["held_by_connection"] is True
    assert not {"grant", "nonce", "envelope"} & set(reply)
    held = await store.envelope("h1", patron)
    assert held is not None and json.loads(held)["nonce"] == NONCE


@pytest.mark.asyncio
async def test_forgetting_an_npub_drops_it_from_every_wallet():
    store = MemoryHolderStore()
    await store.hold("h1", "npub1a", "e", int(time.time()) + 60)
    await store.hold("h2", "npub1a", "e", int(time.time()) + 60)
    await store.hold("h2", "npub1b", "e", int(time.time()) + 60)
    rt = SimpleNamespace(
        operator_credential_service="op",
        patron_credential_service="patron",
        require_caller_proof=AsyncMock(return_value=None),
        courier=AsyncMock(return_value=SimpleNamespace(forget=AsyncMock(return_value={"success": True}))),
        proof_grant_revocations=AsyncMock(return_value=SimpleNamespace(revoke_all=AsyncMock())),
        holder_auth_enabled=True,
        holder_store=AsyncMock(return_value=store),
        _on_forget=None,
    )

    result = await forget_credentials_tool(rt, "patron", "npub1a")

    assert result["proof_grants_revoked"] is True
    assert set(store.wallet) == {("h2", "npub1b")}


# ---------------------------------------------------------------------------
# Hardening
# ---------------------------------------------------------------------------


def test_the_store_refuses_a_vault_without_a_cipher():
    from tollbooth.vaults.holder_store import HolderStoreUnavailable

    with pytest.raises(HolderStoreUnavailable):
        HolderStore(SimpleNamespace(_cipher=None))


@pytest.mark.asyncio
async def test_a_verified_token_is_read_once_a_minute_not_per_request():
    store = MemoryHolderStore()
    provider = _provider(store)
    tokens = await _connect(provider)
    store.token = AsyncMock(wraps=store.token)  # type: ignore[method-assign]

    for _ in range(3):
        assert await provider.verify_token(tokens.access_token) is not None
    assert store.token.await_count == 1


@pytest.mark.asyncio
async def test_a_store_that_cannot_answer_is_a_retry_never_a_401():
    from starlette.testclient import TestClient

    from tollbooth.holder_auth import _AnswerUnavailable
    from tollbooth.vaults.holder_store import HolderStoreUnavailable

    store = MemoryHolderStore()
    store.token = AsyncMock(side_effect=RuntimeError("neon asleep"))  # type: ignore[method-assign]
    with pytest.raises(HolderStoreUnavailable):
        await _provider(store).verify_token("dpyc_ha_x")

    async def failing(scope, receive, send):
        raise HolderStoreUnavailable("neon asleep")

    response = TestClient(_AnswerUnavailable(failing)).get("/mcp")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "10"
    assert "neon" not in response.text


def test_client_registration_is_metered_per_caller():
    from starlette.responses import PlainTextResponse
    from starlette.testclient import TestClient

    from tollbooth.holder_auth import REGISTRATIONS_PER_HOUR, _LimitRegistration

    client = TestClient(_LimitRegistration(PlainTextResponse("registered")))
    codes = [
        client.post("/register", headers={"X-Forwarded-For": "203.0.113.9, 10.0.0.1"}).status_code
        for _ in range(REGISTRATIONS_PER_HOUR + 1)
    ]
    assert codes[:-1] == [200] * REGISTRATIONS_PER_HOUR and codes[-1] == 429
    assert client.post("/register", headers={"X-Forwarded-For": "198.51.100.7"}).status_code == 200
    assert client.get("/mcp", headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 200


def test_an_authorize_refusal_asked_by_fetch_is_readable_and_keeps_its_status():
    from starlette.responses import JSONResponse
    from starlette.testclient import TestClient

    from tollbooth.holder_auth import _AuthorizeByFetch

    client = TestClient(_AuthorizeByFetch(JSONResponse({"error": "invalid_request"}, status_code=400)))
    refused = client.get("/authorize", headers={"Accept": "application/json"})
    assert refused.status_code == 400 and refused.json() == {"error": "invalid_request"}
    assert refused.headers["access-control-allow-origin"] == "*"
    # Without the ask, and on any other path, nothing is changed.
    assert "access-control-allow-origin" not in client.get("/authorize").headers
    assert "access-control-allow-origin" not in client.get("/token", headers={"Accept": "application/json"}).headers


@pytest.mark.asyncio
async def test_authorizing_clears_lapsed_codes():
    store = MemoryHolderStore()
    await store.put_token("dpyc_hc_old", "code", holder_id="h", client_id="c1", expires_at=1, meta={})
    await _connect(_provider(store))
    assert await store.token("dpyc_hc_old", "code") is None


@pytest.mark.asyncio
async def test_an_unreadable_wallet_is_warming_up_never_proof_required():
    rt = OperatorRuntime(nsec_env_var="__UNUSED__", holder_auth_url="https://op.example")
    rt.holder_store = AsyncMock(side_effect=RuntimeError("neon asleep"))  # type: ignore[method-assign]
    with patch("tollbooth.holder_auth.current_holder", return_value="h1"):
        denial = await rt.prove_caller(PrivateKey().public_key.bech32())
    assert denial is not None and denial["error_code"] == ErrorCode.WARMING_UP
