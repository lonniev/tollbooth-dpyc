"""Peer grants: a patron lets another npub see their account and pay for them.

The peer proves only its own npub. With the patron's approval it passes the four
tools behind ``require_caller_or_peer`` (check_balance, account_statement,
purchase_credits, check_payment) for the patron — and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, patch

import pytest
from pynostr.key import PrivateKey

from tests.test_holder_auth import _envelope
from tests.test_proof_tools_characterization import (
    PIN,
    SIGNED,
    FakeExchange,
    _delimited,
    _register,
    _runtime_with_exchange,
)
from tollbooth.constants import ErrorCode
from tollbooth.credential_templates import courier_form_schema
from tollbooth.identity_credential import (
    PEER_GRANT_SCOPE,
    IdentityCredentialError,
    sign_peer_grant,
    sign_proof_grant,
    verify_peer_grant,
    verify_proof_grant,
)
from tollbooth.tools.peer import _peer_channel
from tollbooth.vaults.peer_grant_store import PeerGrantStore

NONCE = "calm-lake-17"
CHALLENGE = hashlib.sha256(NONCE.encode()).hexdigest()


class MemoryPeerGrantStore:
    """PeerGrantStore's contract, kept in memory."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], tuple[str, int]] = {}

    async def put(self, patron_npub, peer_npub, envelope, expires_at):
        self.rows[(patron_npub, peer_npub)] = (envelope, int(expires_at))

    async def get(self, patron_npub, peer_npub):
        import time
        row = self.rows.get((patron_npub, peer_npub))
        return row[0] if row and row[1] > time.time() else None

    async def peers_of(self, patron_npub):
        return [{"peer_npub": q, "expires_at": e} for (p, q), (_, e) in sorted(self.rows.items()) if p == patron_npub]

    async def patrons_of(self, peer_npub):
        return [{"patron_npub": p, "expires_at": e} for (p, q), (_, e) in sorted(self.rows.items()) if q == peer_npub]

    async def remove(self, patron_npub, peer_npub):
        return self.rows.pop((patron_npub, peer_npub), None) is not None

    async def remove_all(self, patron_npub):
        for key in [k for k in self.rows if k[0] == patron_npub]:
            del self.rows[key]


@pytest.fixture(autouse=True)
def _a_holder_connection():
    """Every call here comes from holder connection h1, unless a test says otherwise."""
    with patch("tollbooth.holder_auth.current_holder", return_value="h1"):
        yield


def test_memory_store_keeps_peer_grant_store_contract():
    public = {n for n in vars(PeerGrantStore) if not n.startswith("_")}
    assert public <= set(vars(MemoryPeerGrantStore))


def _npub() -> str:
    return PrivateKey().public_key.bech32()


def _peer_envelope(op_nsec: str, patron: str, peer: str, ttl: int = 3600) -> str:
    grant = sign_peer_grant(
        patron, peer, op_nsec, challenge_hash=CHALLENGE,
        patron_sig="ab" * 64, patron_event_id="seal-1", ttl_seconds=ttl,
    )
    return json.dumps({"grant": json.loads(grant)}, separators=(",", ":"))


# -- the grant -----------------------------------------------------------------


class TestPeerGrantSignature:
    def setup_method(self):
        self.op = PrivateKey()
        self.patron, self.peer = _npub(), _npub()

    def _grant(self, **kw):
        return sign_peer_grant(
            kw.pop("patron", self.patron), kw.pop("peer", self.peer), self.op.bech32(),
            challenge_hash=CHALLENGE, patron_sig="ab" * 64, patron_event_id="seal-1", **kw,
        )

    def test_verifies_for_its_patron_and_peer(self):
        claims = verify_peer_grant(
            self._grant(), expected_patron_npub=self.patron, expected_peer_npub=self.peer,
            expected_operator_hex=self.op.public_key.hex(),
        )
        assert claims["peer_npub"] == self.peer and claims["scope"] == PEER_GRANT_SCOPE

    @pytest.mark.parametrize("field", ["patron", "peer"])
    def test_refuses_another_patron_or_peer(self, field):
        other = _npub()
        expected = {"expected_patron_npub": self.patron, "expected_peer_npub": self.peer}
        expected[f"expected_{field}_npub"] = other
        with pytest.raises(IdentityCredentialError, match="wrong_npub|wrong_peer"):
            verify_peer_grant(self._grant(), **expected)

    def test_refuses_expired_and_foreign_operator(self):
        with pytest.raises(IdentityCredentialError, match="expired"):
            verify_peer_grant(self._grant(ttl_seconds=-1),
                              expected_patron_npub=self.patron, expected_peer_npub=self.peer)
        with pytest.raises(IdentityCredentialError, match="operator_mismatch"):
            verify_peer_grant(self._grant(), expected_patron_npub=self.patron,
                              expected_peer_npub=self.peer, expected_operator_hex="00" * 32)

    def test_a_tampered_peer_breaks_the_signature(self):
        event = json.loads(self._grant())
        event["tags"] = [["peer", PrivateKey().public_key.hex()] if t[0] == "peer" else t for t in event["tags"]]
        with pytest.raises(IdentityCredentialError, match="bad_signature"):
            verify_peer_grant(json.dumps(event), expected_patron_npub=self.patron,
                              expected_peer_npub=self.peer)

    def test_a_proof_grant_is_never_a_peer_grant_and_back(self):
        proof = sign_proof_grant(self.patron, self.op.bech32(), challenge_hash=CHALLENGE,
                                 patron_sig="ab" * 64, patron_event_id="seal-1")
        with pytest.raises(IdentityCredentialError, match="not_peer_grant"):
            verify_peer_grant(proof, expected_patron_npub=self.patron, expected_peer_npub=self.peer)
        with pytest.raises(IdentityCredentialError, match="not_proof_grant"):
            verify_proof_grant(self._grant(), expected_patron_npub=self.patron)

    def test_a_patron_cannot_be_their_own_peer(self):
        with pytest.raises(IdentityCredentialError):
            self._grant(peer=self.patron)


# -- the gate ------------------------------------------------------------------


@pytest.fixture
def setup():
    """A runtime whose holder h1 holds the peer Q; P is the patron."""
    rt, revocations = _runtime_with_exchange(FakeExchange(candidates=[]))
    rt.peers = MemoryPeerGrantStore()  # type: ignore[attr-defined]
    rt.peer_grant_store = AsyncMock(return_value=rt.peers)  # type: ignore[method-assign]
    patron, peer = _npub(), _npub()
    return rt, revocations, patron, peer


async def _hold(rt, npub):
    await rt.wallet.hold("h1", npub, _envelope(rt._nsec, npub), 2**40)


async def _grant(rt, patron, peer, ttl=3600):
    await rt.peers.put(patron, peer, _peer_envelope(rt._nsec, patron, peer, ttl), 2**40)


@pytest.mark.asyncio
async def test_a_granted_peer_passes_the_four_tools_gate(setup):
    rt, _, patron, peer = setup
    await _hold(rt, peer)
    await _grant(rt, patron, peer)
    assert await rt.require_caller_or_peer(patron) is None


@pytest.mark.asyncio
async def test_a_granted_peer_still_fails_every_other_gate(setup):
    rt, _, patron, peer = setup
    await _hold(rt, peer)
    await _grant(rt, patron, peer)
    # Paid tools, credentials, coupons and properties: the patron's own proof only.
    assert (await rt.prove_caller(patron))["error_code"] == ErrorCode.PROOF_REQUIRED
    assert (await rt.require_caller_proof(patron))["error_code"] == ErrorCode.PROOF_REQUIRED


@pytest.mark.asyncio
async def test_a_paid_tool_refuses_the_peer(setup):
    from tollbooth.tool_identity import READS, ToolIdentity
    rt, _, patron, peer = setup
    await _hold(rt, peer)
    await _grant(rt, patron, peer)
    tid = "00000000-0000-4000-8000-00000000beef"
    rt._tool_registry[tid] = ToolIdentity(
        tool_id=tid, capability="paid_read", category="read",
        title="Paid read", behavior=READS, intent="t",
    )
    denied = await rt.debit_or_deny(tid, patron)
    assert denied is not None and denied.get("error_code") == ErrorCode.PROOF_REQUIRED, denied
    # The peer's own account is still the peer's: its proof passes the same gate.
    assert await rt.prove_caller(peer) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["no_grant", "expired", "wrong_peer", "unproven_peer", "revoked"])
async def test_the_gate_refuses(setup, case):
    rt, revocations, patron, peer = setup
    if case != "unproven_peer":
        await _hold(rt, peer)
    if case == "expired":
        await _grant(rt, patron, peer, ttl=-1)
    elif case == "wrong_peer":
        await _grant(rt, patron, _npub())
    elif case in ("unproven_peer", "revoked"):
        await _grant(rt, patron, peer)
    if case == "revoked":
        revocations.is_revoked = AsyncMock(return_value=True)
    err = await rt.require_caller_or_peer(patron)
    assert err is not None and err["error_code"] == ErrorCode.PROOF_REQUIRED
    assert "request_peer_grant" in err["peer_grant"]


@pytest.mark.asyncio
async def test_a_nip98_signer_of_the_peer_passes(setup):
    rt, _, patron, peer = setup
    await _grant(rt, patron, peer)
    with patch("tollbooth.holder_auth.signed_npub", return_value=peer), \
            patch("tollbooth.holder_auth.current_holder", return_value=None):
        assert await rt.require_caller_or_peer(patron) is None


@pytest.mark.asyncio
async def test_several_peers_each_pass_and_revoking_one_leaves_the_rest(setup):
    rt, _, patron, peer = setup
    other = _npub()
    await _hold(rt, peer)
    await _hold(rt, other)
    await _grant(rt, patron, peer)
    await _grant(rt, patron, other)
    tools = _register(rt)
    listed = await tools["list_peer_grants"](npub=patron)
    assert listed["error_code"] == ErrorCode.PROOF_REQUIRED, "only the patron lists"

    gone = await tools["revoke_peer_grant"](npub=patron, peer_npub=peer)
    assert gone["success"] and gone["revoked"], "a peer may drop its own grant"
    assert await rt.peers.get(patron, peer) is None
    assert await rt.require_caller_or_peer(patron) is None, "the other peer still passes"


@pytest.mark.asyncio
async def test_the_patron_lists_and_held_identities_shows_peer_of(setup):
    rt, _, patron, peer = setup
    await _hold(rt, patron)
    await _hold(rt, peer)
    await _grant(rt, patron, peer)
    tools = _register(rt)
    listed = await tools["list_peer_grants"](npub=patron)
    assert [p["peer_npub"] for p in listed["peers"]] == [peer]
    held = await tools["held_identities"]()
    assert held["peer_of"] == [{"patron_npub": patron, "expires_at": 2**40, "peer_npub": peer}]


@pytest.mark.asyncio
async def test_check_balance_answers_a_peer(setup):
    rt, _, patron, peer = setup
    await _hold(rt, peer)
    await _grant(rt, patron, peer)
    tools = _register(rt)
    with patch("tollbooth.tools.credits.check_balance_tool",
               AsyncMock(return_value={"success": True, "balance_api_sats": 5})) as balance, \
            patch.object(rt, "ledger_cache", AsyncMock(return_value=object())):
        r = await tools["check_balance"](npub=patron)
    assert r["success"] and balance.await_args.args[1] == patron


# -- the approval ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_records_the_grant_for_the_peer(setup):
    rt, _, patron, peer = setup
    await _hold(rt, peer)
    ex = FakeExchange(candidates=[{
        "id": "e1", "_relay": PIN, **SIGNED,
        "_plaintext": _delimited(confirm="ok", allow_for="7d", nonce=NONCE),
    }])
    rt._courier.__dict__["_exchange"] = ex
    tools = _register(rt)

    r = await tools["receive_peer_grant"](patron_npub=patron, peer_npub=peer, nonce=NONCE)

    assert r["success"] and r["peer_npub"] == peer
    assert not {"grant", "envelope"} & set(r), "the grant never enters the transcript"
    assert ex.resolved_for == peer, "only a caller proving the peer may collect"
    assert ex.closed == [(patron, f"peer_grant#{_peer_channel(peer)}")]
    envelope = await rt.peers.get(patron, peer)
    claims = verify_peer_grant(json.dumps(json.loads(envelope)["grant"]),
                               expected_patron_npub=patron, expected_peer_npub=peer)
    assert 6 * 86400 < claims["expiration"] - claims["created_at"] <= 7 * 86400
    assert any(peer[:16] in dm for dm in ex.sent_dms), "the patron is told who may watch"


@pytest.mark.asyncio
async def test_receive_refuses_a_caller_that_does_not_prove_the_peer(setup):
    rt, _, patron, peer = setup
    tools = _register(rt)
    r = await tools["receive_peer_grant"](patron_npub=patron, peer_npub=peer, nonce=NONCE)
    assert r["error_code"] == ErrorCode.PROOF_REQUIRED
    assert await rt.peers.get(patron, peer) is None


@pytest.mark.asyncio
async def test_request_goes_through_the_real_courier_and_its_template(setup, monkeypatch):
    """The 0.118.0 lesson: drive the real SecureCourierService, not a fake."""
    rt, _, patron, peer = setup
    await _hold(rt, peer)
    rt._courier = None
    monkeypatch.setenv("TOLLBOOTH_NOSTR_OPERATOR_NSEC", rt._nsec)
    with patch("tollbooth.relay_registry.ensure_relays_seeded", AsyncMock()), \
            patch("tollbooth.nostr_diagnostics.resolve_relays", return_value=["wss://relay.test"]), \
            patch.object(rt, "vault", AsyncMock(side_effect=RuntimeError("no vault"))):
        courier = await rt.courier()
    template = courier._exchange._templates["peer_grant"]
    form = courier_form_schema(template, nonce=NONCE, rendezvous_relay="wss://relay.test")
    assert form["x-courier-kind"] == "approval" and "allow_for" in form["properties"]

    tools = _register(rt)
    with patch.object(courier._exchange, "open_channel",
                      AsyncMock(return_value={"success": True, "nonce": NONCE})) as opened, \
            patch.object(courier._exchange, "_fetch_dms_from_relays"):
        r = await tools["request_peer_grant"](patron_npub=patron, peer_npub=peer)
    assert r["success"] and r["nonce"] == NONCE
    kw = opened.await_args.kwargs
    assert opened.await_args.args[0] == "peer_grant"
    assert kw["recipient_npub"] == patron and kw["bound_to"] == peer
    assert kw["channel"] == _peer_channel(peer) and peer in kw["greeting"]


@pytest.mark.asyncio
async def test_forget_drops_the_patrons_peers(setup):
    rt, revocations, patron, peer = setup
    await _hold(rt, patron)
    await _grant(rt, patron, peer)
    revocations.revoke_all = AsyncMock()
    from types import SimpleNamespace

    from tollbooth.tools.courier import forget_credentials_tool
    courier = SimpleNamespace(
        forget=AsyncMock(return_value={"success": True}), _exchange=rt._courier._exchange,
    )
    rt.courier = AsyncMock(return_value=courier)  # type: ignore[method-assign]
    rt.property_store = AsyncMock(return_value=SimpleNamespace(clear=AsyncMock(return_value=0)))  # type: ignore[method-assign]
    await forget_credentials_tool(rt, "x", patron)
    assert await rt.peers.peers_of(patron) == []
