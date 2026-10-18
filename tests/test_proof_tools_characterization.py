"""Characterization net for the proof tools (audit M2.1f Phase 1).

receive_npub_proof's inline drain loop — decrypt, nonce-match, NACK cap,
stop-at-match, the grant kept in the holder's wallet — is security-critical and was at ~0%
coverage. These tests pin its current behavior at the tool-closure level (real
rt for validation/resolve, fake courier exchange, stubbed persistence) so the
Phase 2 extraction onto rt can be verified behavior-preserving.

NOT a redesign — every assertion captures existing behavior verbatim.
"""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.constants import ErrorCode
from tollbooth.runtime import OperatorRuntime, register_standard_tools

os.environ.setdefault(
    "TOLLBOOTH_NOSTR_OPERATOR_NSEC",
    "nsec1test000000000000000000000000000000000000000000000000000000",
)

PIN = "wss://relay.pinned"


def _delimited(**fields) -> str:
    return "\n".join(f"{k} = @@@{v}@@@" for k, v in fields.items())


# What the real courier stamps on a candidate once it has verified the sender's
# signed layer (NIP-17 seal / NIP-04 event). The fake decrypt below does not,
# so candidates carry it up front; the no-signature case sets them to "".
SIGNED = {"_sender_sig": "ab" * 64, "_sender_event_id": "seal1"}


class FakeExchange:
    """Drives receive_npub_proof's drain loop with canned candidates."""

    def __init__(self, *, candidates, pinned=PIN, resolve_error=None, fetch_delay=0.0):
        self._candidates = candidates
        self._pinned = pinned
        self._resolve_error = resolve_error
        self._fetch_delay = fetch_delay
        self._privkey_hex = "ab" * 32
        self._ephemeral_agents: dict = {}
        self._pending_nonces: dict = {}
        self._pinned_relays: dict = {}
        self._credential_vault = None
        self.popped: list[tuple] = []      # (event_id, nacked: bool)
        self.sent_dms: list[str] = []

    async def _resolve_pinned_record(self, npub, service, nonce, *, bound_to=""):
        self.resolved_for = bound_to
        if self._resolve_error is not None:
            return None, self._resolve_error
        # seed channel state so the cleanup .pop()s have something to remove
        self._pending_nonces[(npub, service)] = (nonce, 0)
        self._pinned_relays[(npub, service)] = self._pinned
        return self._pinned, None

    def _fetch_dms_from_relays(self, relays=None):
        # Simulate a slow/timing-out relay (blocking, like websocket-client).
        if self._fetch_delay:
            time.sleep(self._fetch_delay)

    def _find_dm_candidates(self, patron_hex):
        return list(self._candidates)

    def _decrypt_dm(self, candidate, patron_hex, decrypt_privkey_hex=None):
        if candidate.get("_undecryptable"):
            raise ValueError("cannot decrypt")
        return candidate.get("_plaintext")

    def _pop_event(self, event_id, reply_npub=None, reason=None, target_relay=None):
        self.popped.append((event_id, reason is not None))

    async def _vault_store(self, service, npub, payload):
        return True

    def send_dm(self, npub, msg, target_relay=None):
        self.sent_dms.append(msg)


HOLDER = "h1"


@pytest.fixture(autouse=True)
def _a_holder_connection():
    """receive_npub_proof serves holder connections; every call here is one."""
    with patch("tollbooth.holder_auth.current_holder", return_value=HOLDER):
        yield


def _runtime_with_exchange(exchange, *, on_proven=None):
    from tests.test_holder_auth import MemoryHolderStore

    rt = OperatorRuntime(tool_registry={}, service_name="Test Operator",
                         holder_auth_url="https://op.example")
    rt.wallet = MemoryHolderStore()  # type: ignore[attr-defined]
    rt.holder_store = AsyncMock(return_value=rt.wallet)  # type: ignore[method-assign]
    rt._courier = SimpleNamespace(_exchange=exchange)
    rt._on_npub_proven = on_proven
    # Operator nsec so receive_npub_proof can mint a kind-30080 grant.
    rt._nsec = PrivateKey().bech32()
    # Stub persistence the drain doesn't exercise.
    rt.load_patron_session = AsyncMock(return_value=({}, ""))  # no challenge_ts
    revocations = SimpleNamespace(is_revoked=AsyncMock(return_value=False))
    rt.proof_grant_revocations = AsyncMock(return_value=revocations)
    return rt, revocations


def _register(rt):
    tools: dict = {}

    def fake_slug_tool(_mcp, _slug, _identify):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        register_standard_tools(MagicMock(), "test", rt, service_name="test")
    return tools


@pytest.fixture
def patron():
    pk = PrivateKey()
    return pk.public_key.bech32()


@pytest.mark.asyncio
async def test_match_holds_the_grant_and_never_returns_it(patron):
    nonce = "bold-hawk-42"
    ex = FakeExchange(candidates=[{
        "id": "e1", "_relay": PIN, **SIGNED,
        "_plaintext": _delimited(nonce=nonce),
    }])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, nonce=nonce)

    assert r["success"] is True
    assert r["proven_npub"] == patron
    assert r["held_by_connection"] is True
    assert not {"grant", "nonce", "envelope"} & set(r)
    import hashlib
    import json
    envelope = json.loads(await rt.wallet.envelope(HOLDER, patron))
    assert envelope["nonce"] == nonce
    grant = envelope["grant"]
    assert grant["kind"] == 30080
    tags = {t[0]: t[1] for t in grant["tags"]}
    # The grant binds the challenge hash and the VERIFIED reply signature —
    # never the raw nonce, never a placeholder.
    assert tags["challenge"] == hashlib.sha256(nonce.encode()).hexdigest()
    assert tags["patron_sig"] == SIGNED["_sender_sig"]
    assert tags["patron_event_id"] == SIGNED["_sender_event_id"]
    assert nonce not in json.dumps(grant)
    assert r["expires_in_seconds"] == 7200  # default consent window
    # matched DM popped without a NACK; confirmation DM sent
    assert ex.popped == [("e1", False)]
    assert ex.sent_dms and "confirmed" in ex.sent_dms[0]


@pytest.mark.asyncio
async def test_on_npub_proven_callback_fires_on_match(patron):
    nonce = "bold-hawk-42"
    cb = AsyncMock()
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, **SIGNED, "_plaintext": _delimited(nonce=nonce)}])
    rt, _ = _runtime_with_exchange(ex, on_proven=cb)
    tools = _register(rt)

    await tools["receive_npub_proof"](patron_npub=patron, nonce=nonce)
    cb.assert_awaited_once()
    assert cb.await_args.args[0] == patron


@pytest.mark.asyncio
async def test_wrong_nonce_nacks_and_reports_not_found(patron):
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, "_plaintext": _delimited(nonce="WRONG")}])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, nonce="bold-hawk-42")

    assert r["success"] is False
    assert r["error_code"] == ErrorCode.COURIER_NOT_FOUND
    assert "wrong token" in r["error"]
    assert ex.popped == [("e1", True)]      # NACK'd
    assert "nonce" not in r


@pytest.mark.asyncio
async def test_undecryptable_dm_nacks(patron):
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, "_undecryptable": True}])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, nonce="bold-hawk-42")
    assert r["success"] is False
    assert "undecryptable DM" in r["error"]
    assert ex.popped == [("e1", True)]


@pytest.mark.asyncio
async def test_no_candidates_returns_not_found(patron):
    ex = FakeExchange(candidates=[])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, nonce="bold-hawk-42")
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.COURIER_NOT_FOUND
    assert r["popped_dms"] == 0
    assert "No reply found" in r["error"]


@pytest.mark.asyncio
async def test_pinned_resolve_error_pops_nothing(patron):
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, "_plaintext": _delimited(nonce="x")}],
                      resolve_error=ErrorCode.COURIER_TOKEN_EXPIRED)
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, nonce="bold-hawk-42")
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.COURIER_TOKEN_EXPIRED
    assert r["popped_dms"] == 0
    assert ex.popped == []


@pytest.mark.asyncio
async def test_missing_nonce_rejected(patron):
    ex = FakeExchange(candidates=[])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, nonce="")
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.NONCE_MISSING


@pytest.mark.asyncio
async def test_old_timestamp_dm_with_correct_nonce_matches(patron):
    # Regression for the 0.44.10 fix: a reply whose created_at predates the
    # challenge (clock skew / human-paced reply / relay lag) used to be
    # silently dropped by a 5s timestamp gate. The one-time nonce is the sole
    # scoping mechanism now, so an old-but-correct reply MUST match.
    nonce = "bold-hawk-42"
    ex = FakeExchange(candidates=[
        {"id": "old", "_relay": PIN, "created_at": 100, **SIGNED,
         "_plaintext": _delimited(nonce=nonce)},
    ])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, nonce=nonce)
    assert r["success"] is True
    assert r["proven_npub"] == patron
    assert ex.popped == [("old", False)]  # matched, popped without NACK
    assert r["held_by_connection"] is True


@pytest.mark.asyncio
async def test_cache_duration_overrides_ttl(patron):
    nonce = "bold-hawk-42"
    ex = FakeExchange(candidates=[
        {"id": "e1", "_relay": PIN, **SIGNED,
         "_plaintext": _delimited(nonce=nonce, cache_duration="1 day")},
    ])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, nonce=nonce)
    # The patron's stated consent window is the grant's lifetime.
    import json
    import time as _t
    held = json.loads(await rt.wallet.envelope(HOLDER, patron))
    tags = {t[0]: t[1] for t in held["grant"]["tags"]}
    assert r["expires_in_seconds"] == 86400
    assert abs(int(tags["expiration"]) - (_t.time() + 86400)) < 5


@pytest.mark.asyncio
async def test_reply_without_a_verified_signature_issues_no_grant(patron):
    # The grant attests the patron's verified signature; if the courier could
    # not surface one, there is nothing to attest and no grant is minted.
    nonce = "bold-hawk-42"
    ex = FakeExchange(candidates=[
        {"id": "e1", "_relay": PIN, "_sender_sig": "", "_sender_event_id": "",
         "_plaintext": _delimited(nonce=nonce)},
    ])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, nonce=nonce)
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.PROOF_INVALID
    assert "grant" not in r
    assert await rt.wallet.envelope(HOLDER, patron) is None
    assert "no verifiable patron signature" in r["error"]


@pytest.mark.asyncio
async def test_match_clears_channel_state(patron):
    nonce = "bold-hawk-42"
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, **SIGNED, "_plaintext": _delimited(nonce=nonce)}])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    await tools["receive_npub_proof"](patron_npub=patron, nonce=nonce)
    key = (patron, "npub_ownership")
    # one-time-use cleanup removed the nonce/pin state
    assert key not in ex._pending_nonces
    assert key not in ex._pinned_relays


# ── request_npub_proof ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_request_sends_challenge_and_returns_token(patron):
    ex = FakeExchange(candidates=[])  # nothing stale to purge
    rt = OperatorRuntime(tool_registry={}, service_name="Test Operator",
                         holder_auth_url="https://op.example")
    rt._courier = SimpleNamespace(
        _exchange=ex,
        open_channel=AsyncMock(return_value={
            "success": True, "nonce": "bold-hawk-42", "rendezvous_relay": PIN,
        }),
    )
    rt.store_patron_session = AsyncMock()
    tools = _register(rt)

    r = await tools["request_npub_proof"](patron_npub=patron)
    assert r["success"] is True
    assert r["nonce"] == "bold-hawk-42"
    assert r["rendezvous_relay"] == PIN
    assert PIN in r["message"]
    # opened the channel on the proof service
    assert rt._courier.open_channel.await_args.args[0] == "npub_ownership"
    # bound to the holder that asked, so only it may collect the reply
    assert rt._courier.open_channel.await_args.kwargs["bound_to"] == HOLDER


@pytest.mark.asyncio
async def test_request_greeting_includes_request_timestamp(patron):
    # Issue #120: the proof-request preamble (the greeting the courier places
    # directly above the @@@ fields) must carry the request's date/time so the
    # patron sees when the challenge was raised. Kept to one terse line.
    ex = FakeExchange(candidates=[])
    rt = OperatorRuntime(tool_registry={}, service_name="Test Operator",
                         holder_auth_url="https://op.example")
    rt._courier = SimpleNamespace(
        _exchange=ex,
        open_channel=AsyncMock(return_value={
            "success": True, "nonce": "bold-hawk-42", "rendezvous_relay": PIN,
        }),
    )
    rt.store_patron_session = AsyncMock()
    tools = _register(rt)

    await tools["request_npub_proof"](patron_npub=patron)

    greeting = rt._courier.open_channel.await_args.kwargs["greeting"]
    assert "Requested:" in greeting
    # A UTC date/time stamp of the shape YYYY-MM-DD HH:MM UTC.
    import re
    assert re.search(r"Requested: \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", greeting)


@pytest.mark.asyncio
async def test_request_propagates_open_channel_failure(patron):
    ex = FakeExchange(candidates=[])
    rt = OperatorRuntime(tool_registry={}, service_name="Test",
                         holder_auth_url="https://op.example")
    rt._courier = SimpleNamespace(
        _exchange=ex,
        open_channel=AsyncMock(return_value={"success": False, "error": "relay down"}),
    )
    rt.store_patron_session = AsyncMock()
    tools = _register(rt)

    r = await tools["request_npub_proof"](patron_npub=patron)
    assert r["success"] is False and r["error"] == "relay down"


# ── check_proof_status ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_connection_that_is_no_holder_is_refused_before_draining(patron):
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, **SIGNED, "_plaintext": _delimited(nonce="n")}])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    with patch("tollbooth.holder_auth.current_holder", return_value=None):
        r = await tools["receive_npub_proof"](patron_npub=patron, nonce="n")
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.PROOF_REQUIRED
    assert ex.popped == []  # the reply is still there for a holder to collect


@pytest.mark.asyncio
async def test_check_proof_status_signed_request_is_valid(patron):
    rt = OperatorRuntime(tool_registry={}, service_name="Test")
    tools = _register(rt)
    with patch("tollbooth.holder_auth.signed_npub", return_value=patron):
        r = await tools["check_proof_status"](patron_npub=patron)
    assert r["success"] is True and r["status"] == "valid"


@pytest.mark.asyncio
async def test_check_proof_status_reads_the_holders_wallet(patron):
    nonce = "bold-hawk-42"
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, **SIGNED, "_plaintext": _delimited(nonce=nonce)}])
    rt, revocations = _runtime_with_exchange(ex)
    tools = _register(rt)

    before = await tools["check_proof_status"](patron_npub=patron)
    assert before["status"] == "not_held"

    await tools["receive_npub_proof"](patron_npub=patron, nonce=nonce)
    held = await tools["check_proof_status"](patron_npub=patron)
    assert held["status"] == "valid" and held["expires_in_seconds"] > 0

    revocations.is_revoked = AsyncMock(return_value=True)
    assert (await tools["check_proof_status"](patron_npub=patron))["status"] == "revoked"


# ── event loop not blocked by the relay drain (to_thread follow-up) ────

@pytest.mark.asyncio
async def test_receive_drain_does_not_block_event_loop(patron):
    nonce = "bold-hawk-42"
    # Slow relay fetch: if it ran inline, a concurrent coroutine couldn't run
    # until it finished. With asyncio.to_thread the loop stays responsive.
    ex = FakeExchange(
        candidates=[{"id": "e1", "_relay": PIN, **SIGNED, "_plaintext": _delimited(nonce=nonce)}],
        fetch_delay=0.25,
    )
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    progressed: list[float] = []

    async def _ticker():
        await asyncio.sleep(0.05)
        progressed.append(time.monotonic())

    start = time.monotonic()
    r, _ = await asyncio.gather(
        tools["receive_npub_proof"](patron_npub=patron, nonce=nonce),
        _ticker(),
    )
    assert r["success"] is True
    assert progressed, "ticker never ran"
    # tick landed during the 0.25s drain, not after it
    assert progressed[0] - start < 0.18, (
        f"event loop blocked: tick at {progressed[0] - start:.3f}s"
    )


@pytest.mark.asyncio
async def test_request_needs_a_holder_connection(patron):
    ex = FakeExchange(candidates=[])
    rt = OperatorRuntime(tool_registry={}, service_name="Test Operator",
                         holder_auth_url="https://op.example")
    rt._courier = SimpleNamespace(_exchange=ex, open_channel=AsyncMock())
    tools = _register(rt)
    with patch("tollbooth.holder_auth.current_holder", return_value=None):
        r = await tools["request_npub_proof"](patron_npub=patron)
    assert r["success"] is False
    rt._courier.open_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_receive_collects_only_on_the_requesting_holder(patron):
    ex = FakeExchange(candidates=[])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)
    await tools["receive_npub_proof"](patron_npub=patron, nonce="bold-hawk-42")
    assert ex.resolved_for == HOLDER
