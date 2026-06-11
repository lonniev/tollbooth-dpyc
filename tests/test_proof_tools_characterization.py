"""Characterization net for the proof tools (audit M2.1f Phase 1).

receive_npub_proof's inline drain loop — decrypt, poison-match, NACK cap,
stop-at-match, proven-npub cache write — is security-critical and was at ~0%
coverage. These tests pin its current behavior at the tool-closure level (real
rt for validation/resolve, fake courier exchange, stubbed persistence) so the
Phase 2 extraction onto rt can be verified behavior-preserving.

NOT a redesign — every assertion captures existing behavior verbatim.
"""

from __future__ import annotations

import os
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


class FakeExchange:
    """Drives receive_npub_proof's drain loop with canned candidates."""

    def __init__(self, *, candidates, pinned=PIN, resolve_error=None):
        self._candidates = candidates
        self._pinned = pinned
        self._resolve_error = resolve_error
        self._privkey_hex = "ab" * 32
        self._ephemeral_agents: dict = {}
        self._pending_poisons: dict = {}
        self._pinned_relays: dict = {}
        self._credential_vault = None
        self.popped: list[tuple] = []      # (event_id, nacked: bool)
        self.sent_dms: list[str] = []

    async def _resolve_pinned_record(self, npub, service, poison):
        if self._resolve_error is not None:
            return None, self._resolve_error
        # seed channel state so the cleanup .pop()s have something to remove
        self._pending_poisons[(npub, service)] = (poison, 0)
        self._pinned_relays[(npub, service)] = self._pinned
        return self._pinned, None

    def _fetch_dms_from_relays(self, relays=None):
        return None

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


def _runtime_with_exchange(exchange, *, on_proven=None):
    rt = OperatorRuntime(tool_registry={}, service_name="Test Operator")
    rt._courier = SimpleNamespace(_exchange=exchange)
    rt._on_npub_proven = on_proven
    # Stub persistence the drain doesn't exercise.
    rt.load_patron_session = AsyncMock(return_value={})  # no challenge_ts
    record = SimpleNamespace(verified_at=1000.0, expires_at=1000.0 + 7200)
    cache = SimpleNamespace(mark_proven=AsyncMock(return_value=record))
    rt.proven_npub_cache = AsyncMock(return_value=cache)
    return rt, cache


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


@pytest.fixture
def patron():
    pk = PrivateKey()
    return pk.public_key.bech32()


@pytest.mark.asyncio
async def test_match_marks_proven_and_returns_token(patron):
    poison = "bold-hawk-42"
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, "_plaintext": _delimited(poison=poison)}])
    rt, cache = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, poison=poison)

    assert r["success"] is True
    assert r["proof_token"] == poison
    assert r["proven_npub"] == patron
    # proven-npub cache written with sha256(poison) hash
    import hashlib
    expected_hash = hashlib.sha256(poison.encode()).hexdigest()
    assert cache.mark_proven.await_args.args[0] == expected_hash
    assert cache.mark_proven.await_args.args[1] == patron
    # matched DM popped without a NACK; confirmation DM sent
    assert ex.popped == [("e1", False)]
    assert ex.sent_dms and "confirmed" in ex.sent_dms[0]


@pytest.mark.asyncio
async def test_on_npub_proven_callback_fires_on_match(patron):
    poison = "bold-hawk-42"
    cb = AsyncMock()
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, "_plaintext": _delimited(poison=poison)}])
    rt, _ = _runtime_with_exchange(ex, on_proven=cb)
    tools = _register(rt)

    await tools["receive_npub_proof"](patron_npub=patron, poison=poison)
    cb.assert_awaited_once()
    assert cb.await_args.args[0] == patron


@pytest.mark.asyncio
async def test_wrong_poison_nacks_and_reports_not_found(patron):
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, "_plaintext": _delimited(poison="WRONG")}])
    rt, cache = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, poison="bold-hawk-42")

    assert r["success"] is False
    assert r["error_code"] == ErrorCode.COURIER_NOT_FOUND
    assert "wrong token" in r["error"]
    assert ex.popped == [("e1", True)]      # NACK'd
    cache.mark_proven.assert_not_awaited()


@pytest.mark.asyncio
async def test_undecryptable_dm_nacks(patron):
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, "_undecryptable": True}])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, poison="bold-hawk-42")
    assert r["success"] is False
    assert "undecryptable DM" in r["error"]
    assert ex.popped == [("e1", True)]


@pytest.mark.asyncio
async def test_no_candidates_returns_not_found(patron):
    ex = FakeExchange(candidates=[])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, poison="bold-hawk-42")
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.COURIER_NOT_FOUND
    assert r["popped_dms"] == 0
    assert "No reply found" in r["error"]


@pytest.mark.asyncio
async def test_pinned_resolve_error_pops_nothing(patron):
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, "_plaintext": _delimited(poison="x")}],
                      resolve_error=ErrorCode.COURIER_TOKEN_EXPIRED)
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, poison="bold-hawk-42")
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.COURIER_TOKEN_EXPIRED
    assert r["popped_dms"] == 0
    assert ex.popped == []


@pytest.mark.asyncio
async def test_missing_poison_rejected(patron):
    ex = FakeExchange(candidates=[])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, poison="")
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.POISON_MISSING


@pytest.mark.asyncio
async def test_pre_challenge_dm_popped_silently(patron):
    # A DM older than the challenge timestamp is popped without a NACK and
    # never matched — then the drain ends empty → not_found.
    ex = FakeExchange(candidates=[
        {"id": "old", "_relay": PIN, "created_at": 100,
         "_plaintext": _delimited(poison="bold-hawk-42")},
    ])
    rt, _ = _runtime_with_exchange(ex)
    rt.load_patron_session = AsyncMock(return_value={"challenge_ts": "1000"})
    tools = _register(rt)

    r = await tools["receive_npub_proof"](patron_npub=patron, poison="bold-hawk-42")
    assert r["success"] is False
    assert ex.popped == [("old", False)]  # silent pop, no NACK


@pytest.mark.asyncio
async def test_cache_duration_overrides_ttl(patron):
    poison = "bold-hawk-42"
    ex = FakeExchange(candidates=[
        {"id": "e1", "_relay": PIN,
         "_plaintext": _delimited(poison=poison, cache_duration="1 day")},
    ])
    rt, cache = _runtime_with_exchange(ex)
    tools = _register(rt)

    await tools["receive_npub_proof"](patron_npub=patron, poison=poison)
    # patron-chosen duration parsed and passed as ttl_override (not UNSET)
    assert cache.mark_proven.await_args.kwargs["ttl_override"] == 86400


@pytest.mark.asyncio
async def test_match_clears_channel_state(patron):
    poison = "bold-hawk-42"
    ex = FakeExchange(candidates=[{"id": "e1", "_relay": PIN, "_plaintext": _delimited(poison=poison)}])
    rt, _ = _runtime_with_exchange(ex)
    tools = _register(rt)

    await tools["receive_npub_proof"](patron_npub=patron, poison=poison)
    key = (patron, "npub_ownership")
    # one-time-use cleanup removed the poison/pin state
    assert key not in ex._pending_poisons
    assert key not in ex._pinned_relays


# ── request_npub_proof ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_request_sends_challenge_and_returns_token(patron):
    ex = FakeExchange(candidates=[])  # nothing stale to purge
    rt = OperatorRuntime(tool_registry={}, service_name="Test Operator")
    rt._courier = SimpleNamespace(
        _exchange=ex,
        open_channel=AsyncMock(return_value={
            "success": True, "poison": "bold-hawk-42", "rendezvous_relay": PIN,
        }),
    )
    rt.store_patron_session = AsyncMock()
    tools = _register(rt)

    r = await tools["request_npub_proof"](patron_npub=patron)
    assert r["success"] is True
    assert r["proof_token"] == "bold-hawk-42"
    assert r["rendezvous_relay"] == PIN
    assert PIN in r["message"]
    # opened the channel on the proof service
    assert rt._courier.open_channel.await_args.args[0] == "npub_ownership"


@pytest.mark.asyncio
async def test_request_propagates_open_channel_failure(patron):
    ex = FakeExchange(candidates=[])
    rt = OperatorRuntime(tool_registry={}, service_name="Test")
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
@pytest.mark.parametrize("status,frag", [
    ("valid", "Proof is valid"),
    ("expired", "Proof has expired"),
    ("missing", "No proof record found"),
])
async def test_check_proof_status_messages(patron, status, frag):
    rt = OperatorRuntime(tool_registry={}, service_name="Test")
    cache = SimpleNamespace(proof_status=AsyncMock(
        return_value={"status": status, "expires_in_seconds": 123},
    ))
    rt.proven_npub_cache = AsyncMock(return_value=cache)
    tools = _register(rt)

    r = await tools["check_proof_status"](patron_npub=patron, proof_token="bold-hawk-42")
    assert r["success"] is True
    assert r["status"] == status
    assert r["expires_in_seconds"] == 123
    assert frag in r["message"]
    # queried by sha256(proof_token)
    import hashlib
    assert cache.proof_status.await_args.args[0] == hashlib.sha256(b"bold-hawk-42").hexdigest()
