"""Replay-cache bounding (audit S2 / QW5) for tollbooth.identity_proof.

The consumed-proof set rejects replayed proof event IDs. Entries are only
inserted after full signature + freshness verification, so the set cannot be
filled with junk — but a flood of distinct *valid* proofs could grow it
unbounded between the 120s lazy cleanups. ``_record_consumed`` enforces a hard
cap. These tests prove the cap holds, that eviction prefers expired entries,
and that genuine replay protection is unaffected.
"""

from __future__ import annotations

import json
import time

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

import tollbooth.identity_proof as idp
from tollbooth.identity_proof import (
    PROOF_EVENT_KIND,
    _consumed_proofs,
    _record_consumed,
    verify_proof,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty consumed-proof set."""
    _consumed_proofs.clear()
    yield
    _consumed_proofs.clear()


def _make_proof(private_key: PrivateKey, tool_name: str = "my_tool") -> str:
    event = Event(
        kind=PROOF_EVENT_KIND,
        content="",
        tags=[["u", tool_name]],
        pubkey=private_key.public_key.hex(),
    )
    event.created_at = int(time.time())
    event.sign(private_key.hex())
    return json.dumps(event.to_dict())


def test_cap_holds_under_flood() -> None:
    """50k distinct live entries must not exceed the hard cap."""
    future = time.time() + 3600
    for i in range(50_000):
        _record_consumed(f"event-{i}", future)
    assert len(_consumed_proofs) <= idp._CONSUMED_MAX_ENTRIES


def test_eviction_prefers_expired_entries() -> None:
    """Expired ids are purged before any live id is evicted."""
    now = time.time()
    # Fill to capacity with already-expired entries.
    for i in range(idp._CONSUMED_MAX_ENTRIES):
        _consumed_proofs[f"stale-{i}"] = now - 1.0
    # One more insert should drop expired ids, leaving the fresh one present.
    _record_consumed("fresh", now + 3600)
    assert "fresh" in _consumed_proofs
    assert len(_consumed_proofs) <= idp._CONSUMED_MAX_ENTRIES
    # The live 'fresh' entry survived; stale ones were the eviction target.
    assert _consumed_proofs.get("fresh") == now + 3600


def test_eviction_targets_soonest_to_expire_when_all_live() -> None:
    """With no expired ids, the entry closest to expiry is evicted first."""
    base = time.time() + 3600
    for i in range(idp._CONSUMED_MAX_ENTRIES):
        # Strictly increasing expiry; 'soon-0' is closest to expiry.
        _consumed_proofs[f"live-{i}"] = base + i
    _record_consumed("newest", base + 10_000)
    assert len(_consumed_proofs) <= idp._CONSUMED_MAX_ENTRIES
    assert "newest" in _consumed_proofs
    # The soonest-to-expire entry was the one evicted.
    assert "live-0" not in _consumed_proofs


def test_replay_protection_still_works() -> None:
    """A valid proof verifies once; the same proof replays as rejected."""
    pk = PrivateKey()
    npub = pk.public_key.bech32()
    dpop_token = _make_proof(pk, "my_tool")

    assert verify_proof(dpop_token, npub, "my_tool") is True
    # Same event id presented again → replay rejected.
    assert verify_proof(dpop_token, npub, "my_tool") is False
    assert len(_consumed_proofs) == 1
