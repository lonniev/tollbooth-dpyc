"""Tests for the proven npub ownership cache."""

import json
import time

import pytest
from pynostr.event import Event
from pynostr.key import PrivateKey

from tollbooth.identity_proof import (
    OWNERSHIP_SENTINEL,
    PROOF_EVENT_KIND,
    create_ownership_proof,
    verify_proof,
)
from tollbooth.proven_npub import ProvenNpub, ProvenNpubCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_KEY = PrivateKey()
VALID_NPUB = _TEST_KEY.public_key.bech32()
_TEST_NSEC = _TEST_KEY.bech32()


def _make_ownership_proof(nsec: str = _TEST_NSEC) -> str:
    """Create a valid ownership proof for the test keypair."""
    return create_ownership_proof(nsec)


def _make_expired_proof() -> str:
    """Create an ownership proof that is already stale (>60s old)."""
    pk = _TEST_KEY
    event = Event(
        pubkey=pk.public_key.hex(),
        kind=PROOF_EVENT_KIND,
        content="",
        created_at=int(time.time()) - 120,  # 2 minutes ago
        tags=[["u", OWNERSHIP_SENTINEL]],
    )
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


def _make_wrong_sentinel_proof() -> str:
    """Create a proof with a tool name instead of ownership sentinel."""
    pk = _TEST_KEY
    event = Event(
        pubkey=pk.public_key.hex(),
        kind=PROOF_EVENT_KIND,
        content="",
        created_at=int(time.time()),
        tags=[["u", "get_weather"]],
    )
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


# ---------------------------------------------------------------------------
# Fake runtime for testing (no Neon, no vault)
# ---------------------------------------------------------------------------

class FakeRuntime:
    """Minimal runtime stub — no vault persistence."""

    async def load_patron_session(self, npub, *, service=None):
        return None  # no vault in tests

    async def store_patron_session(self, npub, vault_data, *, service=None):
        pass  # no vault in tests


# ---------------------------------------------------------------------------
# Tests: create_ownership_proof
# ---------------------------------------------------------------------------

def test_create_ownership_proof_valid():
    proof_json = create_ownership_proof(_TEST_NSEC)
    assert verify_proof(proof_json, VALID_NPUB, OWNERSHIP_SENTINEL)


def test_create_ownership_proof_has_sentinel_tag():
    proof_json = create_ownership_proof(_TEST_NSEC)
    event_dict = json.loads(proof_json)
    u_values = [t[1] for t in event_dict["tags"] if t[0] == "u"]
    assert OWNERSHIP_SENTINEL in u_values


# ---------------------------------------------------------------------------
# Tests: ProvenNpubCache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prove_and_is_proven():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    proof = _make_ownership_proof()

    record = await cache.prove(VALID_NPUB, proof)
    assert isinstance(record, ProvenNpub)
    assert record.npub == VALID_NPUB
    assert await cache.is_proven(VALID_NPUB)


@pytest.mark.asyncio
async def test_is_proven_false_initially():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    assert not await cache.is_proven(VALID_NPUB)


@pytest.mark.asyncio
async def test_prove_rejects_expired_proof():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    with pytest.raises(ValueError, match="Invalid ownership proof"):
        await cache.prove(VALID_NPUB, _make_expired_proof())


@pytest.mark.asyncio
async def test_prove_rejects_wrong_sentinel():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    with pytest.raises(ValueError, match="Invalid ownership proof"):
        await cache.prove(VALID_NPUB, _make_wrong_sentinel_proof())


@pytest.mark.asyncio
async def test_prove_rejects_wrong_npub():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    other_key = PrivateKey()
    other_npub = other_key.public_key.bech32()
    proof = _make_ownership_proof()  # signed by _TEST_KEY
    with pytest.raises(ValueError, match="Invalid ownership proof"):
        await cache.prove(other_npub, proof)


@pytest.mark.asyncio
async def test_invalidate_clears_cache():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    proof = _make_ownership_proof()
    await cache.prove(VALID_NPUB, proof)
    assert await cache.is_proven(VALID_NPUB)

    cache.invalidate(VALID_NPUB)
    # Without vault, invalidation removes from memory
    assert not await cache.is_proven(VALID_NPUB)


@pytest.mark.asyncio
async def test_expiry_removes_proven():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=1)
    proof = _make_ownership_proof()
    await cache.prove(VALID_NPUB, proof)
    assert await cache.is_proven(VALID_NPUB)

    # Manually expire by tweaking the cache entry
    entry = cache._cache._cache.get(VALID_NPUB)
    if entry is not None:
        # Force expiry by backdating the TTL entry
        cache._cache._cache._entries[VALID_NPUB] = (
            entry, time.time() - 10,
        )
    assert not await cache.is_proven(VALID_NPUB)
