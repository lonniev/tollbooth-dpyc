"""Tests for the proven npub ownership cache."""

import time

import pytest

from tollbooth.proven_npub import ProvenNpub, ProvenNpubCache


# ---------------------------------------------------------------------------
# Fake runtime for testing (no Neon, no vault)
# ---------------------------------------------------------------------------

class FakeRuntime:
    """Minimal runtime stub — no vault persistence."""

    async def load_patron_session(self, npub, *, service=None):
        return None

    async def store_patron_session(self, npub, vault_data, *, service=None):
        pass


VALID_NPUB = "npub1l94pd4qu4eszrl6ek032ftcnsu3tt9a7xvq2zp7eaxeklp6mrpzssmq8pf"


# ---------------------------------------------------------------------------
# Tests: ProvenNpubCache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mark_proven_and_is_proven():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    record = await cache.mark_proven(VALID_NPUB)
    assert isinstance(record, ProvenNpub)
    assert record.npub == VALID_NPUB
    assert await cache.is_proven(VALID_NPUB)


@pytest.mark.asyncio
async def test_is_proven_false_initially():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    assert not await cache.is_proven(VALID_NPUB)


@pytest.mark.asyncio
async def test_invalidate_clears_cache():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    await cache.mark_proven(VALID_NPUB)
    assert await cache.is_proven(VALID_NPUB)

    cache.invalidate(VALID_NPUB)
    assert not await cache.is_proven(VALID_NPUB)


@pytest.mark.asyncio
async def test_expiry_removes_proven():
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=1)
    await cache.mark_proven(VALID_NPUB)
    assert await cache.is_proven(VALID_NPUB)

    # Force expiry by backdating the TTL entry
    entry = cache._cache._cache._entries.get(VALID_NPUB)
    if entry is not None:
        cache._cache._cache._entries[VALID_NPUB] = (
            entry[0], time.time() - 10,
        )
    assert not await cache.is_proven(VALID_NPUB)


@pytest.mark.asyncio
async def test_record_has_no_proof_json():
    """ProvenNpub is DM-based — no proof_json field."""
    cache = ProvenNpubCache(runtime=FakeRuntime(), ttl_seconds=3600)
    record = await cache.mark_proven(VALID_NPUB)
    assert not hasattr(record, "proof_json")
    assert record.npub == VALID_NPUB
    assert record.expires_at > record.verified_at
