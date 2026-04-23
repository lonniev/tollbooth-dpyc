"""Tests for the channel-bound proven npub ownership cache."""

import time

import pytest

from tollbooth.proven_npub import ProvenNpub, ProvenNpubCache


VALID_NPUB = "npub1l94pd4qu4eszrl6ek032ftcnsu3tt9a7xvq2zp7eaxeklp6mrpzssmq8pf"
SESSION_A = "aaaa-aaaa-aaaa"
SESSION_B = "bbbb-bbbb-bbbb"


@pytest.mark.asyncio
async def test_mark_proven_and_is_proven():
    cache = ProvenNpubCache(ttl_seconds=3600)
    record = await cache.mark_proven(SESSION_A, VALID_NPUB)
    assert isinstance(record, ProvenNpub)
    assert record.session_id == SESSION_A
    assert record.npub == VALID_NPUB
    assert await cache.is_proven(SESSION_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_is_proven_false_initially():
    cache = ProvenNpubCache(ttl_seconds=3600)
    assert not await cache.is_proven(SESSION_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_different_session_not_proven():
    """Proof on session A does not extend to session B."""
    cache = ProvenNpubCache(ttl_seconds=3600)
    await cache.mark_proven(SESSION_A, VALID_NPUB)
    assert await cache.is_proven(SESSION_A, VALID_NPUB)
    assert not await cache.is_proven(SESSION_B, VALID_NPUB)


@pytest.mark.asyncio
async def test_invalidate_clears_cache():
    cache = ProvenNpubCache(ttl_seconds=3600)
    await cache.mark_proven(SESSION_A, VALID_NPUB)
    cache.invalidate(SESSION_A, VALID_NPUB)
    assert not await cache.is_proven(SESSION_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_expiry_removes_proven():
    cache = ProvenNpubCache(ttl_seconds=1)
    await cache.mark_proven(SESSION_A, VALID_NPUB)
    assert await cache.is_proven(SESSION_A, VALID_NPUB)

    # Backdate the ProvenNpub record's expires_at to force expiry.
    # SessionCache uses MAX_PROVEN_TTL as its container TTL; real expiry
    # is checked by is_proven() via the record's expires_at field.
    key = f"{SESSION_A}:{VALID_NPUB}"
    entry = cache._cache._entries.get(key)
    if entry is not None:
        expired_record = ProvenNpub(
            session_id=entry[0].session_id,
            npub=entry[0].npub,
            verified_at=entry[0].verified_at,
            expires_at=time.time() - 10,
        )
        cache._cache._entries[key] = (expired_record, entry[1])
    assert not await cache.is_proven(SESSION_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_record_has_session_id():
    cache = ProvenNpubCache(ttl_seconds=3600)
    record = await cache.mark_proven(SESSION_A, VALID_NPUB)
    assert record.session_id == SESSION_A
    assert record.npub == VALID_NPUB
    assert record.expires_at > record.verified_at
