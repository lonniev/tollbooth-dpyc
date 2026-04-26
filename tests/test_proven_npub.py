"""Tests for the poison-keyed proven npub ownership cache."""

import hashlib
import time

import pytest

from tollbooth.proven_npub import ProvenNpub, ProvenNpubCache


VALID_NPUB = "npub1l94pd4qu4eszrl6ek032ftcnsu3tt9a7xvq2zp7eaxeklp6mrpzssmq8pf"
POISON_A = "bold-hawk-42"
POISON_B = "calm-reef-77"
HASH_A = hashlib.sha256(POISON_A.encode()).hexdigest()
HASH_B = hashlib.sha256(POISON_B.encode()).hexdigest()


@pytest.mark.asyncio
async def test_mark_proven_and_is_proven():
    cache = ProvenNpubCache(ttl_seconds=3600)
    record = await cache.mark_proven(HASH_A, VALID_NPUB)
    assert isinstance(record, ProvenNpub)
    assert record.poison_hash == HASH_A
    assert record.npub == VALID_NPUB
    assert await cache.is_proven(HASH_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_is_proven_false_initially():
    cache = ProvenNpubCache(ttl_seconds=3600)
    assert not await cache.is_proven(HASH_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_different_poison_not_proven():
    """Proof with poison A does not extend to poison B."""
    cache = ProvenNpubCache(ttl_seconds=3600)
    await cache.mark_proven(HASH_A, VALID_NPUB)
    assert await cache.is_proven(HASH_A, VALID_NPUB)
    assert not await cache.is_proven(HASH_B, VALID_NPUB)


@pytest.mark.asyncio
async def test_invalidate_clears_cache():
    cache = ProvenNpubCache(ttl_seconds=3600)
    await cache.mark_proven(HASH_A, VALID_NPUB)
    cache.invalidate(HASH_A, VALID_NPUB)
    assert not await cache.is_proven(HASH_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_expiry_removes_proven():
    cache = ProvenNpubCache(ttl_seconds=1)
    await cache.mark_proven(HASH_A, VALID_NPUB)
    assert await cache.is_proven(HASH_A, VALID_NPUB)

    # Backdate the ProvenNpub record's expires_at to force expiry.
    key = f"{HASH_A}:{VALID_NPUB}"
    entry = cache._cache._entries.get(key)
    if entry is not None:
        expired_record = ProvenNpub(
            poison_hash=entry[0].poison_hash,
            npub=entry[0].npub,
            verified_at=entry[0].verified_at,
            expires_at=time.time() - 10,
        )
        cache._cache._entries[key] = (expired_record, entry[1])
    assert not await cache.is_proven(HASH_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_record_has_poison_hash():
    cache = ProvenNpubCache(ttl_seconds=3600)
    record = await cache.mark_proven(HASH_A, VALID_NPUB)
    assert record.poison_hash == HASH_A
    assert record.npub == VALID_NPUB
    assert record.expires_at > record.verified_at


@pytest.mark.asyncio
async def test_json_round_trip():
    record = ProvenNpub(
        poison_hash=HASH_A,
        npub=VALID_NPUB,
        verified_at=time.time(),
        expires_at=time.time() + 3600,
    )
    restored = ProvenNpub.from_json(record.to_json())
    assert restored == record
