"""Tests for the dpop_token-keyed proven npub ownership cache."""

import hashlib
import time

import pytest

from tollbooth.proven_npub import (
    MAX_PROVEN_TTL,
    ProvenNpub,
    ProvenNpubCache,
    parse_duration,
)

VALID_NPUB = "npub1l94pd4qu4eszrl6ek032ftcnsu3tt9a7xvq2zp7eaxeklp6mrpzssmq8pf"
DPOP_TOKEN_A = "bold-hawk-42"
DPOP_TOKEN_B = "calm-reef-77"
HASH_A = hashlib.sha256(DPOP_TOKEN_A.encode()).hexdigest()
HASH_B = hashlib.sha256(DPOP_TOKEN_B.encode()).hexdigest()


@pytest.mark.asyncio
async def test_mark_proven_and_is_proven():
    cache = ProvenNpubCache(ttl_seconds=3600)
    record = await cache.mark_proven(HASH_A, VALID_NPUB)
    assert isinstance(record, ProvenNpub)
    assert record.dpop_token_hash == HASH_A
    assert record.npub == VALID_NPUB
    assert await cache.is_proven(HASH_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_is_proven_false_initially():
    cache = ProvenNpubCache(ttl_seconds=3600)
    assert not await cache.is_proven(HASH_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_different_dpop_token_not_proven():
    """Proof with dpop_token A does not extend to dpop_token B."""
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
            dpop_token_hash=entry[0].dpop_token_hash,
            npub=entry[0].npub,
            verified_at=entry[0].verified_at,
            expires_at=time.time() - 10,
        )
        cache._cache._entries[key] = (expired_record, entry[1])
    assert not await cache.is_proven(HASH_A, VALID_NPUB)


@pytest.mark.asyncio
async def test_record_has_dpop_token_hash():
    cache = ProvenNpubCache(ttl_seconds=3600)
    record = await cache.mark_proven(HASH_A, VALID_NPUB)
    assert record.dpop_token_hash == HASH_A
    assert record.npub == VALID_NPUB
    assert record.expires_at > record.verified_at


@pytest.mark.asyncio
async def test_json_round_trip():
    record = ProvenNpub(
        dpop_token_hash=HASH_A,
        npub=VALID_NPUB,
        verified_at=time.time(),
        expires_at=time.time() + 3600,
    )
    restored = ProvenNpub.from_json(record.to_json())
    assert restored == record


@pytest.mark.asyncio
async def test_proof_status_unknown_when_no_record():
    cache = ProvenNpubCache(ttl_seconds=3600)
    info = await cache.proof_status(HASH_A, VALID_NPUB)
    assert info["status"] == "unknown"
    assert info["expires_in_seconds"] == 0


@pytest.mark.asyncio
async def test_proof_status_valid_returns_remaining_ttl():
    cache = ProvenNpubCache(ttl_seconds=3600)
    await cache.mark_proven(HASH_A, VALID_NPUB)
    info = await cache.proof_status(HASH_A, VALID_NPUB)
    assert info["status"] == "valid"
    # Runtime-derived — within a small epsilon of the configured TTL
    assert 3590 < info["expires_in_seconds"] <= 3600


@pytest.mark.asyncio
async def test_proof_status_expired_does_not_evict():
    """proof_status is read-only — must not mutate cache state on expiry."""
    cache = ProvenNpubCache(ttl_seconds=3600)
    await cache.mark_proven(HASH_A, VALID_NPUB)

    # Backdate the record's expires_at without going through is_proven
    key = f"{HASH_A}:{VALID_NPUB}"
    entry = cache._cache._entries.get(key)
    assert entry is not None
    expired_record = ProvenNpub(
        dpop_token_hash=entry[0].dpop_token_hash,
        npub=entry[0].npub,
        verified_at=entry[0].verified_at,
        expires_at=time.time() - 10,
    )
    cache._cache._entries[key] = (expired_record, entry[1])

    info = await cache.proof_status(HASH_A, VALID_NPUB)
    assert info["status"] == "expired"
    assert info["expires_in_seconds"] == 0
    # Record must still be in the cache — proof_status is read-only
    assert cache._cache._entries.get(key) is not None


# ---------------------------------------------------------------------------
# Delegation cap — patrons may choose their own duration up to 30 days.
# ---------------------------------------------------------------------------


def test_cap_is_thirty_days():
    assert MAX_PROVEN_TTL == 2592000


def test_parse_duration_honors_thirty_days():
    """A 30-day delegation sits exactly at the cap and is honored verbatim."""
    assert parse_duration("30 days") == MAX_PROVEN_TTL


def test_parse_duration_clamps_above_cap():
    """Durations beyond the cap clamp down rather than erroring."""
    assert parse_duration("60 days") == MAX_PROVEN_TTL
    assert parse_duration("10 weeks") == MAX_PROVEN_TTL


def test_parse_duration_under_cap_unchanged():
    """A sub-cap duration (e.g. a multi-day editorial session) is exact."""
    assert parse_duration("7 days") == 7 * 86400


@pytest.mark.asyncio
async def test_mark_proven_clamps_ttl_override_to_cap():
    cache = ProvenNpubCache(ttl_seconds=3600)
    before = time.time()
    record = await cache.mark_proven(HASH_A, VALID_NPUB, ttl_override=MAX_PROVEN_TTL * 5)
    # Clamped: expiry lands at ~now + cap, not now + 5×cap.
    assert record.expires_at - before == pytest.approx(MAX_PROVEN_TTL, abs=5)
