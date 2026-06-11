"""Tests for identity_proof.require_proof — the canonical proof gate.

Focus: the S4 fix (a poison-shaped token that can't be validated as a cached
proof gets clear "refresh" feedback instead of a confusing malformed-Schnorr
error) plus the surrounding accept/deny paths.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tollbooth.constants import ErrorCode
from tollbooth.identity_proof import require_proof

NPUB = "npub1d999638gqpn8c594teklxtxva0uvxdng80q3ycyqvldjdl457c7qcrq64z"
POISON = "glad-blade-13"  # matches the <word>-<word>-<n> poison shape


@pytest.mark.asyncio
async def test_empty_proof_required():
    r = await require_proof(NPUB, "", "tool")
    assert r["error_code"] == ErrorCode.PROOF_REQUIRED


@pytest.mark.asyncio
async def test_invalid_npub_rejected():
    r = await require_proof("npub1short", POISON, "tool")
    assert r["error_code"] == ErrorCode.NPUB_INVALID


@pytest.mark.asyncio
async def test_poison_shaped_token_without_cache_gives_refresh_feedback():
    # S4: no proven_cache wired, but the token looks like a proof_token. Must
    # not fall through to "Invalid identity proof" — give refresh guidance.
    r = await require_proof(NPUB, POISON, "tool", proven_cache=None)
    assert r["error_code"] == ErrorCode.PROOF_REFRESH_NEEDED
    assert "proof_token" in r["error"]
    assert r["next_steps"]


@pytest.mark.asyncio
async def test_cached_poison_hit_passes():
    cache = SimpleNamespace(is_proven=AsyncMock(return_value=True))
    r = await require_proof(NPUB, POISON, "tool", proven_cache=cache)
    assert r is None  # success → caller proceeds


@pytest.mark.asyncio
async def test_cached_poison_miss_gives_refresh_feedback():
    cache = SimpleNamespace(is_proven=AsyncMock(return_value=False))
    r = await require_proof(NPUB, POISON, "tool", proven_cache=cache)
    assert r["error_code"] == ErrorCode.PROOF_REFRESH_NEEDED


@pytest.mark.asyncio
async def test_non_poison_garbage_is_invalid_proof():
    # A non-poison, non-Schnorr string still fails as a malformed inline proof.
    r = await require_proof(NPUB, "garbage", "tool")
    assert r["error_code"] == ErrorCode.PROOF_INVALID
