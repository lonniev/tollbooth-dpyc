"""Proof-grant revocation — the per-npub watermark (#267).

A grant is self-verifying; the only thing the gate still reads from storage
is whether the patron revoked. These tests pin: the vault write is
authoritative, reads are cached per npub for ``refresh_seconds``, an
unreadable vault raises (the gate fails closed), and the patron's stated
consent window is clamped to the cap.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock

import pytest

from tollbooth.proven_npub import (
    DEFAULT_PROVEN_TTL,
    MAX_PROVEN_TTL,
    ProofGrantRevocations,
    RevocationStoreUnavailable,
    grant_ttl_seconds,
    parse_duration,
)

NPUB = "npub1" + "q" * 58


class FakeVault:
    """set_config/get_config over a dict; encryption is identity for the test."""

    def __init__(self) -> None:
        self.rows: dict[str, str] = {}
        self.set_config = AsyncMock(side_effect=self._set)
        self.get_config = AsyncMock(side_effect=self._get)

    async def _set(self, key: str, value: str) -> None:
        self.rows[key] = value

    async def _get(self, key: str) -> str | None:
        return self.rows.get(key)

    def _encrypt(self, s: str) -> str:
        return s

    def _decrypt(self, s: str) -> str:
        return s


# ---------------------------------------------------------------------------
# watermark
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_never_revoked_is_none_and_not_revoked():
    rev = ProofGrantRevocations()
    assert await rev.revoked_before(NPUB) is None
    assert not await rev.is_revoked(NPUB, int(time.time()))


@pytest.mark.asyncio
async def test_revoke_all_refuses_grants_up_to_now_and_admits_later_ones():
    rev = ProofGrantRevocations()
    mark = await rev.revoke_all(NPUB)
    assert await rev.is_revoked(NPUB, mark)          # same second: refused
    assert await rev.is_revoked(NPUB, mark - 3600)   # earlier: refused
    assert not await rev.is_revoked(NPUB, mark + 1)  # minted after: admitted


@pytest.mark.asyncio
async def test_revoke_all_writes_the_vault_authoritatively():
    vault = FakeVault()
    rev = ProofGrantRevocations(vault=vault)
    mark = await rev.revoke_all(NPUB)
    key = f"proof_grant_revoked_before:{NPUB}"
    assert json.loads(vault.rows[key]) == {"npub": NPUB, "revoked_before": mark}


@pytest.mark.asyncio
async def test_revoke_all_raises_when_the_vault_write_fails():
    vault = FakeVault()
    vault.set_config = AsyncMock(side_effect=RuntimeError("neon down"))
    rev = ProofGrantRevocations(vault=vault)
    with pytest.raises(RuntimeError):
        await rev.revoke_all(NPUB)
    # Nothing was recorded in memory either — no half-revoke.
    assert await rev.revoked_before(NPUB) is None


@pytest.mark.asyncio
async def test_cold_start_reads_the_watermark_from_the_vault():
    vault = FakeVault()
    await ProofGrantRevocations(vault=vault).revoke_all(NPUB)
    fresh = ProofGrantRevocations(vault=vault)  # new process, empty memory
    assert await fresh.is_revoked(NPUB, int(time.time()) - 10)


@pytest.mark.asyncio
async def test_reads_are_served_from_memory_within_refresh_window():
    vault = FakeVault()
    rev = ProofGrantRevocations(vault=vault, refresh_seconds=60)
    await rev.revoked_before(NPUB)
    await rev.revoked_before(NPUB)
    await rev.is_revoked(NPUB, 1)
    assert vault.get_config.await_count == 1


@pytest.mark.asyncio
async def test_reads_go_back_to_the_vault_after_refresh_window():
    vault = FakeVault()
    rev = ProofGrantRevocations(vault=vault, refresh_seconds=0.0)
    await rev.revoked_before(NPUB)
    await rev.revoked_before(NPUB)
    assert vault.get_config.await_count == 2


@pytest.mark.asyncio
async def test_unreadable_vault_raises_so_the_gate_fails_closed():
    vault = FakeVault()
    vault.get_config = AsyncMock(side_effect=RuntimeError("neon down"))
    rev = ProofGrantRevocations(vault=vault)
    with pytest.raises(RevocationStoreUnavailable):
        await rev.is_revoked(NPUB, int(time.time()))


# ---------------------------------------------------------------------------
# the patron's consent window
# ---------------------------------------------------------------------------


def test_cap_is_thirty_days():
    assert MAX_PROVEN_TTL == 30 * 24 * 3600


@pytest.mark.parametrize("text,seconds", [
    ("2h", 7200), ("two days", 172800), ("  30  min ", 1800),
    ("1w", 604800), ("30 days", MAX_PROVEN_TTL),
])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_parse_duration_unlimited_is_none():
    assert parse_duration("forever") is None


def test_parse_duration_clamps_above_cap():
    assert parse_duration("45 days") == MAX_PROVEN_TTL


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("soonish")


@pytest.mark.parametrize("raw,expected", [
    ("", DEFAULT_PROVEN_TTL),
    ("   ", DEFAULT_PROVEN_TTL),
    ("garbage", DEFAULT_PROVEN_TTL),  # a typo must not strand the patron
    ("2h", 7200),
    ("forever", MAX_PROVEN_TTL),
    ("90 days", MAX_PROVEN_TTL),
])
def test_grant_ttl_seconds(raw, expected):
    assert grant_ttl_seconds(raw) == expected
