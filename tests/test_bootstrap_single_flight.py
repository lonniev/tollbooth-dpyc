"""Cold bootstrap must single-flight, and a front must not pay the long ladder.

Field report (2026-09-27, eXcalibur on Horizon): four concurrent first calls on a
cold process each built a BootstrapClient and each walked the full
``_BOOTSTRAP_RETRY_BACKOFF`` (~80 s) before failing with the same miss. One
in-flight read must serve every waiter; a front may try at most twice and answer
quickly, because the miss is not cached and the next call retries.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import tollbooth.bootstrap as bs
from tollbooth.bootstrap import (
    FRONT_BOOTSTRAP_RETRY_BACKOFF,
    BootstrapClient,
    BootstrapResult,
    ensure_bootstrapped,
)

NSEC_HEX = "a" * 64
CONFIG = {"neon_database_url": "postgres://example/db"}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    bs._cached_result = None
    bs._bootstrap_lock = None
    monkeypatch.setattr(bs.asyncio, "sleep", AsyncMock())
    yield
    bs._cached_result = None
    bs._bootstrap_lock = None


class TestSingleFlight:
    @pytest.mark.asyncio
    async def test_concurrent_first_calls_share_one_bootstrap(self):
        """Four waiters, one read — the 2026-09-27 shape.

        Under the lock the second waiter enters only after the first finishes
        (and has cached). All four therefore see exactly one ``bootstrap()``.
        """
        calls = 0

        async def slow_boot(self, *a, **kw):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            return BootstrapResult(success=True, neon_database_url="postgres://x")

        with patch.dict("os.environ", {"TOLLBOOTH_NOSTR_OPERATOR_NSEC": NSEC_HEX}), \
             patch.object(BootstrapClient, "bootstrap", slow_boot):
            results = await asyncio.gather(
                *[ensure_bootstrapped() for _ in range(4)]
            )

        assert calls == 1
        assert all(r.success for r in results)
        assert all(r is results[0] for r in results), "waiters share the same result object"

    @pytest.mark.asyncio
    async def test_a_cached_success_still_skips_the_lock_path(self):
        good = BootstrapResult(success=True, neon_database_url="postgres://x")
        boot = AsyncMock(return_value=good)
        with patch.dict("os.environ", {"TOLLBOOTH_NOSTR_OPERATOR_NSEC": NSEC_HEX}), \
             patch.object(BootstrapClient, "bootstrap", boot):
            await ensure_bootstrapped()
            await ensure_bootstrapped()
        assert boot.await_count == 1


class TestFrontLadder:
    def test_front_ladder_is_short(self):
        """A live agent front answers within ~10s of weather, not ~80s."""
        assert sum(FRONT_BOOTSTRAP_RETRY_BACKOFF) <= 10
        assert FRONT_BOOTSTRAP_RETRY_BACKOFF[-1] == 0
        assert len(FRONT_BOOTSTRAP_RETRY_BACKOFF) <= 3

    @pytest.mark.asyncio
    async def test_ensure_bootstrapped_uses_the_front_ladder_by_default(self):
        """Detached runners pass the long ladder; fronts get the short one."""
        seen: list = []

        async def capture_boot(self, *, retry_backoff=None):
            seen.append(retry_backoff)
            return BootstrapResult(success=True, neon_database_url="postgres://x")

        with patch.dict("os.environ", {"TOLLBOOTH_NOSTR_OPERATOR_NSEC": NSEC_HEX}), \
             patch.object(BootstrapClient, "bootstrap", capture_boot):
            await ensure_bootstrapped()

        assert seen and seen[0] == FRONT_BOOTSTRAP_RETRY_BACKOFF

    @pytest.mark.asyncio
    async def test_client_honours_an_injected_short_ladder(self):
        short = (0, 0)
        with patch("tollbooth.bootstrap_relay.receive_bootstrap_config",
                   return_value=(None, None, "relays=1, events=0")) as poll, \
             patch("tollbooth.oracle_client.default_oracle_client") as oracle:
            oracle.return_value = AsyncMock()
            oracle.return_value.get_relays = AsyncMock(return_value=["wss://a"])
            oracle.return_value.resolve_authority_for = AsyncMock(return_value=None)
            c = BootstrapClient(nsec_hex=NSEC_HEX)
            c._npub, c._pubkey_hex = "npub1test", "b" * 64
            result = await c.bootstrap(retry_backoff=short)

        assert result.transient is True
        assert poll.call_count == len(short)
