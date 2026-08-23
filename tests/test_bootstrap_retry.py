"""A relay that is down for a moment must not become a permanent verdict.

Two failures compound here, and both were live on 2026-08-23.

The relay poll ran ONCE, so a flap lasting seconds read as "this operator has
no bootstrap config". The two relays carrying one operator's config both
refused inside the same window and were serving again about 110 seconds later;
a live drill was thrown away in between.

And `ensure_bootstrapped` memoises for the whole process, so that momentary
verdict was cached — pinning a front to broken until it recycled, and every
later tool call to the same stale answer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import tollbooth.bootstrap as bs
from tollbooth.bootstrap import BootstrapClient, BootstrapResult, ensure_bootstrapped

# Captured at import, BEFORE the fixture patches it — reloading the module to
# read it back would undo every patch and let later tests hit real relays.
REAL_BACKOFF = bs._BOOTSTRAP_RETRY_BACKOFF

NSEC_HEX = "a" * 64
CONFIG = {"neon_database_url": "postgres://example/db"}


@pytest.fixture(autouse=True)
def _no_cache_no_sleep(monkeypatch):
    """Isolate the module global, and never actually wait."""
    bs._cached_result = None
    monkeypatch.setattr(bs, "_BOOTSTRAP_RETRY_BACKOFF", (0, 0, 0, 0))
    monkeypatch.setattr(bs.asyncio, "sleep", AsyncMock())
    yield
    bs._cached_result = None


def _client():
    c = BootstrapClient(nsec_hex=NSEC_HEX)
    c._npub, c._pubkey_hex = "npub1test", "b" * 64
    return c


def _oracle():
    o = AsyncMock()
    o.get_relays = AsyncMock(return_value=["wss://a", "wss://b"])
    o.resolve_authority_for = AsyncMock(return_value=None)
    return o


class TestTheRelayPollIsRetried:
    @pytest.mark.asyncio
    async def test_a_flap_on_the_first_pass_does_not_end_it(self):
        """The 2026-08-23 shape: down, then serving moments later."""
        polls = [(None, None, "relays=2, events=0"),
                 (None, None, "relays=2, events=0"),
                 (CONFIG, "c" * 64, "relays=2, events=1")]
        with patch("tollbooth.bootstrap_relay.receive_bootstrap_config",
                   side_effect=polls) as poll, \
             patch("tollbooth.oracle_client.default_oracle_client", return_value=_oracle()):
            result = await _client().bootstrap()

        assert result.success is True
        assert result.neon_database_url == "postgres://example/db"
        assert poll.call_count == 3, "must keep trying while attempts remain"

    @pytest.mark.asyncio
    async def test_a_first_pass_hit_costs_no_extra_polls(self):
        """The common case must not pay for the retry ladder."""
        with patch("tollbooth.bootstrap_relay.receive_bootstrap_config",
                   return_value=(CONFIG, "c" * 64, "ok")) as poll, \
             patch("tollbooth.oracle_client.default_oracle_client", return_value=_oracle()):
            result = await _client().bootstrap()
        assert result.success is True and poll.call_count == 1

    @pytest.mark.asyncio
    async def test_the_ladder_is_bounded_and_the_failure_is_marked_transient(self):
        with patch("tollbooth.bootstrap_relay.receive_bootstrap_config",
                   return_value=(None, None, "relays=2, events=0")) as poll, \
             patch("tollbooth.oracle_client.default_oracle_client", return_value=_oracle()):
            result = await _client().bootstrap()

        assert result.success is False
        assert result.transient is True, "reachability is a moment, not a fact"
        assert poll.call_count == len(bs._BOOTSTRAP_RETRY_BACKOFF)

    def test_the_shipped_ladder_covers_a_short_outage_without_being_silly(self):
        """Sized for seconds of flap, not for an outage a human should hear about."""
        assert 45 <= sum(REAL_BACKOFF) <= 180, f"ladder sums to {sum(REAL_BACKOFF)}s"
        assert REAL_BACKOFF[-1] == 0, "the last attempt must not wait afterwards"


class TestWhatGetsCached:
    @pytest.mark.asyncio
    async def test_a_transient_failure_is_not_cached(self):
        """Caching it pins the whole process to a second of bad weather."""
        transient = BootstrapResult(error="No bootstrap config on relays", transient=True)
        with patch.dict("os.environ", {"TOLLBOOTH_NOSTR_OPERATOR_NSEC": NSEC_HEX}), \
             patch.object(BootstrapClient, "bootstrap", AsyncMock(return_value=transient)):
            await ensure_bootstrapped()
        assert bs._cached_result is None, "the next call must be free to retry"

    @pytest.mark.asyncio
    async def test_the_next_call_really_does_retry_and_can_succeed(self):
        good = BootstrapResult(success=True, neon_database_url="postgres://x")
        bad = BootstrapResult(error="No bootstrap config on relays", transient=True)
        boot = AsyncMock(side_effect=[bad, good])
        with patch.dict("os.environ", {"TOLLBOOTH_NOSTR_OPERATOR_NSEC": NSEC_HEX}), \
             patch.object(BootstrapClient, "bootstrap", boot):
            first = await ensure_bootstrapped()
            second = await ensure_bootstrapped()
        assert first.success is False and second.success is True
        assert boot.await_count == 2

    @pytest.mark.asyncio
    async def test_success_is_cached(self):
        good = BootstrapResult(success=True, neon_database_url="postgres://x")
        boot = AsyncMock(return_value=good)
        with patch.dict("os.environ", {"TOLLBOOTH_NOSTR_OPERATOR_NSEC": NSEC_HEX}), \
             patch.object(BootstrapClient, "bootstrap", boot):
            await ensure_bootstrapped()
            await ensure_bootstrapped()
        assert boot.await_count == 1, "success must be memoised as before"

    @pytest.mark.asyncio
    async def test_a_definitive_failure_is_still_cached(self):
        """No nsec is a fact about the deployment; retrying cannot help."""
        with patch.dict("os.environ", {"TOLLBOOTH_NOSTR_OPERATOR_NSEC": ""}, clear=False):
            r1 = await ensure_bootstrapped()
            r2 = await ensure_bootstrapped()
        assert r1.success is False and "not set" in r1.error
        assert r1 is r2, "a definitive verdict should not be recomputed"
