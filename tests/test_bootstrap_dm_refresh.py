"""Weekly bootstrap republish must finish, and must say so when it does not.

Field report (2026-09-27): ``bootstrap_dm_sent_at`` sat at 2026-06-22 for three
months despite certify_credits traffic that should have refreshed it weekly.
Call sites fired ``asyncio.create_task(_maybe_refresh_bootstrap_dm(...))`` and
dropped the handle — an unreferenced task is eligible for GC mid-flight, and a
serverless host idles the process the moment the response returns. The throttle
stamp was also set before the vault read, so a swallowed exception silenced the
npub for an hour at DEBUG only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tollbooth.authority.tools as at


@pytest.fixture(autouse=True)
def _reset_throttle():
    at._bootstrap_dm_last_check.clear()
    at._bootstrap_dm_tasks.clear()
    yield
    at._bootstrap_dm_last_check.clear()
    at._bootstrap_dm_tasks.clear()


class TestRefreshTaskIsRetained:
    @pytest.mark.asyncio
    async def test_schedule_keeps_a_strong_ref_until_done(self):
        gate = asyncio.Event()

        async def slow(_npub: str) -> None:
            await gate.wait()

        with patch.object(at, "_maybe_refresh_bootstrap_dm", side_effect=slow):
            at._schedule_bootstrap_dm_refresh("npub1abc")
            assert len(at._bootstrap_dm_tasks) == 1
            gate.set()
            await asyncio.gather(*list(at._bootstrap_dm_tasks), return_exceptions=True)
            await asyncio.sleep(0)
            assert len(at._bootstrap_dm_tasks) == 0


class TestRefreshLoggingAndThrottle:
    @pytest.mark.asyncio
    async def test_due_refresh_that_fails_to_publish_logs_warning(self, caplog):
        """A due stamp + failed publish must not vanish at DEBUG."""
        vault = MagicMock()
        runtime = MagicMock()
        runtime.vault = AsyncMock(return_value=vault)
        npub = "npub1operatorxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        with patch.object(at, "_get_runtime", return_value=runtime), \
             patch(
                 "tollbooth.authority.tenant_provisioner.get_operator_config_value",
                 new=AsyncMock(return_value="1"),  # epoch → ancient
             ), \
             patch.object(at, "_resend_bootstrap_dm", new=AsyncMock(return_value=False)), \
             caplog.at_level(logging.WARNING):
            await at._maybe_refresh_bootstrap_dm(npub)

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "did not publish" in m or "refresh" in m.lower() for m in messages
        ), messages

    @pytest.mark.asyncio
    async def test_vault_read_failure_does_not_arm_the_hour_throttle(self):
        """A swallowed exception must not silence the npub for an hour."""
        runtime = MagicMock()
        runtime.vault = AsyncMock(side_effect=RuntimeError("vault down"))
        npub = "npub1operatorxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        with patch.object(at, "_get_runtime", return_value=runtime):
            await at._maybe_refresh_bootstrap_dm(npub)

        assert npub not in at._bootstrap_dm_last_check

    @pytest.mark.asyncio
    async def test_successful_check_arms_the_throttle(self):
        vault = MagicMock()
        runtime = MagicMock()
        runtime.vault = AsyncMock(return_value=vault)
        fresh = str(int(time.time()))
        npub = "npub1operatorxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        import tollbooth.authority.tenant_provisioner as tp

        with patch.object(at, "_get_runtime", return_value=runtime), \
             patch.object(
                 tp, "get_operator_config_value",
                 new=AsyncMock(return_value=fresh),
             ), \
             patch.object(at, "_resend_bootstrap_dm", new=AsyncMock()) as resend:
            await at._maybe_refresh_bootstrap_dm(npub)
            resend.assert_not_awaited()
            assert npub in at._bootstrap_dm_last_check

    @pytest.mark.asyncio
    async def test_first_check_is_not_throttled_on_a_young_host(self):
        """Missing-key must not act like last_check=0 (monotonic-since-boot trap)."""
        vault = MagicMock()
        runtime = MagicMock()
        runtime.vault = AsyncMock(return_value=vault)
        npub = "npub1operatorxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        called = AsyncMock(return_value=str(int(time.time())))

        import tollbooth.authority.tenant_provisioner as tp

        with patch.object(at, "_get_runtime", return_value=runtime), \
             patch.object(tp, "get_operator_config_value", new=called), \
             patch.object(at, "_resend_bootstrap_dm", new=AsyncMock()), \
             patch.object(at.time, "monotonic", return_value=60.0):  # uptime 1 min
            await at._maybe_refresh_bootstrap_dm(npub)

        called.assert_awaited()

    @pytest.mark.asyncio
    async def test_resend_logs_accepted_and_rejected_counts(self, caplog):
        vault = MagicMock()
        runtime = MagicMock()
        runtime.vault = AsyncMock(return_value=vault)
        signer = MagicMock()
        signer.nsec = "a" * 64

        with patch.object(at, "_get_runtime", return_value=runtime), \
             patch.object(at, "_get_nostr_signer", return_value=signer), \
             patch(
                 "tollbooth.authority.tenant_provisioner.get_all_operator_config",
                 new=AsyncMock(return_value={
                     "neon_database_url": "postgres://x",
                     "schema": "op_x",
                 }),
             ), \
             patch(
                 "tollbooth.authority.tenant_provisioner.store_operator_config",
                 new=AsyncMock(),
             ), \
             patch.object(
                 at.asyncio, "to_thread",
                 new=AsyncMock(return_value=_PublishResult(3, 2)),
             ), \
             caplog.at_level(logging.INFO, logger=at.logger.name):
            ok = await at._resend_bootstrap_dm(
                "npub1operatorxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            )

        assert ok is True
        joined = " ".join(r.message for r in caplog.records)
        assert "3" in joined and ("accepted" in joined.lower() or "relay" in joined.lower())


class _PublishResult:
    """Stand-in for bootstrap_relay.PublishResult without importing the real one."""

    def __init__(self, accepted: int, rejected: int) -> None:
        self.accepted = accepted
        self.rejected = rejected

    def __bool__(self) -> bool:
        return self.accepted > 0
