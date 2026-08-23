"""Recovering an orphaned job must use the DETACHED executor, not this process.

The whole reason detached execution exists is that a container recycle kills an
in-flight job. The recovery path is what picks such a job back up — so if
recovery runs the retry locally, the retry is orphanable by the very next
recycle and the detached runner was never in the loop at all.

That is what happened on 2026-08-23. `_ensure_async_executor` is called only
from `start_async_job`, so a process reached first by a POLL — exactly what a
fresh container sees after a recycle — still held the InProcessExecutor
default. The recovery submitted into it, and a live drill that had been
dispatched to Modal finished on the Horizon front instead.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tollbooth.async_executor import InProcessExecutor


class _Store:
    """Just enough job store for the recovery branch."""

    def __init__(self, row):
        self._row = row
        self.handles = []

    async def get(self, claim, npub=None):
        return self._row

    async def set_run_handle(self, claim, handle):
        self.handles.append(handle)

    async def fail(self, claim, err):  # pragma: no cover - not this path
        raise AssertionError("should not fail the job")


def _stalled_row():
    return {
        "status": "running", "stalled": True, "expired": False, "result": None, "error": None,
        "run_handle": None, "tool_id": "t", "npub": "npub1pmywygyld7vprgda3v8sd3lxnzuqceyt6fpsr0qatwkl8xdzzg4s4zhndw", "params": {},
        "elapsed_seconds": 800, "max_runtime_seconds": 700, "expected_seconds": 480,
    }


@pytest.mark.asyncio
async def test_recovery_resolves_the_executor_before_submitting(monkeypatch):
    """A poll-first process must probe, not fall through to in-process."""
    from tollbooth.runtime import OperatorRuntime

    rt = OperatorRuntime.__new__(OperatorRuntime)
    rt._async_executor = InProcessExecutor(rt)      # the default a fresh process holds
    rt._async_executor_explicit = False
    rt._async_executor_resolved = False
    rt._async_dispatch_error = None
    # __new__ skips __init__; suppress the opportunistic purge that runs after
    # the branch under test by making it look freshly rate-limited.
    import time as _t
    rt._async_jobs_purge_last = _t.monotonic()

    detached = AsyncMock()
    detached.submit = AsyncMock(return_value="fc-detached-handle")

    async def _resolve():
        # What the real probe does when Modal creds are vaulted.
        rt._async_executor = detached
        rt._async_executor_resolved = True

    with patch.object(OperatorRuntime, "_ensure_async_executor", side_effect=_resolve):
        store = _Store(_stalled_row())
        with patch.object(OperatorRuntime, "async_job_store",
                          AsyncMock(return_value=store)):
            resp = await OperatorRuntime.fetch_async_job(rt, "claim-1", "npub1pmywygyld7vprgda3v8sd3lxnzuqceyt6fpsr0qatwkl8xdzzg4s4zhndw")

    detached.submit.assert_awaited_once_with("claim-1")
    assert store.handles == ["fc-detached-handle"], (
        "the recovery must record the DETACHED handle; an in-process retry "
        "records nothing and is orphanable by the next recycle"
    )
    assert resp["status"] == "running" and resp.get("recovered") is True
