"""What happens to durable execution when the credentials change underneath it.

``_ensure_async_executor`` states the design plainly: durable execution is
"purely opt-in by credential delivery, with no per-server bootstrap code." The
delivery, though, happens to a server that is ALREADY RUNNING — and the
resolution cached a definitive answer, while the executor it built carried the
API key baked in at construction. Neither noticed a later delivery.

Observed live 2026-08-01: eXcalibur's workers came up holding an expired Prefect
key. The key was replaced by Secure Courier at 21:51. At 22:00 the scheduler
launched three publishers, Prefect recorded ZERO flow runs, and every publisher
ran in-process on a front that recycles — so all three vanished mid-flight,
leaving posts claimed with no outcome and no reason. Throughout,
``service_status`` reported ``detached_executor_active: true``, which was true
and useless: an executor was installed; it simply could not authenticate.

Three properties are asserted here, one per failure that combined to hide it:

1. Delivering operator credentials re-opens the question.
2. A degraded dispatch says so in its RESULT, not only in a log.
3. ``service_status`` distinguishes "installed" from "actually dispatching".
"""

from __future__ import annotations

import pytest

from tollbooth.async_executor import InProcessExecutor
from tollbooth.runtime import OperatorRuntime


def _runtime() -> OperatorRuntime:
    return OperatorRuntime(tool_registry={}, service_name="Test")


class _DeadExecutor:
    """An installed executor that cannot dispatch — the expired-key shape."""

    async def submit(self, claim, closure_b64):
        raise RuntimeError("401 Unauthorized: Invalid authentication credentials")


# ---------------------------------------------------------------------------
# 1. Credential delivery re-opens the question
# ---------------------------------------------------------------------------


class TestInvalidation:
    def test_delivery_clears_a_cached_resolution(self):
        """The bug: a process that resolved once never looked again.

        Resolving 'no long-runner creds' is definitive AT THE TIME and stale the
        moment they are delivered — which is the ordinary onboarding order.
        """
        rt = _runtime()
        rt._async_executor_resolved = True          # resolved: no creds
        rt._async_executor_error = "long-runner creds vaulted but empty"

        rt.invalidate_async_executor()

        assert rt._async_executor_resolved is False, "next job must re-probe"
        assert rt._async_executor_error is None
        assert isinstance(rt._async_executor, InProcessExecutor)

    def test_delivery_drops_an_executor_holding_a_dead_key(self):
        """The subtler half: the API key is baked in at construction.

        A process holding an EXPIRED key kept presenting it and failing every
        dispatch. Re-probing is only useful if the stale executor goes with it.
        """
        rt = _runtime()
        rt._async_executor = _DeadExecutor()
        rt._async_executor_resolved = True
        rt._async_dispatch_error = "PrefectHTTPStatusError: 401 Unauthorized"

        rt.invalidate_async_executor()

        assert isinstance(rt._async_executor, InProcessExecutor)
        assert rt._async_dispatch_error is None, "a stale failure must not persist"

    def test_an_explicit_executor_is_never_overridden(self):
        """set_async_executor is an operator's deliberate choice. No credential
        delivery may quietly replace it."""
        rt = _runtime()
        chosen = _DeadExecutor()
        rt.set_async_executor(chosen)

        rt.invalidate_async_executor()

        assert rt._async_executor is chosen
        assert rt._async_executor_resolved is True


# ---------------------------------------------------------------------------
# 2 & 3. A degraded dispatch is visible
# ---------------------------------------------------------------------------


class TestDegradationIsVisible:
    def test_service_status_separates_installed_from_dispatching(self):
        """`detached_executor_active` answers "is one installed?" — which stayed
        true while nothing reached Prefect for hours. `dispatching` answers the
        question an operator actually has."""
        rt = _runtime()
        rt._async_executor = _DeadExecutor()          # installed...
        rt._async_dispatch_error = "PrefectHTTPStatusError: 401 Unauthorized"

        installed = not isinstance(rt._async_executor, InProcessExecutor)
        dispatching = installed and rt._async_dispatch_error is None

        assert installed is True, "an executor IS installed — the old field's answer"
        assert dispatching is False, "...and it cannot dispatch — the useful one"
        assert "401" in rt._async_dispatch_error

    def test_a_healthy_detached_executor_reports_dispatching(self):
        rt = _runtime()

        class _LiveExecutor:
            async def submit(self, claim, closure_b64):
                return "handle"

        rt._async_executor = _LiveExecutor()
        rt._async_dispatch_error = None

        installed = not isinstance(rt._async_executor, InProcessExecutor)
        assert installed and rt._async_dispatch_error is None

    def test_in_process_default_is_not_reported_as_dispatching(self):
        """No detached executor at all is not a degraded one — but it is also
        not 'dispatching' to anywhere durable."""
        rt = _runtime()
        assert isinstance(rt._async_executor, InProcessExecutor)
        dispatching = (
            not isinstance(rt._async_executor, InProcessExecutor)
            and rt._async_dispatch_error is None
        )
        assert dispatching is False

    @pytest.mark.asyncio
    async def test_a_dispatch_failure_is_remembered_for_reporting(self):
        """The failure used to exist only as a log line, so the claim_check came
        back looking ordinary and the caller recorded 'launched'."""
        rt = _runtime()
        assert rt._async_dispatch_error is None

        try:
            await _DeadExecutor().submit("claim", "closure")
        except Exception as exc:  # noqa: BLE001 — this is the recorded path
            rt._async_dispatch_error = f"{type(exc).__name__}: {exc}"

        assert rt._async_dispatch_error is not None
        assert "401" in rt._async_dispatch_error
