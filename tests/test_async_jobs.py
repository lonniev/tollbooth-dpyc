"""Tests for claim-check async jobs (tollbooth/async_jobs.py + runtime helpers)."""

import asyncio
import json
import uuid

import pytest
from pynostr.key import PrivateKey as _PK

from tollbooth.async_jobs import AsyncJobStore, poll_backoff_seconds, valid_claim
from tollbooth.runtime import OperatorRuntime

_TEST_KEY = _PK()
NPUB = _TEST_KEY.public_key.bech32()
OTHER_NPUB = _PK().public_key.bech32()
TOOL_ID = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Fake vault — in-memory simulation of the async_jobs table with a fake clock
# ---------------------------------------------------------------------------

class FakeVault:
    """Dispatches the store's fixed queries against an in-memory table.

    Mimics the Neon HTTP SQL API result shape — including the camelCase
    ``rowCount`` key the real API returns.
    """

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.now = 1_000_000.0  # fake epoch seconds; advance to simulate time

    def _t(self, table: str) -> str:
        return table

    def _computed(self, r: dict) -> dict:
        out = dict(r)
        out["elapsed_seconds"] = self.now - r["created_at"]
        out["stalled"] = (
            r["status"] == "running"
            and r["started_at"] is not None
            and r["started_at"] < self.now - r["max_runtime_seconds"]
        )
        out["expired"] = r["expires_at"] is not None and r["expires_at"] < self.now
        return out

    async def _execute(self, query: str, params: list | None = None):
        params = params or []
        if query.startswith("INSERT INTO async_jobs"):
            claim = str(uuid.uuid4())
            self.rows[claim] = {
                "claim": claim,
                "npub": params[0],
                "kind": params[1],
                "tool_id": params[2],
                # keep params as the JSON string Neon may hand back,
                # to exercise the deserialization-boundary casting
                "params": params[3],
                "status": "pending",
                "attempts": 0,
                "max_runtime_seconds": params[4],
                "result_ttl_seconds": params[5],
                "expected_seconds": params[6] if len(params) > 6 else 0,
                "result": None,
                "error": "",
                "created_at": self.now,
                "started_at": None,
                "expires_at": None,
            }
            return {"rows": [{"claim": claim}], "rowCount": 1}
        if "SET status = 'running'" in query:
            r = self.rows.get(params[0])
            claimable = r is not None and (
                r["status"] == "pending"
                or (
                    r["status"] == "running"
                    and r["started_at"] is not None
                    and r["started_at"] < self.now - r["max_runtime_seconds"]
                )
            )
            if not claimable:
                return {"rows": [], "rowCount": 0}
            r["status"] = "running"
            r["attempts"] += 1
            r["started_at"] = self.now
            return {"rows": [self._computed(r)], "rowCount": 1}
        if "SET status = 'done'" in query:
            r = self.rows[params[0]]
            r["status"] = "done"
            r["result"] = params[1]  # JSON string, as Neon may return it
            r["expires_at"] = self.now + r["result_ttl_seconds"]
            return {"rows": [], "rowCount": 1}
        if "SET status = 'error'" in query:
            r = self.rows[params[0]]
            r["status"] = "error"
            r["error"] = params[1]
            r["expires_at"] = self.now + r["result_ttl_seconds"]
            return {"rows": [], "rowCount": 1}
        if "SET status = 'pending'" in query:
            r = self.rows.get(params[0])
            if r is not None and r["status"] == "running":
                r["status"] = "pending"
            return {"rows": [], "rowCount": 1}
        if query.startswith("SELECT"):
            r = self.rows.get(params[0])
            if r is None or r["npub"] != params[1]:
                return {"rows": [], "rowCount": 0}
            return {"rows": [self._computed(r)], "rowCount": 1}
        if query.startswith("DELETE"):
            doomed = [
                c for c, r in self.rows.items()
                if (r["expires_at"] is not None and r["expires_at"] < self.now)
                or (
                    r["status"] in ("pending", "running")
                    and r["created_at"] < self.now - 24 * 3600
                )
            ]
            for c in doomed:
                del self.rows[c]
            return {"rows": [], "rowCount": len(doomed)}
        raise AssertionError(f"FakeVault: unhandled query: {query[:80]}")


@pytest.fixture
def vault():
    return FakeVault()


@pytest.fixture
def store(vault):
    return AsyncJobStore(vault)


async def _create(store, **kw) -> str:
    defaults = dict(
        npub=NPUB, kind="slow_thing", tool_id=TOOL_ID,
        params={"mode": "live", "difficulty": "hard"},
        max_runtime_seconds=300, result_ttl_seconds=900,
    )
    defaults.update(kw)
    return await store.create(**defaults)


# ---------------------------------------------------------------------------
# AsyncJobStore
# ---------------------------------------------------------------------------

class TestAsyncJobStore:

    async def test_lifecycle_create_claim_complete(self, store):
        claim = await _create(store)
        assert valid_claim(claim)

        job = await store.get(claim, NPUB)
        assert job["status"] == "pending"
        assert job["attempts"] == 0
        # params round-trip through the JSON boundary as a typed dict
        assert job["params"] == {"mode": "live", "difficulty": "hard"}

        claimed = await store.claim_for_run(claim)
        assert claimed is not None
        assert claimed["status"] == "running"
        assert claimed["attempts"] == 1
        assert claimed["kind"] == "slow_thing"

        await store.complete(claim, {"scenario": {"ticker": "XYZ"}})
        job = await store.get(claim, NPUB)
        assert job["status"] == "done"
        assert job["result"] == {"scenario": {"ticker": "XYZ"}}
        assert not job["expired"]

    async def test_claim_atomicity(self, store):
        claim = await _create(store)
        assert await store.claim_for_run(claim) is not None
        # second claimant loses — job is running and not stalled
        assert await store.claim_for_run(claim) is None

    async def test_stale_running_job_is_reclaimable(self, store, vault):
        claim = await _create(store, max_runtime_seconds=300)
        assert await store.claim_for_run(claim) is not None
        vault.now += 200
        assert await store.claim_for_run(claim) is None  # still within budget
        vault.now += 200  # now 400s in — past max_runtime
        reclaimed = await store.claim_for_run(claim)
        assert reclaimed is not None
        assert reclaimed["attempts"] == 2

    async def test_claim_check_is_npub_bound(self, store):
        claim = await _create(store)
        assert await store.get(claim, OTHER_NPUB) is None
        assert await store.get(claim, NPUB) is not None

    async def test_garbage_claims_never_reach_sql(self, store):
        for junk in ("", "DROP TABLE", "not-a-uuid", None):
            assert await store.get(junk, NPUB) is None
            assert await store.claim_for_run(junk) is None

    async def test_release_returns_job_to_pending(self, store):
        claim = await _create(store)
        await store.claim_for_run(claim)
        await store.release(claim)
        job = await store.get(claim, NPUB)
        assert job["status"] == "pending"
        assert job["attempts"] == 1  # attempts survive the release

    async def test_fail_sets_error_and_ttl(self, store, vault):
        claim = await _create(store, result_ttl_seconds=900)
        await store.claim_for_run(claim)
        await store.fail(claim, "Job execution failed. Check operator logs.")
        job = await store.get(claim, NPUB)
        assert job["status"] == "error"
        assert "operator logs" in job["error"]
        vault.now += 1000
        job = await store.get(claim, NPUB)
        assert job["expired"]

    async def test_purge_expired_and_abandoned(self, store, vault):
        done = await _create(store, result_ttl_seconds=100)
        await store.claim_for_run(done)
        await store.complete(done, {"ok": True})

        fresh = await _create(store)

        abandoned = await _create(store)  # pending, never claimed

        vault.now += 200  # done's TTL elapses; abandoned not yet 24h old
        assert await store.purge_expired() == 1
        assert await store.get(done, NPUB) is None
        assert await store.get(fresh, NPUB) is not None

        vault.now += 24 * 3600
        assert await store.purge_expired() == 2  # fresh + abandoned both stale
        assert await store.get(abandoned, NPUB) is None


# ---------------------------------------------------------------------------
# Runtime orchestration (start / _run_job / fetch)
# ---------------------------------------------------------------------------

def _make_runtime(vault: FakeVault) -> OperatorRuntime:
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    store = AsyncJobStore(vault)

    async def _store():
        return store

    rt.async_job_store = _store  # bypass vault bootstrap
    return rt


async def _drain(predicate, timeout: float = 2.0) -> None:
    """Let spawned tasks run until predicate() or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("background job did not settle in time")
        await asyncio.sleep(0.01)


class TestRuntimeAsyncJobs:

    async def test_start_returns_claim_and_job_completes(self, vault):
        rt = _make_runtime(vault)
        ran: list[dict] = []

        async def runner(**params):
            ran.append(params)
            return {"answer": 42}

        rt.register_job_runner("slow_thing", runner)
        out = await rt.start_async_job(
            "slow_thing", NPUB, {"x": "1"},
            tool_id=TOOL_ID, max_runtime_seconds=60, result_ttl_seconds=300,
        )
        assert out["success"] is True
        assert out["status"] == "pending"
        claim = out["claim_check"]
        assert valid_claim(claim)

        await _drain(lambda: ran)
        await _drain(
            lambda: vault.rows[claim]["status"] == "done"
        )
        assert ran == [{"x": "1"}]

        fetched = await rt.fetch_async_job(claim, NPUB)
        assert fetched["status"] == "done"
        assert fetched["result"] == {"answer": 42}

    async def test_start_requires_registered_runner(self, vault):
        rt = _make_runtime(vault)
        with pytest.raises(RuntimeError, match="No job runner or spec registered"):
            await rt.start_async_job(
                "unregistered", NPUB, {},
                tool_id=TOOL_ID, max_runtime_seconds=60, result_ttl_seconds=300,
            )

    async def test_terminal_failure_refunds_and_reports_error(self, vault):
        rt = _make_runtime(vault)
        rt._ASYNC_JOB_MAX_ATTEMPTS = 1  # first failure is terminal
        refunds: list[tuple] = []

        async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
            refunds.append((tool_id, npub, tool_kwargs))

        rt.rollback_debit = fake_rollback

        async def runner(**params):
            raise RuntimeError("nsec hunter2 leaked? never surface this")

        rt.register_job_runner("slow_thing", runner)
        out = await rt.start_async_job(
            "slow_thing", NPUB, {"x": "1"},
            tool_id=TOOL_ID, max_runtime_seconds=60, result_ttl_seconds=300,
        )
        claim = out["claim_check"]
        await _drain(lambda: vault.rows[claim]["status"] == "error")

        fetched = await rt.fetch_async_job(claim, NPUB)
        assert fetched["status"] == "error"
        assert fetched["refunded"] is True
        # the raw exception text must NOT reach the patron
        assert "hunter2" not in json.dumps(fetched)
        assert refunds == [(TOOL_ID, NPUB, {"x": "1"})]

    async def test_runner_exceeding_budget_is_cancelled_and_refunds(self, vault):
        rt = _make_runtime(vault)
        refunds: list[tuple] = []

        async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
            refunds.append((tool_id, npub, tool_kwargs))

        rt.rollback_debit = fake_rollback

        cancelled = {"hit": False}

        async def runner(**params):
            try:
                await asyncio.sleep(30)  # far longer than the 1s budget
            except asyncio.CancelledError:
                cancelled["hit"] = True
                raise
            return {"never": "reached"}

        rt.register_job_runner("slow_thing", runner)
        out = await rt.start_async_job(
            "slow_thing", NPUB, {"x": "1"},
            tool_id=TOOL_ID, max_runtime_seconds=1, result_ttl_seconds=300,
        )
        claim = out["claim_check"]
        await _drain(lambda: vault.rows[claim]["status"] == "error", timeout=5.0)

        # the runner was actually cancelled at its await point, not left running
        assert cancelled["hit"] is True

        fetched = await rt.fetch_async_job(claim, NPUB)
        assert fetched["status"] == "error"
        assert fetched["refunded"] is True
        # a budget timeout is a terminal, refundable, transient situation
        assert fetched["error_code"] == "job_timed_out"
        assert fetched["transient"] is True
        assert refunds == [(TOOL_ID, NPUB, {"x": "1"})]

    async def test_failure_then_retry_succeeds(self, vault):
        rt = _make_runtime(vault)
        calls = {"n": 0}

        async def runner(**params):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return {"ok": True}

        rt.register_job_runner("slow_thing", runner)
        out = await rt.start_async_job(
            "slow_thing", NPUB, {},
            tool_id=TOOL_ID, max_runtime_seconds=60, result_ttl_seconds=300,
        )
        claim = out["claim_check"]
        await _drain(lambda: vault.rows[claim]["status"] == "done", timeout=5.0)
        assert calls["n"] == 2

    async def test_fetch_unknown_claim_is_expired_guidance(self, vault):
        rt = _make_runtime(vault)
        fetched = await rt.fetch_async_job(str(uuid.uuid4()), NPUB)
        assert fetched["success"] is True
        assert fetched["status"] == "expired"
        assert "claim check" in fetched["next_steps"]

    async def test_fetch_rekicks_orphaned_pending_job(self, vault):
        """A pending row whose creator died gets re-kicked by the fetch."""
        rt = _make_runtime(vault)
        store = AsyncJobStore(vault)
        claim = await _create(store)  # row exists, but no task was spawned

        ran: list[dict] = []

        async def runner(**params):
            ran.append(params)
            return {"recovered": True}

        rt.register_job_runner("slow_thing", runner)
        fetched = await rt.fetch_async_job(claim, NPUB)
        assert fetched["status"] == "running"

        await _drain(lambda: vault.rows[claim]["status"] == "done")
        fetched = await rt.fetch_async_job(claim, NPUB)
        assert fetched["status"] == "done"
        assert fetched["result"] == {"recovered": True}

    async def test_fetch_rekicks_stalled_running_job(self, vault):
        rt = _make_runtime(vault)
        store = AsyncJobStore(vault)
        claim = await _create(store, max_runtime_seconds=300)
        await store.claim_for_run(claim)  # container "died" mid-run
        vault.now += 400  # past max_runtime

        async def runner(**params):
            return {"second_wind": True}

        rt.register_job_runner("slow_thing", runner)
        fetched = await rt.fetch_async_job(claim, NPUB)
        assert fetched["status"] == "running"
        assert fetched.get("recovered") is True

        await _drain(lambda: vault.rows[claim]["status"] == "done")
        assert vault.rows[claim]["attempts"] == 2

    async def test_fetch_running_within_budget_does_not_rekick(self, vault):
        rt = _make_runtime(vault)
        store = AsyncJobStore(vault)
        claim = await _create(store, max_runtime_seconds=300)
        await store.claim_for_run(claim)

        boom: list[str] = []

        async def runner(**params):
            boom.append("should not run")
            return {}

        rt.register_job_runner("slow_thing", runner)
        fetched = await rt.fetch_async_job(claim, NPUB)
        assert fetched["status"] == "running"
        assert "recovered" not in fetched
        await asyncio.sleep(0.05)
        assert boom == []

    async def test_missing_runner_on_recovery_fails_and_refunds(self, vault):
        """Re-kick on a container that never registered the runner."""
        rt = _make_runtime(vault)
        refunds: list[str] = []

        async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
            refunds.append(tool_id)

        rt.rollback_debit = fake_rollback
        store = AsyncJobStore(vault)
        claim = await _create(store, kind="never_registered")

        await rt._run_job(claim)
        job = await store.get(claim, NPUB)
        assert job["status"] == "error"
        assert "register" in job["error"]
        assert refunds == [TOOL_ID]

    async def test_running_poll_cadence_tightens_toward_deadline(self, vault):
        """Polling advice should tighten, not lengthen, as the job nears done."""
        rt = _make_runtime(vault)
        store = AsyncJobStore(vault)
        claim = await _create(store, max_runtime_seconds=300)
        await store.claim_for_run(claim)

        async def runner(**params):
            return {}

        rt.register_job_runner("slow_thing", runner)

        # Mid-run: advised at the steady ceiling, not a 3s hammer.
        early = await rt.fetch_async_job(claim, NPUB)
        assert early["poll_after_seconds"] == 30

        # 290s in (home stretch): poll MORE often, since done is imminent.
        vault.now += 290
        later = await rt.fetch_async_job(claim, NPUB)
        assert later["poll_after_seconds"] < early["poll_after_seconds"]

    async def test_declared_budget_drives_long_first_wait(self, vault):
        """A job with an author time budget sleeps most of it before polling.

        Exercises the full flow: expected_seconds persists on create, is read
        back on the row, and selects the budget poll curve in fetch.
        """
        rt = _make_runtime(vault)
        store = AsyncJobStore(vault)
        claim = await _create(store, max_runtime_seconds=200, expected_seconds=200)
        await store.claim_for_run(claim)

        async def runner(**params):
            return {}

        rt.register_job_runner("slow_thing", runner)

        # Fresh: ~75% of the declared budget up front (not max//10 == 20).
        early = await rt.fetch_async_job(claim, NPUB)
        assert early["poll_after_seconds"] == 150

        # 150s in: keeps using the budget curve, tightening (0.75 * remaining).
        vault.now += 150
        mid = await rt.fetch_async_job(claim, NPUB)
        assert mid["poll_after_seconds"] == 37


# ---------------------------------------------------------------------------
# poll_backoff_seconds
# ---------------------------------------------------------------------------

class TestPollBackoff:

    def test_steady_ceiling_through_the_bulk(self):
        # Early and mid-run both sit at the ceiling (max_runtime // 10, cap 30),
        # bounding how long a finished result waits to be noticed.
        assert poll_backoff_seconds(0, 300) == 30
        assert poll_backoff_seconds(120, 300) == 30

    def test_tightens_toward_the_deadline(self):
        # As the guaranteed-resolved deadline nears, intervals shrink — the
        # opposite of exponential backoff.
        assert poll_backoff_seconds(270, 300) == 15
        assert poll_backoff_seconds(290, 300) == 5

    def test_never_increases_as_job_ages(self):
        seq = [poll_backoff_seconds(e, 300) for e in range(0, 320, 10)]
        assert seq == sorted(seq, reverse=True)

    def test_floors_at_and_past_the_deadline(self):
        assert poll_backoff_seconds(300, 300) == 5
        assert poll_backoff_seconds(400, 300) == 5  # overrunning → stay tight

    def test_ceiling_scales_with_short_runtimes(self):
        # A 100s ceiling holds the bulk rate at max_runtime // 10 = 10s.
        assert poll_backoff_seconds(0, 100) == 10

    def test_unknown_max_runtime_floors(self):
        # No declared deadline → nothing to count down to; stay tight.
        assert poll_backoff_seconds(200, 0) == 5

    # --- author-declared time budget (expected_seconds) ---

    def test_budget_first_wait_is_75pct(self):
        # A 200s budget: trust it, sleep ~75% up front (150s) rather than
        # polling through the middle.
        assert poll_backoff_seconds(0, 200, expected_seconds=200) == 150

    def test_budget_tightens_geometrically(self):
        # Each poll waits ~75% of what's left, so it converges as the finish nears.
        assert poll_backoff_seconds(150, 200, expected_seconds=200) == 37  # 0.75*50
        assert poll_backoff_seconds(190, 200, expected_seconds=200) == 7   # 0.75*10

    def test_budget_floors_at_and_past_deadline(self):
        assert poll_backoff_seconds(200, 200, expected_seconds=200) == 5
        assert poll_backoff_seconds(260, 200, expected_seconds=200) == 5

    def test_budget_single_hop_capped(self):
        # Even a long budget never advises a single wait beyond the absolute cap.
        assert poll_backoff_seconds(0, 900, expected_seconds=900) == 300

    def test_budget_overrides_ceiling_regime(self):
        # With a budget, the steady max//10 ceiling does NOT apply — the whole
        # point is to wait long up front, not poll every ~max/10 seconds.
        assert poll_backoff_seconds(0, 200, expected_seconds=200) > 200 // 10
