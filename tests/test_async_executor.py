"""The pluggable async-job executor, and the detached path through
``OperatorRuntime.start_async_job`` / ``fetch_async_job``.

This file used to test a sealed-closure path: a job spec encrypted with
AES-256-GCM, shipped to a generic Prefect flow that interpreted an op
vocabulary, with the raw result shaped on the way back. All of that was deleted
in 0.82.0 — Modal runs the operator's own registered runner, so there is nothing
to seal, no vocabulary to validate and no result to shape.

What survives are the behaviours that were never really about Prefect:

* the executor is chosen by CREDENTIAL DELIVERY, cached only on a definitive
  answer, and never overrides an explicitly installed one;
* a transient vault miss must NOT be cached — otherwise a container that
  warm-started against a cold Neon is pinned to in-process for its whole life;
* a dispatch failure REFUNDS and says so, and never quietly runs the work here
  instead — that silent fallback is what hid the ``closure_b64=None`` 409 for
  days while ``service_status`` reported healthy;
* the job row is the source of truth; the detached handle is polled only to
  catch a run that died without writing one.
"""

import sys
import types
import uuid

import pytest
from pynostr.key import PrivateKey as _PK

from tollbooth.async_executor import InProcessExecutor, JobExecutor, ModalExecutor
from tollbooth.async_jobs import AsyncJobStore
from tollbooth.runtime import OperatorRuntime

NPUB = _PK().public_key.bech32()
TOOL_ID = str(uuid.uuid4())
MODAL_CREDS = {
    "modal_token_id": "ak-test",
    "modal_token_secret": "as-test",
    "modal_app_name": "excalibur-render",
}


class FakeVault:
    """In-memory async_jobs table, including the run_handle the detached path sets."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.now = 1_000_000.0

    def _t(self, table: str) -> str:
        return table

    def _computed(self, r: dict) -> dict:
        out = dict(r)
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
                "claim": claim, "npub": params[0], "kind": params[1],
                "tool_id": params[2], "params": params[3], "status": "pending",
                "attempts": 0, "max_runtime_seconds": params[4],
                "result_ttl_seconds": params[5], "result": None, "error": "",
                "run_handle": None, "created_at": self.now, "started_at": None,
                "expires_at": None,
            }
            return {"rows": [{"claim": claim}], "rowCount": 1}
        if "SET run_handle" in query:
            r = self.rows.get(params[0])
            if r is not None:
                r["run_handle"] = params[1]
            return {"rows": [], "rowCount": 1}
        if "SET status = 'done'" in query:
            r = self.rows[params[0]]
            r["status"], r["result"] = "done", params[1]
            r["expires_at"] = self.now + r["result_ttl_seconds"]
            return {"rows": [], "rowCount": 1}
        if "SET status = 'error'" in query:
            r = self.rows[params[0]]
            r["status"], r["error"] = "error", params[1]
            r["expires_at"] = self.now + r["result_ttl_seconds"]
            return {"rows": [], "rowCount": 1}
        if query.startswith("SELECT"):
            r = self.rows.get(params[0])
            if r is None or r["npub"] != params[1]:
                return {"rows": [], "rowCount": 0}
            return {"rows": [self._computed(r)], "rowCount": 1}
        if query.startswith("DELETE"):
            return {"rows": [], "rowCount": 0}
        raise AssertionError(f"FakeVault: unhandled query: {query[:80]}")


class RecordingExecutor:
    """A detached (non-in-process) executor: records submits, scripts poll."""

    def __init__(self, outcome=None, fail_submit: Exception | None = None):
        self.submits: list[str] = []
        self.outcome = outcome
        self.fail_submit = fail_submit

    async def submit(self, claim):
        if self.fail_submit:
            raise self.fail_submit
        self.submits.append(claim)
        return "handle-1"

    async def poll(self, handle):
        return self.outcome


@pytest.fixture
def vault_and_rt():
    vault = FakeVault()
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    store = AsyncJobStore(vault)

    async def _store():
        return store

    async def _load_credentials(field_names, *, service=None):
        return dict(MODAL_CREDS)

    rt.async_job_store = _store
    rt.load_credentials = _load_credentials
    rt.rollback_debit = _recorder(rt)
    return vault, rt


def _recorder(rt):
    rt.refunds = []

    async def _rollback(tool_id, npub, *, tool_kwargs=None):
        rt.refunds.append(tool_id)

    return _rollback


def _register_runner(rt, result=None, raises: Exception | None = None):
    async def runner(**params):
        if raises:
            raise raises
        return result if result is not None else {"text": "done"}

    rt.register_job_runner("resolve", runner)


# ---------------------------------------------------------------------------
# Executor primitives
# ---------------------------------------------------------------------------


def test_both_shipped_executors_satisfy_the_protocol():
    # The protocol is what lets the substrate change without start_async_job or
    # fetch_async_job knowing. It is the whole reason the Modal swap was small.
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    assert isinstance(InProcessExecutor(rt), JobExecutor)
    assert isinstance(ModalExecutor(app_name="a"), JobExecutor)


def test_submit_takes_only_a_claim():
    # `closure_b64` was removed rather than accepted-and-ignored: a vestigial
    # argument is an invitation to reintroduce the thing it served.
    import inspect

    for cls in (InProcessExecutor, ModalExecutor):
        params = list(inspect.signature(cls.submit).parameters)
        assert params == ["self", "claim"], f"{cls.__name__}.submit{params}"


# ---------------------------------------------------------------------------
# Dispatch — one path, and no silent fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_submits_the_claim_to_the_installed_executor(vault_and_rt):
    vault, rt = vault_and_rt
    _register_runner(rt)
    ex = RecordingExecutor()
    rt.set_async_executor(ex)

    out = await rt.start_async_job("resolve", NPUB, {"prompt": "hi"}, tool_id=TOOL_ID,
        max_runtime_seconds=600, result_ttl_seconds=3600)

    assert out["status"] == "pending" and out["claim_check"]
    assert ex.submits == [out["claim_check"]]
    assert vault.rows[out["claim_check"]]["run_handle"] == "handle-1"


@pytest.mark.asyncio
async def test_an_unregistered_kind_is_refused_before_anything_is_charged(vault_and_rt):
    _, rt = vault_and_rt
    with pytest.raises(RuntimeError, match="No job runner registered"):
        await rt.start_async_job("nope", NPUB, {}, tool_id=TOOL_ID,
        max_runtime_seconds=600, result_ttl_seconds=3600)


@pytest.mark.asyncio
async def test_a_dispatch_failure_refunds_and_never_runs_the_work_here(vault_and_rt):
    """The lesson of #178, kept after the code that taught it was deleted.

    The old handler caught a dispatch failure, quietly started an in-process
    task and returned an ordinary claim check. Callers recorded a launch,
    service_status reported the executor active, and the work ran on a front
    that recycles — so a job could simply vanish with no outcome and no reason
    for anyone to read.
    """
    _vault, rt = vault_and_rt
    ran = []

    async def runner(**params):
        ran.append(params)
        return {"text": "should never run"}

    rt.register_job_runner("resolve", runner)
    rt.set_async_executor(RecordingExecutor(fail_submit=RuntimeError("modal down")))

    out = await rt.start_async_job("resolve", NPUB, {}, tool_id=TOOL_ID,
        max_runtime_seconds=600, result_ttl_seconds=3600)

    assert out["status"] == "error" and out["refunded"] is True
    assert rt.refunds == [TOOL_ID]
    assert ran == [], "a dispatch failure must not fall back to in-process"
    assert rt._async_dispatch_error and "modal down" in rt._async_dispatch_error


# ---------------------------------------------------------------------------
# Fetch — the row is truth; the handle catches what never reached it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_reports_running_while_the_detached_run_is_open(vault_and_rt):
    _vault, rt = vault_and_rt
    _register_runner(rt)
    rt.set_async_executor(RecordingExecutor(outcome=None))
    claim = (await rt.start_async_job("resolve", NPUB, {}, tool_id=TOOL_ID,
        max_runtime_seconds=600, result_ttl_seconds=3600))["claim_check"]

    out = await rt.fetch_async_job(claim, NPUB)

    assert out["status"] == "running" and out["poll_after_seconds"] > 0
    assert rt.refunds == []


@pytest.mark.asyncio
async def test_a_crashed_or_cancelled_run_refunds_without_leaking_its_error(vault_and_rt):
    """Modal raises on `get` for a cancelled run — proven live:
    RemoteError("Function call was cancelled by user or a failure.").
    That must settle the job, refund once, and never reach the patron verbatim.
    """
    _vault, rt = vault_and_rt
    _register_runner(rt)
    rt.set_async_executor(RecordingExecutor(outcome={
        "status": "failed", "result": None,
        "error": "RemoteError: Function call was cancelled by user or a failure.",
    }))
    claim = (await rt.start_async_job("resolve", NPUB, {}, tool_id=TOOL_ID,
        max_runtime_seconds=600, result_ttl_seconds=3600))["claim_check"]

    out = await rt.fetch_async_job(claim, NPUB)

    assert out["status"] == "error" and out["refunded"] is True
    assert rt.refunds == [TOOL_ID]
    assert "RemoteError" not in str(out), "remote error text is operator-side only"


@pytest.mark.asyncio
async def test_a_completed_run_lets_the_row_settle_it(vault_and_rt):
    """The remote runner writes its own outcome — it is the operator's code with
    the operator's Neon. So a 'completed' poll shapes nothing and asserts
    nothing; it reports running once more and the next poll reads the row."""
    vault, rt = vault_and_rt
    _register_runner(rt)
    rt.set_async_executor(RecordingExecutor(outcome={
        "status": "completed", "result": {"text": "words"}, "error": None,
    }))
    claim = (await rt.start_async_job("resolve", NPUB, {}, tool_id=TOOL_ID,
        max_runtime_seconds=600, result_ttl_seconds=3600))["claim_check"]

    out = await rt.fetch_async_job(claim, NPUB)
    assert out["status"] == "running"

    vault.rows[claim].update(status="done", result={"text": "words"})
    assert (await rt.fetch_async_job(claim, NPUB))["result"] == {"text": "words"}


@pytest.mark.asyncio
async def test_recovery_of_a_stalled_job_goes_back_through_the_executor(vault_and_rt):
    """A Modal-configured operator must recover DETACHED. Re-kicking straight
    into an asyncio task would silently run the retry on the recycling front —
    the same substitution the dispatch-failure handler used to make."""
    vault, rt = vault_and_rt
    _register_runner(rt)
    ex = RecordingExecutor(outcome=None)
    rt.set_async_executor(ex)
    claim = (await rt.start_async_job("resolve", NPUB, {}, tool_id=TOOL_ID,
        max_runtime_seconds=600, result_ttl_seconds=3600))["claim_check"]
    vault.rows[claim].update(status="running", started_at=vault.now - 10_000)
    vault.rows[claim]["run_handle"] = None  # handle lost with the dead worker

    await rt.fetch_async_job(claim, NPUB)

    assert ex.submits == [claim, claim], "the stalled job was re-submitted detached"


# ---------------------------------------------------------------------------
# Auto-resolve — chosen by credential delivery, never by code
# ---------------------------------------------------------------------------


def _fake_modal_module():
    mod = types.ModuleType("modal")
    mod.Function = types.SimpleNamespace(from_name=lambda *a, **k: None)
    mod.FunctionCall = types.SimpleNamespace(from_id=lambda *a, **k: None)
    return mod


@pytest.mark.asyncio
async def test_modal_is_installed_when_the_creds_are_vaulted(vault_and_rt, monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", _fake_modal_module())
    _, rt = vault_and_rt
    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, ModalExecutor)
    assert rt._async_executor._app_name == "excalibur-render"
    assert rt._async_executor_error is None
    # The token comes from the VAULT, not ambient env: a host carrying its own
    # Modal identity must not silently receive another operator's work.
    import os

    assert os.environ["MODAL_TOKEN_ID"] == "ak-test"


@pytest.mark.asyncio
async def test_without_creds_it_stays_in_process(vault_and_rt, monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", _fake_modal_module())
    _, rt = vault_and_rt

    async def _empty(field_names, *, service=None):
        return {}

    rt.load_credentials = _empty
    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, InProcessExecutor)
    assert rt._async_executor_resolved is True  # definitive: cache it


@pytest.mark.asyncio
async def test_a_transient_vault_miss_is_not_cached(vault_and_rt, monkeypatch):
    """A cold Neon on warm-up is exactly when the first job arrives. Caching that
    miss would pin the container to in-process for its whole life."""
    monkeypatch.setitem(sys.modules, "modal", _fake_modal_module())
    _, rt = vault_and_rt
    calls = {"n": 0}

    async def _flaky(field_names, *, service=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("vault cold")
        return dict(MODAL_CREDS)

    rt.load_credentials = _flaky
    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, InProcessExecutor)
    assert rt._async_executor_resolved is False, "transient miss must stay unresolved"

    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, ModalExecutor)


@pytest.mark.asyncio
async def test_missing_modal_extra_degrades_loudly_rather_than_crashing(
    vault_and_rt, monkeypatch,
):
    _, rt = vault_and_rt
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _no_modal(name, *a, **k):
        if name == "modal" or name.startswith("modal."):
            raise ImportError("No module named 'modal'")
        return real_import(name, *a, **k)

    monkeypatch.delitem(sys.modules, "modal", raising=False)
    monkeypatch.setattr("builtins.__import__", _no_modal)
    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, InProcessExecutor)
    assert "modal" in (rt._async_executor_error or "")


@pytest.mark.asyncio
async def test_an_explicitly_installed_executor_is_never_overridden(vault_and_rt):
    _, rt = vault_and_rt
    ex = RecordingExecutor()
    rt.set_async_executor(ex)
    await rt._ensure_async_executor()
    assert rt._async_executor is ex
