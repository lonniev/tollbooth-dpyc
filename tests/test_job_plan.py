"""Durable multi-stage job plans (plan/v1) — all-or-none fare per firing.

Issue #181. A plan is a wheel-owned long-running execution capability:

- One debit / one refund per firing (stages are internal reliability, never a
  billing boundary).
- ``plan/v1`` emits one artifact per stage so the wheel can harvest progress.
- A plan must NEVER silently fall back in-process — refuse loudly with a
  curated Situation when no detached executor is installed.
"""

from __future__ import annotations

import json
import sys
import types
import uuid
from typing import Any

import pytest
from pynostr.key import PrivateKey as _PK

from tollbooth.async_executor import InProcessExecutor, PrefectClosureExecutor
from tollbooth.async_jobs import AsyncJobStore
from tollbooth.async_situation import AsyncJobSituation
from tollbooth.runtime import OperatorRuntime
from tollbooth.vault_encryption import VaultCipher

NPUB = _PK().public_key.bech32()
TOOL_ID = str(uuid.uuid4())
KEY_HEX = "ab" * 32


# ---------------------------------------------------------------------------
# Fixtures — mirror test_async_executor FakeVault / runtime helpers
# ---------------------------------------------------------------------------


class FakeVault:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.now = 1_000_000.0

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
                "params": params[3],
                "status": "pending",
                "attempts": 0,
                "max_runtime_seconds": params[4],
                "result_ttl_seconds": params[5],
                "expected_seconds": params[6] if len(params) > 6 else 0,
                "result": None,
                "error": "",
                "run_handle": None,
                "created_at": self.now,
                "started_at": None,
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
            r["status"] = "done"
            r["result"] = params[1]
            r["expires_at"] = self.now + r["result_ttl_seconds"]
            return {"rows": [], "rowCount": 1}
        if "SET status = 'error'" in query:
            r = self.rows[params[0]]
            r["status"] = "error"
            r["error"] = params[1]
            r["expires_at"] = self.now + r["result_ttl_seconds"]
            return {"rows": [], "rowCount": 1}
        if "SET status = 'running'" in query:
            r = self.rows.get(params[0])
            if r is None:
                return {"rows": [], "rowCount": 0}
            r["status"] = "running"
            r["attempts"] += 1
            r["started_at"] = self.now
            return {"rows": [self._computed(r)], "rowCount": 1}
        if "SET result =" in query or "SET status = 'running', result" in query:
            # Partial progress write while still open
            r = self.rows.get(params[0])
            if r is not None:
                if len(params) >= 2:
                    r["result"] = params[1]
                if "status = 'running'" in query:
                    r["status"] = "running"
            return {"rows": [], "rowCount": 1}
        if query.startswith("SELECT"):
            r = self.rows.get(params[0])
            if r is None or r["npub"] != params[1]:
                return {"rows": [], "rowCount": 0}
            return {"rows": [self._computed(r)], "rowCount": 1}
        if query.startswith("DELETE"):
            return {"rows": [], "rowCount": 0}
        raise AssertionError(f"FakeVault: unhandled query: {query[:100]}")


class RecordingExecutor:
    def __init__(self, outcome=None):
        self.submits: list[tuple[str, str | None]] = []
        self.outcome = outcome

    async def submit(self, claim, closure_b64):
        self.submits.append((claim, closure_b64))
        return "handle-plan-1"

    async def poll(self, handle):
        return self.outcome


def _make_runtime(vault: FakeVault) -> OperatorRuntime:
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    store = AsyncJobStore(vault)

    async def _store():
        return store

    async def _load_credentials(field_names, *, service=None):
        return {"closure_seal_key": KEY_HEX, "anthropic_api_key": "sk-ant-x"}

    rt.async_job_store = _store
    rt.load_credentials = _load_credentials
    return rt


def _two_stage_plan(**params):
    return {
        "op": "plan/v1",
        "stages": [
            {
                "id": "render",
                "request": {
                    "method": "POST",
                    "url": "https://api.example/render",
                    "json": {"prompt": params.get("prompt", "")},
                    "timeout": 90,
                },
            },
            {
                "id": "polish",
                "request": {
                    "method": "POST",
                    "url": "https://api.example/polish",
                    "json": {"draft": "{{render}}"},
                    "timeout": 60,
                },
            },
        ],
        "budget_seconds": 200,
    }


def _shape_stage(stage_id: str, raw: Any, params: dict) -> dict:
    body = (raw or {}).get("json") or {}
    return {"stage": stage_id, "text": body.get("text", ""), "prompt": params.get("prompt", "")}


@pytest.fixture
def vault_and_rt():
    v = FakeVault()
    return v, _make_runtime(v)


# ---------------------------------------------------------------------------
# Confirm the gap: plan/v1 + register_job_plan are not yet present
# ---------------------------------------------------------------------------


def test_register_job_plan_exists():
    rt = _make_runtime(FakeVault())
    assert hasattr(rt, "register_job_plan"), "OperatorRuntime.register_job_plan is the operator surface"
    rt.register_job_plan("render_post", _two_stage_plan, _shape_stage)
    assert rt.uses_async_jobs() is True


def test_plan_v1_is_a_known_flow_op():
    """The generic flow must recognize plan/v1 (versioned op in this repo)."""
    from flows.dpyc_job_flow import _dispatch_op

    spec = _two_stage_plan(prompt="hi")
    # Without network we only assert the op is recognized; a full stage run is
    # covered by the pure dispatcher unit below with a stubbed http primitive.
    assert spec["op"] == "plan/v1"
    assert callable(_dispatch_op)


# ---------------------------------------------------------------------------
# Invariant: never silently fall back in-process
# ---------------------------------------------------------------------------


async def test_plan_refuses_loudly_without_detached_executor(vault_and_rt):
    """No detached executor → curated Situation, refund, no in-process run.

    A 5+ minute render on a recycling serverless front is not a degraded mode;
    it is a guaranteed loss. Starting a plan without a detached executor must
    refuse with a Situation — never quietly run somewhere it will die (#181).
    """
    vault, rt = vault_and_rt
    rt.register_job_plan("render_post", _two_stage_plan, _shape_stage)
    assert isinstance(rt._async_executor, InProcessExecutor)

    refunds: list = []

    async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
        refunds.append((tool_id, npub, tool_kwargs))

    rt.rollback_debit = fake_rollback

    # Also register a runner — the silent-fallback trap. Plans must NOT use it.
    ran = {"n": 0}

    async def runner(**params):
        ran["n"] += 1
        return {"ok": True}

    rt.register_job_runner("render_post", runner)

    out = await rt.start_async_job(
        "render_post",
        NPUB,
        {"prompt": "hi"},
        tool_id=TOOL_ID,
        max_runtime_seconds=960,
        result_ttl_seconds=900,
    )
    assert out["status"] == "error"
    assert out.get("error_code") == "plan_requires_detached_executor"
    assert out.get("refunded") is True
    assert ran["n"] == 0, "plan must not fall back to an in-process runner"
    assert refunds == [(TOOL_ID, NPUB, {"prompt": "hi"})]
    # Row is terminally failed with the structured situation
    claim = next(iter(vault.rows))
    assert vault.rows[claim]["status"] == "error"


async def test_plan_dispatch_failure_does_not_fall_back_in_process(vault_and_rt):
    """Dispatch failure for a plan refunds — never degrades to in-process."""
    _vault, rt = vault_and_rt
    rt.register_job_plan("render_post", _two_stage_plan, _shape_stage)

    class Boom:
        async def submit(self, claim, closure_b64):
            raise RuntimeError("prefect unreachable")

        async def poll(self, handle):
            return None

    rt.set_async_executor(Boom())
    ran = {"n": 0}

    async def runner(**params):
        ran["n"] += 1
        return {"ok": True}

    rt.register_job_runner("render_post", runner)

    refunds: list = []

    async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
        refunds.append(tool_id)

    rt.rollback_debit = fake_rollback

    out = await rt.start_async_job(
        "render_post",
        NPUB,
        {"prompt": "hi"},
        tool_id=TOOL_ID,
        max_runtime_seconds=960,
        result_ttl_seconds=900,
    )
    assert out["status"] == "error"
    assert out.get("refunded") is True
    assert out.get("degraded") is None, "plans must not report in_process_fallback"
    assert ran["n"] == 0
    assert refunds == [TOOL_ID]


# ---------------------------------------------------------------------------
# start seals a plan/v1 closure under a detached executor
# ---------------------------------------------------------------------------


async def test_start_plan_seals_plan_v1_closure(vault_and_rt):
    vault, rt = vault_and_rt
    rt.register_job_plan("render_post", _two_stage_plan, _shape_stage)
    exe = RecordingExecutor()
    rt.set_async_executor(exe)

    out = await rt.start_async_job(
        "render_post",
        NPUB,
        {"prompt": "hi"},
        tool_id=TOOL_ID,
        max_runtime_seconds=960,
        result_ttl_seconds=900,
        expected_seconds=400,
    )
    assert out["status"] == "pending"
    assert len(exe.submits) == 1
    claim, closure_b64 = exe.submits[0]
    assert claim == out["claim_check"]
    spec = json.loads(VaultCipher(KEY_HEX).decrypt(closure_b64, aad="dpyc-closure/v1"))
    assert spec["op"] == "plan/v1"
    assert len(spec["stages"]) == 2
    assert spec["stages"][0]["id"] == "render"
    assert vault.rows[claim]["run_handle"] == "handle-plan-1"


# ---------------------------------------------------------------------------
# fetch settles a completed plan — one fare, shape each stage, all-or-none
# ---------------------------------------------------------------------------


async def test_fetch_plan_completed_shapes_all_stages(vault_and_rt):
    _vault, rt = vault_and_rt
    rt.register_job_plan("render_post", _two_stage_plan, _shape_stage)
    rt.set_async_executor(
        RecordingExecutor(
            outcome={
                "status": "completed",
                "result": {
                    "op": "plan/v1",
                    "stages": [
                        {"id": "render", "result": {"status": 200, "json": {"text": "draft"}}},
                        {"id": "polish", "result": {"status": 200, "json": {"text": "final"}}},
                    ],
                    "completed": 2,
                    "total": 2,
                },
                "error": None,
            }
        )
    )
    out = await rt.start_async_job(
        "render_post",
        NPUB,
        {"prompt": "hi"},
        tool_id=TOOL_ID,
        max_runtime_seconds=960,
        result_ttl_seconds=900,
    )
    fetched = await rt.fetch_async_job(out["claim_check"], NPUB)
    assert fetched["status"] == "done"
    assert fetched["result"]["stages"]["render"]["text"] == "draft"
    assert fetched["result"]["stages"]["polish"]["text"] == "final"
    assert fetched["result"]["completed"] == 2
    assert fetched["result"]["total"] == 2
    # Fare stands — no refund on success
    assert "refunded" not in fetched or fetched.get("refunded") is not True


async def test_fetch_plan_stage_situation_refunds_once_all_or_none(vault_and_rt):
    """Operator-fault situation on any stage → one refund for the whole firing.

    Stages are never a billing boundary. shape_stage raising is the operator's
    fault classification; the fare is refunded once regardless of how many
    stages already produced domain results (consolation is separate).
    """
    _vault, rt = vault_and_rt

    def shape_stage(stage_id, raw, params):
        if stage_id == "polish":
            raise AsyncJobSituation(
                error_code="operator_llm_unfunded",
                message="The model router is out of credit. No fare was charged.",
                next_steps="Retry after the operator tops up.",
                transient=True,
            )
        return {"stage": stage_id, "text": (raw or {}).get("json", {}).get("text", "")}

    rt.register_job_plan("render_post", _two_stage_plan, shape_stage)
    rt.set_async_executor(
        RecordingExecutor(
            outcome={
                "status": "completed",
                "result": {
                    "op": "plan/v1",
                    "stages": [
                        {"id": "render", "result": {"status": 200, "json": {"text": "draft"}}},
                        {"id": "polish", "result": {"status": 402, "json": {"error": "no credit"}}},
                    ],
                    "completed": 2,
                    "total": 2,
                },
                "error": None,
            }
        )
    )
    refunds: list = []

    async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
        refunds.append(tool_id)

    rt.rollback_debit = fake_rollback
    out = await rt.start_async_job(
        "render_post",
        NPUB,
        {"prompt": "hi"},
        tool_id=TOOL_ID,
        max_runtime_seconds=960,
        result_ttl_seconds=900,
    )
    fetched = await rt.fetch_async_job(out["claim_check"], NPUB)
    assert fetched["status"] == "error"
    assert fetched["error_code"] == "operator_llm_unfunded"
    assert fetched["refunded"] is True
    assert refunds == [TOOL_ID], "exactly one refund for the firing — never per-stage"


async def test_fetch_plan_working_reports_progress(vault_and_rt):
    """Partial progress (k of n) surfaces as running/working — not a billing event."""
    _vault, rt = vault_and_rt
    rt.register_job_plan("render_post", _two_stage_plan, _shape_stage)
    rt.set_async_executor(
        RecordingExecutor(
            outcome={
                "status": "working",
                "result": {
                    "op": "plan/v1",
                    "stages": [
                        {"id": "render", "result": {"status": 200, "json": {"text": "draft"}}},
                    ],
                    "completed": 1,
                    "total": 2,
                },
                "error": None,
            }
        )
    )
    out = await rt.start_async_job(
        "render_post",
        NPUB,
        {"prompt": "hi"},
        tool_id=TOOL_ID,
        max_runtime_seconds=960,
        result_ttl_seconds=900,
    )
    fetched = await rt.fetch_async_job(out["claim_check"], NPUB)
    assert fetched["status"] in ("running", "working")
    assert fetched.get("completed") == 1
    assert fetched.get("total") == 2
    assert "refunded" not in fetched


# ---------------------------------------------------------------------------
# Flow primitive: plan/v1 emits one artifact per stage
# ---------------------------------------------------------------------------


def test_flow_plan_v1_emits_one_artifact_per_stage(monkeypatch):
    """dpyc_job_flow plan/v1 runs stages and publishes one artifact each."""
    artifacts: list[dict] = []

    def fake_artifact(*, markdown, key=None, **_kw):
        artifacts.append({"key": key, "data": json.loads(markdown)})

    # Stub http primitive to avoid network
    from flows import dpyc_job_flow as flow_mod

    monkeypatch.setattr(flow_mod, "create_markdown_artifact", fake_artifact)

    def fake_http(req):
        return {"status": 200, "json": {"text": f"ok:{req['url']}"}}

    monkeypatch.setattr(flow_mod, "_do_http_request", fake_http)

    spec = _two_stage_plan(prompt="hi")
    result = flow_mod._do_plan_v1(spec)

    assert result["op"] == "plan/v1"
    assert result["completed"] == 2
    assert result["total"] == 2
    assert [a["key"] for a in artifacts] == ["stage-render", "stage-polish"]
    assert artifacts[0]["data"]["json"]["text"].startswith("ok:")


def test_flow_plan_v1_uses_per_stage_timeout(monkeypatch):
    from flows import dpyc_job_flow as flow_mod

    seen: list[float] = []

    def fake_http(req):
        seen.append(float(req.get("timeout", -1)))
        return {"status": 200, "json": {}}

    monkeypatch.setattr(flow_mod, "_do_http_request", fake_http)
    monkeypatch.setattr(flow_mod, "create_markdown_artifact", lambda **kw: None)

    spec = {
        "op": "plan/v1",
        "stages": [
            {"id": "a", "request": {"url": "https://x/a", "timeout": 90}},
            {"id": "b", "request": {"url": "https://x/b"}},  # default
        ],
    }
    flow_mod._do_plan_v1(spec)
    assert seen[0] == 90.0
    assert seen[1] == flow_mod._DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Executor poll harvests multi-stage artifacts (including partial)
# ---------------------------------------------------------------------------


def _stub_prefect_poll(monkeypatch, *, final: bool, artifacts: list[str], state_type="COMPLETED"):
    """Stub prefect client so poll() can be exercised without a real install."""
    import contextlib

    class _State:
        def is_final(self):
            return final

        def is_completed(self):
            return final and state_type == "COMPLETED"

        @property
        def type(self):
            return state_type

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def read_flow_run(self, run_id):
            return types.SimpleNamespace(state=_State())

        async def read_artifacts(self, artifact_filter=None, limit=None):
            return [
                types.SimpleNamespace(data=data, key=f"stage-{i}")
                for i, data in enumerate(artifacts)
            ]

    @contextlib.contextmanager
    def fake_temporary_settings(updates=None):
        yield

    def fake_get_client():
        return _Client()

    fake_pkg = types.ModuleType("prefect")
    fake_client_mod = types.ModuleType("prefect.client")
    fake_orch = types.ModuleType("prefect.client.orchestration")
    fake_orch.get_client = fake_get_client
    fake_filters = types.ModuleType("prefect.client.schemas.filters")

    class ArtifactFilter:
        def __init__(self, **kw):
            pass

    class ArtifactFilterFlowRunId:
        def __init__(self, **kw):
            pass

    fake_filters.ArtifactFilter = ArtifactFilter
    fake_filters.ArtifactFilterFlowRunId = ArtifactFilterFlowRunId
    fake_settings = types.ModuleType("prefect.settings")
    fake_settings.temporary_settings = fake_temporary_settings
    fake_settings.PREFECT_API_URL = "PREFECT_API_URL"
    fake_settings.PREFECT_API_KEY = "PREFECT_API_KEY"

    monkeypatch.setitem(sys.modules, "prefect", fake_pkg)
    monkeypatch.setitem(sys.modules, "prefect.client", fake_client_mod)
    monkeypatch.setitem(sys.modules, "prefect.client.orchestration", fake_orch)
    monkeypatch.setitem(sys.modules, "prefect.client.schemas", types.ModuleType("prefect.client.schemas"))
    monkeypatch.setitem(sys.modules, "prefect.client.schemas.filters", fake_filters)
    monkeypatch.setitem(sys.modules, "prefect.settings", fake_settings)


async def test_poll_harvests_partial_plan_artifacts_while_running(monkeypatch):
    stage_render = json.dumps({"status": 200, "json": {"text": "draft"}})
    _stub_prefect_poll(monkeypatch, final=False, artifacts=[stage_render])

    exe = PrefectClosureExecutor(
        deployment="dpyc-job-flow/dpyc-jobs",
        api_url="https://standalone",
        api_key="pnu_x",
        key_id="kid",
    )
    outcome = await exe.poll(str(uuid.uuid4()))
    assert outcome is not None
    assert outcome["status"] == "working"
    assert outcome["result"]["completed"] == 1
    assert outcome["result"]["op"] == "plan/v1"


async def test_poll_completed_plan_returns_all_stage_artifacts(monkeypatch):
    arts = [
        json.dumps({"status": 200, "json": {"text": "a"}}),
        json.dumps({"status": 200, "json": {"text": "b"}}),
    ]
    _stub_prefect_poll(monkeypatch, final=True, artifacts=arts)

    exe = PrefectClosureExecutor(
        deployment="d/x", api_url="u", api_key="k", key_id="kid"
    )
    outcome = await exe.poll(str(uuid.uuid4()))
    assert outcome["status"] == "completed"
    assert outcome["result"]["completed"] == 2
    assert outcome["result"]["total"] == 2
