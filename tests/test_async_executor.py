"""Tests for the pluggable async-job executor (tollbooth/async_executor.py
and the closure path through OperatorRuntime.start_async_job / fetch_async_job)."""

import json
import sys
import types
import uuid

import pytest
from pynostr.key import PrivateKey as _PK

from tollbooth.async_executor import (
    InProcessExecutor,
    JobExecutor,
    PrefectClosureExecutor,
)
from tollbooth.async_jobs import AsyncJobStore
from tollbooth.runtime import OperatorRuntime
from tollbooth.vault_encryption import VaultCipher

NPUB = _PK().public_key.bech32()
TOOL_ID = str(uuid.uuid4())
KEY_HEX = "ab" * 32  # 64 hex chars → 32-byte AES-256 key
ANTHROPIC_SECRET = "sk-ant-SUPER-SECRET-do-not-leak"


# ---------------------------------------------------------------------------
# A run_handle-aware in-memory vault (closure path uses set_run_handle)
# ---------------------------------------------------------------------------

class FakeVault:
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
        if query.startswith("SELECT"):
            r = self.rows.get(params[0])
            if r is None or r["npub"] != params[1]:
                return {"rows": [], "rowCount": 0}
            return {"rows": [self._computed(r)], "rowCount": 1}
        if query.startswith("DELETE"):
            return {"rows": [], "rowCount": 0}
        raise AssertionError(f"FakeVault: unhandled query: {query[:80]}")


class RecordingExecutor:
    """A non-in-process executor that records submits and returns a scripted poll."""

    def __init__(self, outcome=None):
        self.submits: list[tuple[str, str | None]] = []
        self.outcome = outcome  # what poll() returns

    async def submit(self, claim, closure_b64):
        self.submits.append((claim, closure_b64))
        return "handle-1"

    async def poll(self, handle):
        return self.outcome


def _make_runtime(vault: FakeVault) -> OperatorRuntime:
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    store = AsyncJobStore(vault)

    async def _store():
        return store

    async def _load_credentials(field_names, *, service=None):
        return {"closure_seal_key": KEY_HEX, "anthropic_api_key": ANTHROPIC_SECRET}

    rt.async_job_store = _store
    rt.load_credentials = _load_credentials
    return rt


def _register_http_spec(rt):
    """A spec whose closure embeds the Anthropic key in request headers."""

    async def build_closure(**params):
        creds = await rt.load_credentials(["anthropic_api_key"])
        return {
            "op": "http_request",
            "request": {
                "method": "POST",
                "url": "https://api.anthropic.com/v1/messages",
                "headers": {"x-api-key": creds["anthropic_api_key"]},
                "json": {"prompt": params.get("prompt", "")},
            },
        }

    def shape_result(raw, params):
        # params is the job's persisted kwargs — threaded through so a stateful
        # job can run its param-dependent side effects while shaping.
        assert isinstance(params, dict)
        return {"x_text": (raw or {}).get("text", ""), "prompt": params.get("prompt", "")}

    rt.register_job_spec("resolve", build_closure, shape_result)


# ---------------------------------------------------------------------------
# Executor primitives
# ---------------------------------------------------------------------------

def test_executors_satisfy_protocol():
    assert isinstance(InProcessExecutor(object()), JobExecutor)
    assert isinstance(
        PrefectClosureExecutor(deployment="d/x", api_url="u", api_key="k", key_id="id"),
        JobExecutor,
    )


def test_closure_seal_open_roundtrip_and_aad_binding():
    spec = {"op": "http_request", "secret": ANTHROPIC_SECRET}
    cipher = VaultCipher(KEY_HEX)
    sealed = cipher.encrypt(json.dumps(spec), aad="dpyc-closure/v1")
    # ciphertext never reveals the plaintext secret
    assert ANTHROPIC_SECRET not in sealed
    # round-trips with the right key + AAD
    assert json.loads(cipher.decrypt(sealed, aad="dpyc-closure/v1")) == spec
    # wrong AAD fails the tag check
    with pytest.raises(Exception):  # noqa: B017
        cipher.decrypt(sealed, aad="wrong")
    # wrong key fails
    with pytest.raises(Exception):  # noqa: B017
        VaultCipher("cd" * 32).decrypt(sealed, aad="dpyc-closure/v1")


# ---------------------------------------------------------------------------
# Closure-path selection
# ---------------------------------------------------------------------------

async def test_closure_path_requires_spec_and_detached_executor():
    rt = _make_runtime(FakeVault())
    _register_http_spec(rt)
    # default executor is in-process → no closure path even with a spec
    assert rt._uses_closure_path("resolve") is False
    rt.set_async_executor(RecordingExecutor())
    assert rt._uses_closure_path("resolve") is True
    # unknown kind never takes the closure path
    assert rt._uses_closure_path("nope") is False


# ---------------------------------------------------------------------------
# start_async_job — closure path seals, submits, stores handle
# ---------------------------------------------------------------------------

async def test_start_closure_path_seals_and_only_ciphertext_crosses(vault_and_rt):
    vault, rt = vault_and_rt
    _register_http_spec(rt)
    exe = RecordingExecutor()
    rt.set_async_executor(exe)

    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    assert out["status"] == "pending"
    claim = out["claim_check"]

    # exactly one submit, carrying ciphertext (NOT the Anthropic key)
    assert len(exe.submits) == 1
    sub_claim, closure_b64 = exe.submits[0]
    assert sub_claim == claim
    assert closure_b64 and ANTHROPIC_SECRET not in closure_b64
    # the handle was persisted
    assert vault.rows[claim]["run_handle"] == "handle-1"
    # and the sealed closure decrypts back to a spec containing the key
    spec = json.loads(VaultCipher(KEY_HEX).decrypt(closure_b64, aad="dpyc-closure/v1"))
    assert spec["request"]["headers"]["x-api-key"] == ANTHROPIC_SECRET


# ---------------------------------------------------------------------------
# fetch_async_job — closure path settles from the executor poll
# ---------------------------------------------------------------------------

async def test_fetch_closure_completed_shapes_and_completes(vault_and_rt):
    vault, rt = vault_and_rt
    _register_http_spec(rt)
    rt.set_async_executor(RecordingExecutor(
        outcome={"status": "completed", "result": {"text": "woven answer"}, "error": None}
    ))
    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    claim = out["claim_check"]
    fetched = await rt.fetch_async_job(claim, NPUB)
    assert fetched["status"] == "done"
    assert fetched["result"] == {"x_text": "woven answer", "prompt": "hi"}
    assert vault.rows[claim]["status"] == "done"


async def test_fetch_closure_failed_refunds(vault_and_rt):
    _vault, rt = vault_and_rt
    _register_http_spec(rt)
    rt.set_async_executor(RecordingExecutor(
        outcome={"status": "failed", "result": None, "error": "CRASHED"}
    ))
    refunds: list[tuple] = []

    async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
        refunds.append((tool_id, npub, tool_kwargs))

    rt.rollback_debit = fake_rollback
    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    claim = out["claim_check"]
    fetched = await rt.fetch_async_job(claim, NPUB)
    assert fetched["status"] == "error"
    assert fetched["refunded"] is True
    # never surface the upstream error body
    assert "CRASHED" not in json.dumps(fetched)
    assert refunds == [(TOOL_ID, NPUB, {"prompt": "hi"})]


async def test_fetch_closure_completed_but_unshapeable_refunds(vault_and_rt):
    """A run that finishes but whose result can't be shaped (e.g. an upstream
    non-2xx surfaced, or empty output) refunds — symmetric with in-process."""
    vault, rt = vault_and_rt

    async def build_closure(**params):
        return {"op": "http_request", "request": {"url": "https://x"}}

    def shape_result(raw, params):
        raise ValueError("no text returned")

    rt.register_job_spec("resolve", build_closure, shape_result)
    rt.set_async_executor(RecordingExecutor(
        outcome={"status": "completed", "result": {"status": 500}, "error": None}
    ))
    refunds: list[tuple] = []

    async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
        refunds.append((tool_id, npub))

    rt.rollback_debit = fake_rollback
    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    claim = out["claim_check"]
    fetched = await rt.fetch_async_job(claim, NPUB)
    assert fetched["status"] == "error"
    assert fetched["refunded"] is True
    # the shaping error detail never leaks to the caller
    assert "no text returned" not in json.dumps(fetched)
    assert refunds == [(TOOL_ID, NPUB)]
    assert vault.rows[claim]["status"] == "error"


async def test_fetch_closure_situation_surfaces_curated_fields(vault_and_rt):
    """shape_result raising AsyncJobSituation -> fetch returns the curated code +
    message + transient (the frontend's UX data), refunds, and persists it
    structured so a later poll returns the same thing — no raw upstream body."""
    from tollbooth.async_situation import AsyncJobSituation

    _vault, rt = vault_and_rt

    async def build_closure(**params):
        return {"op": "http_request", "request": {"url": "https://x"}}

    def shape_result(raw, params):
        # the operator classifies the raw upstream response into a situation;
        # the raw body (with a fake request_id) must NOT reach the patron
        assert raw["status"] == 400  # operator sees the raw status
        raise AsyncJobSituation(
            error_code="operator_llm_unfunded",
            message="The AI provider is temporarily unavailable. No fare was charged.",
            next_steps="Try again shortly.",
            transient=False,
        )

    rt.register_job_spec("resolve", build_closure, shape_result)
    rt.set_async_executor(RecordingExecutor(outcome={
        "status": "completed",
        "result": {"status": 400, "json": {"error": {"message": "credit balance too low",
                                                      "request_id": "req_SECRET123"}}},
        "error": None,
    }))
    refunds: list = []

    async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
        refunds.append(tool_id)

    rt.rollback_debit = fake_rollback
    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    claim = out["claim_check"]

    fetched = await rt.fetch_async_job(claim, NPUB)
    assert fetched["status"] == "error"
    assert fetched["error_code"] == "operator_llm_unfunded"
    assert fetched["transient"] is False
    assert fetched["refunded"] is True
    # the raw upstream body / request_id never reaches the patron
    assert "req_SECRET123" not in json.dumps(fetched)
    assert "credit balance too low" not in json.dumps(fetched)
    assert refunds == [TOOL_ID]

    # a SUBSEQUENT poll (row already 'error') returns the same structured situation
    again = await rt.fetch_async_job(claim, NPUB)
    assert again["error_code"] == "operator_llm_unfunded"
    assert again["transient"] is False
    assert "req_SECRET123" not in json.dumps(again)


async def test_fetch_closure_generic_exception_never_leaks_on_repoll(vault_and_rt):
    """A non-situation shape_result exception refunds generically AND the stored
    row must not carry the raw exception text (fixes a latent leak on re-poll)."""
    _vault, rt = vault_and_rt

    async def build_closure(**params):
        return {"op": "http_request", "request": {"url": "https://x"}}

    def shape_result(raw, params):
        raise ValueError("RAW-SECRET-abc in the exception text")

    rt.register_job_spec("resolve", build_closure, shape_result)
    rt.set_async_executor(RecordingExecutor(outcome={
        "status": "completed", "result": {"status": 500}, "error": None,
    }))

    async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
        pass

    rt.rollback_debit = fake_rollback
    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    claim = out["claim_check"]
    first = await rt.fetch_async_job(claim, NPUB)
    second = await rt.fetch_async_job(claim, NPUB)  # row now 'error'
    for resp in (first, second):
        assert resp["status"] == "error" and resp["refunded"] is True
        assert "RAW-SECRET-abc" not in json.dumps(resp)


async def test_start_closure_build_situation_is_terminal_refund(vault_and_rt):
    """A build_closure that raises AsyncJobSituation (a pre-flight rejection —
    not-found entry, unfunded probe) settles terminally: the job row is failed
    with the curated situation, the fare is refunded, and the caller gets the
    situation response — NOT a dispatch-failure fallback / retry."""
    from tollbooth.async_situation import AsyncJobSituation

    vault, rt = vault_and_rt

    async def build_closure(**params):
        raise AsyncJobSituation(
            error_code="journal_entry_not_found",
            message="That scenario is gone. No fare was charged.",
            next_steps="Deal a fresh one.",
            transient=False,
        )

    def shape_result(raw, params):  # never reached
        return {}

    rt.register_job_spec("resolve", build_closure, shape_result)
    rt.set_async_executor(RecordingExecutor())
    refunds: list = []

    async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
        refunds.append((tool_id, tool_kwargs))

    rt.rollback_debit = fake_rollback

    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    # terminal, curated, refunded — no claim check to poll
    assert out["status"] == "error"
    assert out["error_code"] == "journal_entry_not_found"
    assert out["transient"] is False
    assert refunds == [(TOOL_ID, {"prompt": "hi"})]
    # the row was persisted as a structured error, so a later poll agrees
    claim = next(iter(vault.rows))
    assert vault.rows[claim]["status"] == "error"
    again = await rt.fetch_async_job(claim, NPUB)
    assert again["error_code"] == "journal_entry_not_found"


async def test_fetch_closure_still_running(vault_and_rt):
    vault, rt = vault_and_rt
    _register_http_spec(rt)
    rt.set_async_executor(RecordingExecutor(outcome=None))  # not final yet
    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    claim = out["claim_check"]
    fetched = await rt.fetch_async_job(claim, NPUB)
    assert fetched["status"] == "running"
    assert vault.rows[claim]["status"] == "pending"  # untouched


async def test_fetch_closure_path_never_rekicks_run_job(vault_and_rt):
    _vault, rt = vault_and_rt
    _register_http_spec(rt)
    rt.set_async_executor(RecordingExecutor(outcome=None))

    called = {"n": 0}

    async def boom(claim):
        called["n"] += 1

    rt._run_job = boom  # the watchdog must NOT fire on the closure path
    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    await rt.fetch_async_job(out["claim_check"], NPUB)
    assert called["n"] == 0


# ---------------------------------------------------------------------------
# start_async_job dispatch-failure fallback
# ---------------------------------------------------------------------------

async def test_dispatch_failure_without_runner_refunds(vault_and_rt):
    _vault, rt = vault_and_rt
    _register_http_spec(rt)

    class Boom:
        async def submit(self, claim, closure_b64):
            raise RuntimeError("prefect unreachable")

        async def poll(self, handle):
            return None

    rt.set_async_executor(Boom())
    refunds: list[str] = []

    async def fake_rollback(tool_id, npub, *, tool_kwargs=None):
        refunds.append(tool_id)

    rt.rollback_debit = fake_rollback
    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    assert out["status"] == "error"
    assert out["refunded"] is True
    assert refunds == [TOOL_ID]


# ---------------------------------------------------------------------------
# #178 — never send closure_b64=None to Prefect (409 Conflict)
# ---------------------------------------------------------------------------

async def test_runner_only_kind_does_not_submit_null_closure_to_detached(
    vault_and_rt, monkeypatch
):
    """Detached executor installed + kind with ONLY a runner (no job_spec).

    Live shape of #178 / excalibur-mcp#341: PrefectClosureExecutor is wired
    (long-runner creds present) but the job kind never took the closure path,
    so start_async_job used to call submit(claim, None). Prefect's parameter
    schema requires closure_b64: string → every tick 409'd and silently fell
    back in-process. Detached execution never actually ran.

    The runner-only kind must run in-process without ever calling Prefect.
    """
    vault, rt = vault_and_rt
    captured: dict = {}
    _stub_prefect(monkeypatch, captured)

    # Detached executor is installed (the live posture after long-runner creds).
    rt.set_async_executor(
        PrefectClosureExecutor(
            deployment="dpyc-job-flow/dpyc-jobs",
            api_url="https://standalone",
            api_key="pnu_real",
            key_id="op16hex",
        )
    )

    ran = {"n": 0}

    async def runner(**params):
        ran["n"] += 1
        return {"ok": True, "prompt": params.get("prompt")}

    # Runner only — NO register_job_spec. _uses_closure_path is False.
    rt.register_job_runner("resolve", runner)
    assert rt._uses_closure_path("resolve") is False

    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )

    # Job accepted; Prefect was never contacted with a null closure.
    assert out["status"] == "pending"
    assert "claim_check" in out
    assert "parameters" not in captured, (
        "PrefectClosureExecutor.submit must not be called for a runner-only kind "
        f"(would have sent closure_b64={captured.get('parameters', {}).get('closure_b64')!r})"
    )
    # Not a degraded dispatch — runner-only under a detached executor is the
    # legitimate in-process path for that kind, not a failed Prefect hop.
    assert out.get("degraded") is None
    assert vault.rows[out["claim_check"]]["run_handle"] in (None, "")


async def test_prefect_submit_rejects_null_closure(monkeypatch):
    """Defense in depth: even a direct submit(None) must not reach Prefect.

    The schema on the deployment requires closure_b64: string; sending None
    costs a network hop and buries the real cause under a remote 409. Fail
    fast locally with a clear error instead.
    """
    captured: dict = {}
    _stub_prefect(monkeypatch, captured)

    exe = PrefectClosureExecutor(
        deployment="dpyc-job-flow/dpyc-jobs",
        api_url="https://standalone",
        api_key="pnu_real",
        key_id="op16hex",
    )
    with pytest.raises(ValueError, match="closure_b64"):
        await exe.submit("claim-x", None)
    with pytest.raises(ValueError, match="closure_b64"):
        await exe.submit("claim-x", "")
    assert "parameters" not in captured, "null/empty closure must not call run_deployment"


async def test_build_closure_returning_none_does_not_submit_null(vault_and_rt, monkeypatch):
    """Issue #178 candidate 2: build_closure yields nothing → no Prefect hop.

    Sealing/submitting a null spec used to either encrypt JSON null or send
    closure_b64=None. Now build must return a dict; otherwise dispatch fails
    locally (and falls back to the in-process runner when one exists) without
    a remote 409.
    """
    vault, rt = vault_and_rt
    captured: dict = {}
    _stub_prefect(monkeypatch, captured)

    async def build_closure(**params):
        return None  # nothing to seal

    def shape_result(raw, params):
        return raw

    rt.register_job_spec("resolve", build_closure, shape_result)

    ran = {"n": 0}

    async def runner(**params):
        ran["n"] += 1
        return {"ok": True}

    rt.register_job_runner("resolve", runner)
    rt.set_async_executor(
        PrefectClosureExecutor(
            deployment="dpyc-job-flow/dpyc-jobs",
            api_url="https://standalone",
            api_key="pnu_real",
            key_id="op16hex",
        )
    )

    out = await rt.start_async_job(
        "resolve", NPUB, {"prompt": "hi"},
        tool_id=TOOL_ID, max_runtime_seconds=210, result_ttl_seconds=900,
    )
    assert out["status"] == "pending"
    assert out.get("degraded") == "in_process_fallback"
    assert "dict" in (out.get("degraded_reason") or "").lower() or "build_closure" in (
        out.get("degraded_reason") or ""
    )
    assert "parameters" not in captured, "null build must not reach Prefect"
    assert vault.rows[out["claim_check"]]["run_handle"] in (None, "")


# ---------------------------------------------------------------------------
# PrefectClosureExecutor.submit — only ciphertext crosses (stubbed prefect)
# ---------------------------------------------------------------------------

def _stub_prefect(monkeypatch, captured):
    """Stub prefect.deployments + prefect.settings so submit() runs without a
    real Prefect install. ``temporary_settings`` is faked as a no-op CM that
    records the API URL/key updates so a test can assert submit targets the
    operator's standalone account (NOT the host platform's ambient env)."""
    import contextlib

    async def fake_run_deployment(*, name, parameters, timeout):
        captured["name"] = name
        captured["parameters"] = parameters
        captured["timeout"] = timeout
        return types.SimpleNamespace(id=uuid.uuid4())

    @contextlib.contextmanager
    def fake_temporary_settings(updates=None):
        captured["settings_updates"] = updates
        yield

    fake_pkg = types.ModuleType("prefect")
    fake_dep = types.ModuleType("prefect.deployments")
    fake_dep.run_deployment = fake_run_deployment
    fake_settings = types.ModuleType("prefect.settings")
    fake_settings.temporary_settings = fake_temporary_settings
    # The real settings objects are keys; sentinels suffice for the stub.
    fake_settings.PREFECT_API_URL = "PREFECT_API_URL"
    fake_settings.PREFECT_API_KEY = "PREFECT_API_KEY"
    monkeypatch.setitem(sys.modules, "prefect", fake_pkg)
    monkeypatch.setitem(sys.modules, "prefect.deployments", fake_dep)
    monkeypatch.setitem(sys.modules, "prefect.settings", fake_settings)


async def test_prefect_submit_passes_only_ciphertext(monkeypatch):
    captured: dict = {}
    _stub_prefect(monkeypatch, captured)

    exe = PrefectClosureExecutor(
        deployment="dpyc-job-flow/dpyc-jobs", api_url="u", api_key="k", key_id="op16hex"
    )
    sealed = VaultCipher(KEY_HEX).encrypt(
        json.dumps({"op": "http_request", "secret": ANTHROPIC_SECRET}),
        aad="dpyc-closure/v1",
    )
    handle = await exe.submit("claim-x", sealed)

    assert handle  # a flow-run id string
    assert captured["name"] == "dpyc-job-flow/dpyc-jobs"
    assert captured["timeout"] == 0
    # only ciphertext + the non-secret key_id selector cross
    assert captured["parameters"] == {"closure_b64": sealed, "key_id": "op16hex"}
    # the secret never appears in cleartext anywhere in the parameters
    assert ANTHROPIC_SECRET not in json.dumps(captured["parameters"])


async def test_prefect_submit_targets_standalone_account(monkeypatch):
    """submit() must force the operator's vaulted API URL/key via
    temporary_settings — NOT rely on the host platform's ambient PREFECT_* env
    (which points at a different account and would 401 → in-process fallback)."""
    captured: dict = {}
    _stub_prefect(monkeypatch, captured)

    exe = PrefectClosureExecutor(
        deployment="d/x", api_url="https://standalone", api_key="pnu_real", key_id="kid"
    )
    await exe.submit("claim-x", "sealed")

    # the settings override carried THIS operator's creds into run_deployment
    assert captured["settings_updates"] == {
        "PREFECT_API_URL": "https://standalone",
        "PREFECT_API_KEY": "pnu_real",
    }


# ---------------------------------------------------------------------------
# Generic DRY layer: per-operator key_id + automatic long-runner-creds wiring
# ---------------------------------------------------------------------------

def test_durable_key_id_is_deterministic_public_selector():
    rt = _make_runtime(FakeVault())
    rt.operator_npub = lambda: NPUB
    kid = rt.durable_key_id()
    assert kid == rt.durable_key_id()  # deterministic
    assert len(kid) == 16 and all(c in "0123456789abcdef" for c in kid)
    # the npub is public; the selector leaks nothing beyond it
    assert NPUB not in kid
    # a different operator names a different Secret block
    other = _make_runtime(FakeVault())
    other.operator_npub = lambda: _PK().public_key.bech32()
    assert other.durable_key_id() != kid


async def test_auto_resolve_enables_prefect_when_longrunner_creds_present():
    rt = _make_runtime(FakeVault())
    rt.operator_npub = lambda: NPUB

    async def creds(field_names, *, service=None):
        # long-runner creds are normal operator secrets — loaded from the
        # default operator service (service=None), not a separate one.
        return {
            "prefect_api_url": "https://api.prefect.cloud/x",
            "prefect_api_key": "pk",
            "closure_seal_key": KEY_HEX,
        }

    rt.load_credentials = creds
    _register_http_spec(rt)

    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, PrefectClosureExecutor)
    assert rt._async_executor._key_id == rt.durable_key_id()
    assert rt._async_executor._deployment == rt._DURABLE_DEPLOYMENT
    assert rt._uses_closure_path("resolve") is True
    # the probe is one-shot
    assert rt._async_executor_resolved is True


async def test_auto_resolve_stays_in_process_without_creds():
    rt = _make_runtime(FakeVault())
    rt.operator_npub = lambda: NPUB

    async def creds(field_names, *, service=None):
        return {}

    rt.load_credentials = creds
    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, InProcessExecutor)
    # a definitive "no creds" answer IS cached — no point re-probing every job
    assert rt._async_executor_resolved is True


async def test_auto_resolve_retries_after_transient_creds_failure():
    """A cold-vault hiccup on the first job must NOT pin the container to
    in-process for life. The probe stays unresolved on a load EXCEPTION and
    installs the detached executor on the next job once the vault is reachable."""
    rt = _make_runtime(FakeVault())
    rt.operator_npub = lambda: NPUB
    calls = {"n": 0}

    async def creds(field_names, *, service=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("neon warming up")  # transient cold-vault throw
        return {
            "prefect_api_url": "https://api.prefect.cloud/x",
            "prefect_api_key": "pk",
            "closure_seal_key": KEY_HEX,
        }

    rt.load_credentials = creds
    _register_http_spec(rt)

    # First job: transient failure — stays in-process AND unresolved (will retry).
    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, InProcessExecutor)
    assert rt._async_executor_resolved is False
    assert rt._uses_closure_path("resolve") is False

    # Next job: vault reachable — the detached executor now installs.
    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, PrefectClosureExecutor)
    assert rt._async_executor_resolved is True
    assert rt._uses_closure_path("resolve") is True


async def test_auto_resolve_degrades_gracefully_without_prefect_extra(monkeypatch):
    """Creds vaulted but the [prefect] extra absent must NOT crash the drill.

    Constructing the executor raises ImportError (prefect not installed); the
    runtime must fall back to in-process, cache the (definitive) resolution,
    and record an actionable error for service_status — never propagate."""
    import tollbooth.async_executor as ae

    class _NoPrefect:
        def __init__(self, *a, **k):
            raise ImportError("No module named 'prefect'")

    monkeypatch.setattr(ae, "PrefectClosureExecutor", _NoPrefect)

    rt = _make_runtime(FakeVault())
    rt.operator_npub = lambda: NPUB

    async def creds(field_names, *, service=None):
        return {
            "prefect_api_url": "https://api.prefect.cloud/x",
            "prefect_api_key": "pk",
            "closure_seal_key": KEY_HEX,
        }

    rt.load_credentials = creds
    _register_http_spec(rt)

    # Must not raise even though executor construction fails.
    await rt._ensure_async_executor()
    assert isinstance(rt._async_executor, InProcessExecutor)
    assert rt._uses_closure_path("resolve") is False
    # Definitive (a missing extra won't change without a redeploy) — cache it.
    assert rt._async_executor_resolved is True
    assert rt._async_executor_error is not None
    assert "prefect" in rt._async_executor_error.lower()


async def test_explicit_executor_disables_auto_resolve():
    rt = _make_runtime(FakeVault())
    rt.operator_npub = lambda: NPUB
    sentinel = RecordingExecutor()
    rt.set_async_executor(sentinel)

    async def creds(field_names, *, service=None):
        return {
            "prefect_api_url": "u",
            "prefect_api_key": "k",
            "closure_seal_key": KEY_HEX,
        }

    rt.load_credentials = creds
    await rt._ensure_async_executor()
    assert rt._async_executor is sentinel  # an explicit choice always wins


@pytest.fixture
def vault_and_rt():
    v = FakeVault()
    return v, _make_runtime(v)
