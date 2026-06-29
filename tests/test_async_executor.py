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

    def shape_result(raw):
        return {"x_text": (raw or {}).get("text", "")}

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
    with pytest.raises(Exception):
        cipher.decrypt(sealed, aad="wrong")
    # wrong key fails
    with pytest.raises(Exception):
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
    assert fetched["result"] == {"x_text": "woven answer"}
    assert vault.rows[claim]["status"] == "done"


async def test_fetch_closure_failed_refunds(vault_and_rt):
    vault, rt = vault_and_rt
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

    def shape_result(raw):
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
    vault, rt = vault_and_rt
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
    vault, rt = vault_and_rt
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
# Generic DRY layer: per-operator key_id + automatic dpyc-longrunner wiring
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
        if service == rt._LONGRUNNER_SERVICE:
            return {
                "prefect_api_url": "https://api.prefect.cloud/x",
                "prefect_api_key": "pk",
                "closure_seal_key": KEY_HEX,
            }
        return {}

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
