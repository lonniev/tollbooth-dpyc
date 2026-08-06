"""ModalExecutor — the detached path that needs no wire format.

Proven live on 2026-08-06 before this landed: a 200s three-block job spawned in
0.22s, ran on Modal (`ran_in_pid: 2`), and was claimed by id from a process that
had never seen the spawning one. These tests pin the contract that made that
work, without needing an account.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from tollbooth.async_executor import JobExecutor, ModalExecutor


def _fake_modal(*, spawn_id="fc-1", get=None, raises=None):
    """A stand-in for the `modal` module, installed into sys.modules."""
    mod = types.ModuleType("modal")
    call = MagicMock()
    call.object_id = spawn_id
    fn = MagicMock()
    fn.spawn.return_value = call
    mod.Function = MagicMock()
    mod.Function.from_name.return_value = fn

    handle = MagicMock()
    if raises is not None:
        handle.get.side_effect = raises
    else:
        handle.get.return_value = get
    mod.FunctionCall = MagicMock()
    mod.FunctionCall.from_id.return_value = handle
    mod._fn, mod._handle = fn, handle
    return mod


@pytest.fixture
def modal_mod(monkeypatch):
    mod = _fake_modal()
    monkeypatch.setitem(sys.modules, "modal", mod)
    return mod


def test_it_satisfies_the_executor_protocol():
    # The whole refactor rests on this: a drop-in for PrefectClosureExecutor,
    # so nothing in start_async_job / fetch_async_job has to change.
    assert isinstance(ModalExecutor(app_name="a"), JobExecutor)


@pytest.mark.asyncio
async def test_submit_spawns_the_claim_and_returns_modals_call_id(modal_mod):
    ex = ModalExecutor(app_name="excalibur-render", function_name="run_job")
    handle = await ex.submit("claim-123")
    modal_mod.Function.from_name.assert_called_once_with("excalibur-render", "run_job")
    modal_mod._fn.spawn.assert_called_once_with("claim-123")
    assert handle == "fc-1"


@pytest.mark.asyncio
async def test_only_the_claim_crosses_to_modal(monkeypatch):
    """The point of the refactor. Prefect REQUIRED a sealed closure and 409'd on
    None (#178); the seal, its key_id and a per-operator Secret block all existed
    because a generic flow could not run operator code. Modal runs the operator's
    own app, so a claim id is the entire payload — nothing else may cross."""
    mod = _fake_modal()
    monkeypatch.setitem(sys.modules, "modal", mod)
    ex = ModalExecutor(app_name="a")
    assert await ex.submit("claim-1") == "fc-1"
    assert await ex.submit("claim-2") == "fc-1"
    assert [c.args for c in mod._fn.spawn.call_args_list] == [("claim-1",), ("claim-2",)]
    for c in mod._fn.spawn.call_args_list:
        assert not c.kwargs, "spawn takes the claim and nothing else"


@pytest.mark.asyncio
async def test_poll_returns_none_while_the_job_is_still_running(monkeypatch):
    mod = _fake_modal(raises=TimeoutError("not done"))
    monkeypatch.setitem(sys.modules, "modal", mod)
    assert await ModalExecutor(app_name="a").poll("fc-1") is None


@pytest.mark.asyncio
async def test_poll_returns_the_result_when_the_job_has_finished(monkeypatch):
    mod = _fake_modal(get={"text": "resolved words"})
    monkeypatch.setitem(sys.modules, "modal", mod)
    out = await ModalExecutor(app_name="a").poll("fc-1")
    assert out == {"status": "completed", "result": {"text": "resolved words"}, "error": None}


@pytest.mark.asyncio
async def test_a_cancelled_run_is_a_distinguishable_failure_not_silence(monkeypatch):
    """Live behaviour: cancelling raised RemoteError("Function call was cancelled
    by user or a failure."). Silence here is what let a dead job look healthy."""
    class RemoteError(Exception):
        pass

    mod = _fake_modal(raises=RemoteError("Function call was cancelled by user or a failure."))
    monkeypatch.setitem(sys.modules, "modal", mod)
    out = await ModalExecutor(app_name="a").poll("fc-1")
    assert out is not None and out["status"] == "failed"
    assert "RemoteError" in out["error"] and "cancelled" in out["error"]
    assert out["result"] is None
