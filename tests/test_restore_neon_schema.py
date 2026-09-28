"""restore_neon_schema prepares the schema again, ignoring the breadcrumb,
and reports each concern's step — including Neon's own words for a failure."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.runtime import OperatorRuntime, register_standard_tools
from tollbooth.vaults.schema import SchemaPrepError

os.environ.setdefault(
    "TOLLBOOTH_NOSTR_OPERATOR_NSEC",
    "nsec1test000000000000000000000000000000000000000000000000000000",
)

OP = "npub1operatorXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


def _tools(rt):
    tools: dict = {}

    def fake_slug_tool(_mcp, _slug):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        register_standard_tools(MagicMock(), "test", rt, service_name="test")
    return tools


def _runtime():
    rt = OperatorRuntime(tool_registry={}, service_name="Test Operator")
    vault = MagicMock()
    rt.vault = AsyncMock(return_value=vault)
    rt.require_caller_proof = AsyncMock(return_value=None)
    rt.operator_npub = MagicMock(return_value=OP)
    return rt, vault


@pytest.mark.asyncio
async def test_restore_forces_a_full_prepare_and_reports_every_step():
    rt, vault = _runtime()
    steps = [{"step": "ledger", "ok": True}, {"step": "credentials", "ok": True}, {"step": "crumb", "ok": True}]
    prepare = AsyncMock(return_value=steps)
    with patch("tollbooth.vaults.schema.prepare_schema", prepare):
        r = await _tools(rt)["restore_neon_schema"](dpop_token="ok")
    prepare.assert_awaited_once_with(vault, force=True)
    assert r["success"] is True and r["steps"] == steps


@pytest.mark.asyncio
async def test_restore_reports_the_failing_step_inline():
    rt, _ = _runtime()
    steps = [{"step": "ledger", "ok": True},
             {"step": "credentials", "ok": False, "error_type": "NeonQueryError", "error": "grants missing"}]
    with patch("tollbooth.vaults.schema.prepare_schema", AsyncMock(side_effect=SchemaPrepError(steps))):
        r = await _tools(rt)["restore_neon_schema"](dpop_token="ok")
    assert r["success"] is False
    assert r["steps"][-1]["step"] == "credentials" and "grants missing" in r["steps"][-1]["error"]


@pytest.mark.asyncio
async def test_restore_requires_the_operators_proof():
    rt, _ = _runtime()
    r = await _tools(rt)["restore_neon_schema"](dpop_token="")
    assert r["success"] is False and "proof" in r["error"]
