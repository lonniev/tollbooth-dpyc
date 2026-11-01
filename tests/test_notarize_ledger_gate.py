"""notarize_ledger is restricted: only the proven operator may run it."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.runtime import OperatorRuntime, register_standard_tools

os.environ.setdefault(
    "TOLLBOOTH_NOSTR_OPERATOR_NSEC",
    "nsec1test000000000000000000000000000000000000000000000000000000",
)

OP = "npub1operatorXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


def _tools(rt):
    tools: dict = {}

    def fake_slug_tool(_mcp, _slug, _identify):
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
async def test_a_caller_without_the_operators_proof_is_refused_before_any_work():
    rt, _ = _runtime()
    denial = {"success": False, "error_code": "proof_required", "error": "proof required"}
    rt.require_caller_proof = AsyncMock(return_value=denial)
    notarize = AsyncMock()
    with patch("tollbooth.tools.notarization.notarize_ledger_tool", notarize):
        r = await _tools(rt)["notarize_ledger"]()
    assert r == denial
    rt.require_caller_proof.assert_awaited_once_with(OP)
    notarize.assert_not_awaited()
    rt.vault.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_proven_operator_notarizes():
    rt, vault = _runtime()
    done = {"success": True, "leaf_count": 3}
    notarize = AsyncMock(return_value=done)
    with patch("tollbooth.tools.notarization.notarize_ledger_tool", notarize):
        r = await _tools(rt)["notarize_ledger"]()
    assert r == done
    notarize.assert_awaited_once_with(vault, ots_calendars=rt._ots_calendars)
