"""Regression test for restore_neon_schema's credential-vault schema step.

Before this was fixed, restore_neon_schema read a non-existent
``rt._credential_vault`` attribute (always None), so the credential-vault
schema was silently never re-created during a restore. The fix builds a
NeonCredentialVault directly from the operator's NeonVault (mirroring the
PricingModelStore step). This pins that the credential schema step now runs.
"""

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


def _register(rt):
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
    fake_vault = MagicMock()
    fake_vault.ensure_schema = AsyncMock()
    rt.vault = AsyncMock(return_value=fake_vault)
    rt.require_caller_proof = AsyncMock(return_value=None)  # proof passes
    rt.operator_npub = MagicMock(return_value=OP)
    return rt, fake_vault


@pytest.mark.asyncio
async def test_restore_runs_credential_vault_ensure_schema():
    rt, fake_vault = _runtime()
    tools = _register(rt)

    pricing_cls = MagicMock()
    pricing_cls.return_value.ensure_schema = AsyncMock()
    cred_cls = MagicMock()
    cred_cls.return_value.ensure_schema = AsyncMock()

    with patch("tollbooth.pricing_store.PricingModelStore", pricing_cls), \
         patch("tollbooth.vaults.neon.NeonCredentialVault", cred_cls):
        r = await tools["restore_neon_schema"](proof="ok")

    assert r["success"] is True
    step_names = {s["step"] for s in r["steps"]}
    # The credential-vault step now runs (it never did under the dead getattr).
    assert "CredentialVault.ensure_schema" in step_names
    cred_step = next(s for s in r["steps"] if s["step"] == "CredentialVault.ensure_schema")
    assert cred_step["ok"] is True
    # Built from the same NeonVault we already hold, and schema ensured.
    cred_cls.assert_called_once_with(neon_vault=fake_vault)
    cred_cls.return_value.ensure_schema.assert_awaited_once()


@pytest.mark.asyncio
async def test_restore_reports_credential_step_failure_inline():
    rt, _ = _runtime()
    tools = _register(rt)

    pricing_cls = MagicMock()
    pricing_cls.return_value.ensure_schema = AsyncMock()
    cred_cls = MagicMock()
    cred_cls.return_value.ensure_schema = AsyncMock(side_effect=RuntimeError("grants missing"))

    with patch("tollbooth.pricing_store.PricingModelStore", pricing_cls), \
         patch("tollbooth.vaults.neon.NeonCredentialVault", cred_cls):
        r = await tools["restore_neon_schema"](proof="ok")

    cred_step = next(s for s in r["steps"] if s["step"] == "CredentialVault.ensure_schema")
    assert cred_step["ok"] is False
    assert "grants missing" in cred_step["error"]
    assert r["success"] is False  # a failed step makes the whole run fail
