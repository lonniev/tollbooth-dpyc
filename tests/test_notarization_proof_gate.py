"""get_notarization_proof answers only the npub's own connection (scan F29).

Whether an npub is in a snapshot says whether it is a patron of this operator,
so a stranger must not be able to ask.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.ots import MerkleTree
from tollbooth.runtime import OperatorRuntime, register_standard_tools

PATRON = PrivateKey().public_key.bech32()
OTHER = PrivateKey().public_key.bech32()


def _tools(vault):
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    rt._ots_enabled = True
    rt._proof_grant_revocations = MagicMock()
    rt.vault = AsyncMock(return_value=vault)
    tools: dict = {}

    def fake_slug_tool(_mcp, _slug, _identify):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        register_standard_tools(MagicMock(), "test", rt, service_name="test")
    return tools


def _vault():
    tree = MerkleTree([(PATRON, "{\"balance_api_sats\": 5}"), (OTHER, "{}")])
    vault = MagicMock()
    vault.fetch_anchor = AsyncMock(return_value={
        "leaf_hashes_json": json.dumps(tree.get_leaf_hashes()),
        "root_hash": tree.root_hex,
        "status": "confirmed",
    })
    return vault


@pytest.mark.asyncio
async def test_a_stranger_learns_nothing_about_the_npub():
    vault = _vault()
    tools = _tools(vault)
    denied = AsyncMock(return_value={"success": False, "error_code": "proof_required"})
    with patch("tollbooth.runtime.require_held_grant", denied):
        r = await tools["get_notarization_proof"](notarization_id="n1", npub=PATRON)
    assert r["success"] is False and r["error_code"] == "proof_required"
    vault.fetch_anchor.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_proven_patron_gets_their_inclusion_proof():
    tools = _tools(_vault())
    with patch("tollbooth.runtime.require_held_grant", AsyncMock(return_value=None)):
        r = await tools["get_notarization_proof"](notarization_id="n1", npub=PATRON)
    assert r["success"] is True and r["verified"] is True
