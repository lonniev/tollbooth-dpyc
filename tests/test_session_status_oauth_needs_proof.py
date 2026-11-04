"""session_status shows a patron's upstream OAuth only to a connection proving that npub (F9/F10).

check_oauth_status already demands proof, because which upstream accounts a
patron linked, and when their token expires, is theirs to see. session_status
answered the same thing for any npub anyone passed. Its readiness answer stays
free and unchanged; only the upstream_oauth block needs the proof.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pynostr.key import PrivateKey

from tests.test_session_status_orphan import _register_tools
from tollbooth.constants import ErrorCode

pytestmark = pytest.mark.asyncio

PATRON = PrivateKey().public_key.bech32()
TOKENS = {"access_token": "a", "refresh_token": "r", "access_token_expires_at": time.time() + 600}


def _ready_rt(proof):
    rt = MagicMock()
    rt.operator_npub = MagicMock(return_value="npub1op")
    rt._vault = object()
    rt.pricing_resolver = AsyncMock(return_value=SimpleNamespace(
        _ensure_fresh=AsyncMock(), neon_available=True,
    ))
    rt._oauth_provider = SimpleNamespace(service_name="schwab", refresh_enabled=True)
    rt.load_patron_session = AsyncMock(return_value=(dict(TOKENS), ""))
    rt.prove_caller = proof
    return rt


async def test_a_connection_proving_the_npub_sees_its_upstream_oauth():
    rt = _ready_rt(AsyncMock(return_value=None))
    r = await _register_tools(rt)["session_status"](patron_npub=PATRON)
    assert r["lifecycle"] == "ready"
    assert r["upstream_oauth"]["has_access_token"] is True
    rt.prove_caller.assert_awaited_once_with(PATRON)


async def test_an_unproven_connection_gets_readiness_only():
    rt = _ready_rt(AsyncMock(return_value={"success": False, "error_code": ErrorCode.PROOF_REQUIRED}))
    r = await _register_tools(rt)["session_status"](patron_npub=PATRON)
    assert r["lifecycle"] == "ready" and r["success"] is True
    assert "upstream_oauth" not in r and "upstream_oauth_unreadable" not in r
    rt.load_patron_session.assert_not_awaited()


async def test_a_proof_check_that_errors_gets_readiness_only():
    rt = _ready_rt(AsyncMock(side_effect=RuntimeError("wallet down")))
    r = await _register_tools(rt)["session_status"](patron_npub=PATRON)
    assert r["lifecycle"] == "ready"
    assert "upstream_oauth" not in r
    rt.load_patron_session.assert_not_awaited()


async def test_no_npub_asks_for_no_proof():
    rt = _ready_rt(AsyncMock(return_value=None))
    r = await _register_tools(rt)["session_status"]()
    assert r["lifecycle"] == "ready"
    rt.prove_caller.assert_not_awaited()
