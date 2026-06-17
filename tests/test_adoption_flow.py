"""Tests for the deferred operator-adoption flow (courtship).

Authority side (receive/list/approve/reject/get_status) is driven through the
same tool-capture harness as test_authority_tools; the adoption_store is
patched so these tests focus on gating + orchestration (the store has its own
unit tests). Operator side (request_adoption/adoption_status) is driven
through register_standard_tools with a faked FastMCP Client + registry.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import tollbooth.authority.tools as at
from tollbooth.constants import ErrorCode


# ── Authority-side harness ────────────────────────────────────────────────


def _settings():
    return SimpleNamespace(
        certificate_ttl_seconds=3600,
        dpyc_enforce_membership=False,
        neon_database_url="postgresql://x",
        dpyc_registry_cache_ttl_seconds=60,
        tollbooth_nostr_operator_nsec="nsec1x",
        tollbooth_nostr_relays=None,
    )


def _passthrough_paid_tool(*_a, **_k):
    def deco(fn):
        return fn
    return deco


def _rt():
    return SimpleNamespace(
        operator_npub=MagicMock(return_value="npub1auth"),
        runtime_name=MagicMock(side_effect=lambda c: f"authority_{c}"),
        vault=AsyncMock(return_value=SimpleNamespace(_cipher=None)),
        proven_npub_cache=AsyncMock(return_value=MagicMock()),
        paid_tool=_passthrough_paid_tool,
    )


@contextmanager
def _authority_tools(rt):
    tools: dict = {}

    def fake_slug_tool(_m, _s):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch.dict("os.environ"), patch.multiple(
        at,
        make_slug_tool=MagicMock(side_effect=fake_slug_tool),
        _get_settings=MagicMock(return_value=_settings()),
        _get_nostr_signer=MagicMock(return_value=SimpleNamespace(npub="npub1auth")),
        _get_nostr_exchange=MagicMock(
            return_value=SimpleNamespace(open_channel=AsyncMock())
        ),
    ):
        os.environ.pop("NEON_DATABASE_URL", None)
        at.register_authority_tools(MagicMock(), rt)
        yield tools


# ── receive_adoption_request ──────────────────────────────────────────────


async def test_receive_rejects_invalid_proof():
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "verify_proof", MagicMock(return_value=False)):
            r = await tools["receive_adoption_request"](
                operator_npub="npub1op", proof="bad", service_url="https://svc"
            )
    assert r["success"] is False and r["error_code"] == ErrorCode.PROOF_INVALID


async def test_receive_requires_operator_npub():
    with _authority_tools(_rt()) as tools:
        r = await tools["receive_adoption_request"](operator_npub="", proof="p")
    assert r["success"] is False and "operator_npub is required" in r["error"]


async def test_receive_happy_records_pending_and_notifies():
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "verify_proof", MagicMock(return_value=True)), \
             patch.object(at.adoption_store, "ensure_schema", AsyncMock()), \
             patch.object(at.adoption_store, "upsert_pending", AsyncMock()) as upsert:
            r = await tools["receive_adoption_request"](
                operator_npub="npub1op", proof="good", service_url="https://svc"
            )
    assert r["success"] is True and r["status"] == "pending"
    upsert.assert_awaited_once()
    assert upsert.await_args.args[1] == "npub1op"  # operator_npub
    assert upsert.await_args.args[2] == "https://svc"  # service_url


# ── list_adoption_requests ────────────────────────────────────────────────


async def test_list_requires_authority_consent():
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "_require_authority_consent",
                          AsyncMock(return_value={"success": False, "error": "consent"})):
            r = await tools["list_adoption_requests"](authority_proof="")
    assert r["success"] is False and r["error"] == "consent"


async def test_list_returns_pending():
    pending = [{"operator_npub": "npub1op", "service_url": "u"}]
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "_require_authority_consent", AsyncMock(return_value=None)), \
             patch.object(at.adoption_store, "ensure_schema", AsyncMock()), \
             patch.object(at.adoption_store, "prune_expired", AsyncMock(return_value=0)), \
             patch.object(at.adoption_store, "list_pending", AsyncMock(return_value=pending)):
            r = await tools["list_adoption_requests"](authority_proof="ap")
    assert r["success"] is True and r["count"] == 1 and r["requests"] == pending


# ── approve_adoption ──────────────────────────────────────────────────────


async def test_approve_requires_consent():
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "_require_authority_consent",
                          AsyncMock(return_value={"success": False, "error": "consent"})):
            r = await tools["approve_adoption"](operator_npub="npub1op", authority_proof="")
    assert r["error"] == "consent"


async def test_approve_not_found():
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "_require_authority_consent", AsyncMock(return_value=None)), \
             patch.object(at.adoption_store, "ensure_schema", AsyncMock()), \
             patch.object(at.adoption_store, "get", AsyncMock(return_value=None)):
            r = await tools["approve_adoption"](operator_npub="npub1op", authority_proof="ap")
    assert r["error_code"] == ErrorCode.ADOPTION_NOT_FOUND


async def test_approve_already_provisioned():
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "_require_authority_consent", AsyncMock(return_value=None)), \
             patch.object(at.adoption_store, "ensure_schema", AsyncMock()), \
             patch.object(at.adoption_store, "get",
                          AsyncMock(return_value={"status": "provisioned"})):
            r = await tools["approve_adoption"](operator_npub="npub1op", authority_proof="ap")
    assert r["error_code"] == ErrorCode.ADOPTION_ALREADY_PROVISIONED


async def test_approve_happy_provisions_via_shared_helper():
    provisioned = {"success": True, "npub": "npub1op", "neon_database_url": "postgresql://op"}
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "_require_authority_consent", AsyncMock(return_value=None)), \
             patch.object(at.adoption_store, "ensure_schema", AsyncMock()), \
             patch.object(at.adoption_store, "get",
                          AsyncMock(return_value={"status": "pending", "service_url": "https://svc"})), \
             patch.object(at.adoption_store, "mark", AsyncMock()) as mark, \
             patch.object(at, "_provision_operator",
                          AsyncMock(return_value=dict(provisioned))) as prov:
            r = await tools["approve_adoption"](operator_npub="npub1op", authority_proof="ap")
    assert r["success"] is True and r["adoption"] == "approved"
    prov.assert_awaited_once()
    assert prov.await_args.args[1] == "npub1op" and prov.await_args.args[2] == "https://svc"
    mark.assert_awaited_once_with(mark.await_args.args[0], "npub1op", "provisioned")


# ── reject_adoption / get_adoption_status ─────────────────────────────────


async def test_reject_not_found_and_happy():
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "_require_authority_consent", AsyncMock(return_value=None)), \
             patch.object(at.adoption_store, "ensure_schema", AsyncMock()), \
             patch.object(at.adoption_store, "mark", AsyncMock(return_value=False)):
            miss = await tools["reject_adoption"](operator_npub="npub1op", authority_proof="ap")
        assert miss["error_code"] == ErrorCode.ADOPTION_NOT_FOUND

        with patch.object(at, "_require_authority_consent", AsyncMock(return_value=None)), \
             patch.object(at.adoption_store, "ensure_schema", AsyncMock()), \
             patch.object(at.adoption_store, "mark", AsyncMock(return_value=True)):
            ok = await tools["reject_adoption"](operator_npub="npub1op", authority_proof="ap", reason="no")
        assert ok["success"] is True and ok["status"] == "rejected" and ok["reason"] == "no"


async def test_get_status_found_and_missing():
    with _authority_tools(_rt()) as tools:
        with patch.object(at.adoption_store, "ensure_schema", AsyncMock()), \
             patch.object(at.adoption_store, "get",
                          AsyncMock(return_value={"status": "pending"})):
            ok = await tools["get_adoption_status"](operator_npub="npub1op")
        assert ok["success"] is True and ok["status"] == "pending"

        with patch.object(at.adoption_store, "ensure_schema", AsyncMock()), \
             patch.object(at.adoption_store, "get", AsyncMock(return_value=None)):
            miss = await tools["get_adoption_status"](operator_npub="npub1op")
        assert miss["error_code"] == ErrorCode.ADOPTION_NOT_FOUND


# ── Operator side: request_adoption / adoption_status ─────────────────────


def _register_operator_tools(rt):
    import tollbooth.runtime as rtmod
    tools: dict = {}

    def fake_slug_tool(_m, _s):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        rtmod.register_standard_tools(MagicMock(), "test", rt, service_name="test")
    return tools


def _operator_rt():
    rt = MagicMock()
    rt.operator_npub = MagicMock(return_value="npub1op")
    rt.runtime_name = MagicMock(return_value="test_request_adoption")
    rt._get_nsec = MagicMock(return_value="nsec1operator")
    # An orphan has no vault — touching it is the regression we guard against.
    rt.vault = AsyncMock(side_effect=AssertionError("request_adoption must not touch the vault"))
    return rt


async def test_request_adoption_requires_proof():
    tools = _register_operator_tools(_operator_rt())
    r = await tools["request_adoption"](authority_npub="npub1auth", proof="")
    assert r["success"] is False and "provide proof" in r["error"]


async def test_request_adoption_verifies_proof_inline_without_vault():
    # Regression: request_adoption verifies the caller's proof INLINE
    # (proven_cache=None) so an un-adopted orphan — which has no vault — can
    # still request adoption. Previously it went through the vault-backed
    # proven-npub cache, forcing a bootstrap that an orphan cannot complete.
    captured: dict = {}

    async def fake_require_proof(npub, proof, tool_name, *, proven_cache=None, **kw):
        captured["proven_cache_is_none"] = proven_cache is None
        return None  # proof accepted

    tools = _register_operator_tools(_operator_rt())
    with patch("tollbooth.identity_proof.require_proof", side_effect=fake_require_proof):
        # Bad authority npub so the call returns at the guard right after the
        # proof check — proving the gate was passed without the remote leg.
        r = await tools["request_adoption"](authority_npub="not-an-npub", proof="p")

    assert captured.get("proven_cache_is_none") is True
    assert r["success"] is False and "valid npub1" in r["error"]


async def test_request_adoption_rejects_bad_authority_npub():
    # Proof accepted (patched); the bad-npub guard returns before the
    # MCP-to-MCP leg (which needs fastmcp, not an SDK test dep).
    tools = _register_operator_tools(_operator_rt())
    with patch("tollbooth.identity_proof.require_proof", AsyncMock(return_value=None)):
        r = await tools["request_adoption"](authority_npub="not-an-npub", proof="p")
    assert r["success"] is False and "valid npub1" in r["error"]


async def test_adoption_status_rejects_bad_authority_npub():
    tools = _register_operator_tools(_operator_rt())
    r = await tools["adoption_status"](authority_npub="not-an-npub")
    assert r["success"] is False and "valid npub1" in r["error"]

# NOTE: the request_adoption / adoption_status MCP-to-MCP happy paths are not
# unit-tested here because fastmcp is not an SDK test dependency — the operator
# only has it at deploy time. The remote-call plumbing (FastMCP Client +
# registry resolution by suffix) mirrors the proven dpyc-oracle list_services
# pattern; covered by live verification post-release.
