"""Tests for the deferred operator-adoption flow (courtship).

Authority side (receive/list/approve/reject/get_status) is driven through the
same tool-capture harness as test_authority_tools; the operator_requests store is
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
    )


def _passthrough_paid_tool(*_a, **_k):
    def deco(fn):
        return fn
    return deco


def _rt():
    return SimpleNamespace(
        identity_for=MagicMock(),
        operator_npub=MagicMock(return_value="npub1auth"),
        runtime_name=MagicMock(side_effect=lambda c: f"authority_{c}"),
        vault=AsyncMock(return_value=SimpleNamespace(_cipher=None)),
        proof_grant_revocations=AsyncMock(return_value=MagicMock()),
        paid_tool=_passthrough_paid_tool,
    )


@contextmanager
def _authority_tools(rt):
    tools: dict = {}

    def fake_slug_tool(_m, _s, _identify):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch.dict("os.environ"), patch.multiple(
        at,
        make_slug_tool=MagicMock(side_effect=fake_slug_tool),
        _get_settings=MagicMock(return_value=_settings()),
        _get_nostr_signer=MagicMock(return_value=SimpleNamespace(npub="npub1auth")),
        _dm_owner=AsyncMock(),
    ):
        os.environ.pop("NEON_DATABASE_URL", None)
        at.register_authority_tools(MagicMock(), rt)
        yield tools


# ── receive_adoption_request ──────────────────────────────────────────────


async def test_receive_rejects_an_unsigned_request():
    with _authority_tools(_rt()) as tools, patch.object(at, "_signed_by", MagicMock(return_value=False)):
        r = await tools["receive_adoption_request"](operator_npub="npub1op", service_url="https://svc")
    assert r["success"] is False and r["error_code"] == ErrorCode.PROOF_INVALID


async def test_receive_requires_operator_npub():
    with _authority_tools(_rt()) as tools:
        r = await tools["receive_adoption_request"](operator_npub="")
    assert r["success"] is False and "operator_npub is required" in r["error"]


async def test_receive_happy_records_pending_and_notifies():
    with _authority_tools(_rt()) as tools, patch.object(at, "_signed_by", MagicMock(return_value=True)), \
             patch.object(at.operator_requests, "ensure_schema", AsyncMock()), \
             patch.object(at.operator_requests, "upsert_pending", AsyncMock()) as upsert:
        r = await tools["receive_adoption_request"](
            operator_npub="npub1op", service_url="https://svc", display_name="Svc", note="hi",
        )
    assert r["success"] is True and r["status"] == "pending"
    upsert.assert_awaited_once()
    assert upsert.await_args.args[1:] == ("adoption", "npub1op")
    assert upsert.await_args.kwargs == {"service_url": "https://svc", "display_name": "Svc", "note": "hi"}


# ── list_operator_requests ────────────────────────────────────────────────


async def test_list_requires_authority_consent():
    with _authority_tools(_rt()) as tools, patch.object(at, "_require_owner",
                      AsyncMock(return_value={"success": False, "error": "consent"})):
        r = await tools["list_operator_requests"]()
    assert r["success"] is False and r["error"] == "consent"


async def test_list_returns_pending():
    pending = [{"operator_npub": "npub1op", "service_url": "u"}]
    with _authority_tools(_rt()) as tools:  # noqa: SIM117
        with patch.object(at, "_require_owner", AsyncMock(return_value=None)), \
             patch.object(at.operator_requests, "ensure_schema", AsyncMock()), \
             patch.object(at.operator_requests, "prune_expired", AsyncMock(return_value=0)), \
             patch.object(at.operator_requests, "list_pending", AsyncMock(return_value=pending)):
            r = await tools["list_operator_requests"]()
    assert r["success"] is True and r["count"] == 1 and r["requests"] == pending


# ── approve_operator_request ──────────────────────────────────────────────────────


async def test_approve_requires_consent():
    with _authority_tools(_rt()) as tools, patch.object(at, "_require_owner",
                      AsyncMock(return_value={"success": False, "error": "consent"})):
        r = await tools["approve_operator_request"](operator_npub="npub1op")
    assert r["error"] == "consent"


async def test_approve_not_found():
    with _authority_tools(_rt()) as tools:  # noqa: SIM117
        with patch.object(at, "_require_owner", AsyncMock(return_value=None)), \
             patch.object(at.operator_requests, "ensure_schema", AsyncMock()), \
             patch.object(at.operator_requests, "get", AsyncMock(return_value=None)):
            r = await tools["approve_operator_request"](operator_npub="npub1op")
    assert r["error_code"] == ErrorCode.ADOPTION_NOT_FOUND


async def test_approve_already_provisioned():
    with _authority_tools(_rt()) as tools:  # noqa: SIM117
        with patch.object(at, "_require_owner", AsyncMock(return_value=None)), \
             patch.object(at.operator_requests, "ensure_schema", AsyncMock()), \
             patch.object(at.operator_requests, "get",
                          AsyncMock(return_value={"status": "provisioned"})):
            r = await tools["approve_operator_request"](operator_npub="npub1op")
    assert r["error_code"] == ErrorCode.ADOPTION_ALREADY_PROVISIONED


async def test_approve_happy_provisions_via_shared_helper():
    provisioned = {"success": True, "npub": "npub1op", "neon_database_url": "postgresql://op"}
    with _authority_tools(_rt()) as tools:  # noqa: SIM117
        with patch.object(at, "_require_owner", AsyncMock(return_value=None)), \
             patch.object(at.operator_requests, "ensure_schema", AsyncMock()), \
             patch.object(at.operator_requests, "get",
                          AsyncMock(return_value={"status": "pending", "service_url": "https://svc"})), \
             patch.object(at.operator_requests, "mark", AsyncMock()) as mark, \
             patch.object(at, "_provision_operator",
                          AsyncMock(return_value=dict(provisioned))) as prov:
            r = await tools["approve_operator_request"](operator_npub="npub1op")
    assert r["success"] is True
    prov.assert_awaited_once()
    assert prov.await_args.args[1] == "npub1op" and prov.await_args.args[2] == "https://svc"
    mark.assert_awaited_once_with(mark.await_args.args[0], "adoption", "npub1op", "provisioned")


# ── reject_operator_request / get_adoption_status ─────────────────────────────────


async def test_reject_not_found_and_happy():
    with _authority_tools(_rt()) as tools:
        with patch.object(at, "_require_owner", AsyncMock(return_value=None)), \
             patch.object(at.operator_requests, "ensure_schema", AsyncMock()), \
             patch.object(at.operator_requests, "mark", AsyncMock(return_value=False)):
            miss = await tools["reject_operator_request"](operator_npub="npub1op")
        assert miss["error_code"] == ErrorCode.ADOPTION_NOT_FOUND

        with patch.object(at, "_require_owner", AsyncMock(return_value=None)), \
             patch.object(at.operator_requests, "ensure_schema", AsyncMock()), \
             patch.object(at.operator_requests, "mark", AsyncMock(return_value=True)):
            ok = await tools["reject_operator_request"](operator_npub="npub1op", reason="no")
        assert ok["success"] is True and ok["status"] == "rejected" and ok["reason"] == "no"


async def test_get_status_found_and_missing():
    with _authority_tools(_rt()) as tools:
        with patch.object(at.operator_requests, "ensure_schema", AsyncMock()), \
             patch.object(at.operator_requests, "get",
                          AsyncMock(return_value={"status": "pending"})):
            ok = await tools["get_adoption_status"](operator_npub="npub1op")
        assert ok["success"] is True and ok["status"] == "pending"

        with patch.object(at.operator_requests, "ensure_schema", AsyncMock()), \
             patch.object(at.operator_requests, "get", AsyncMock(return_value=None)):
            miss = await tools["get_adoption_status"](operator_npub="npub1op")
        assert miss["error_code"] == ErrorCode.ADOPTION_NOT_FOUND


# ── Operator side: request_adoption / adoption_status ─────────────────────


def _register_operator_tools(rt):
    import tollbooth.runtime as rtmod
    tools: dict = {}

    def fake_slug_tool(_m, _s, _identify):
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


async def test_request_adoption_needs_the_operators_signature():
    tools = _register_operator_tools(_operator_rt())
    with patch("tollbooth.holder_auth.signed_npub", return_value=None):
        r = await tools["request_adoption"](authority_npub="npub1auth")
    assert r["success"] is False and "sign the request" in r["error"]


async def test_a_signed_request_never_touches_the_vault():
    # Regression: an un-adopted orphan has no vault; the operator's own
    # signature is the whole gate, so asking for adoption cannot need one.
    tools = _register_operator_tools(_operator_rt())
    with patch("tollbooth.holder_auth.signed_npub", return_value="npub1op"):
        # Bad authority npub so the call returns at the guard right after the
        # gate — proving it was passed without the remote leg.
        r = await tools["request_adoption"](authority_npub="not-an-npub")
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
