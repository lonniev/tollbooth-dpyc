"""Unit tests for tollbooth.tools.identities.build_canonical_identities (M2.1c)."""

from __future__ import annotations

from types import SimpleNamespace

from tollbooth.tool_identity import capability_uuid
from tollbooth.tools.identities import build_canonical_identities


def _identity(category, intent, capability):
    return SimpleNamespace(category=category, intent=intent, capability=capability)


def test_empty_registry():
    r = build_canonical_identities({}, lambda tid: tid, "npub1op")
    assert r == {
        "success": True,
        "operator_npub": "npub1op",
        "count": 0,
        "unregistered_count": 0,
        "unregistered": [],
        "tools": [],
    }


def test_shapes_each_entry_and_resolves_mcp_name():
    registry = {
        "uuid-a": _identity("read", "Get a quote", "get_quote"),
        "uuid-b": _identity("write", "Post a tweet", "post_tweet"),
    }
    r = build_canonical_identities(registry, lambda tid: f"svc_{tid}", "npub1op")

    assert r["success"] is True and r["operator_npub"] == "npub1op" and r["count"] == 2
    by_id = {t["tool_id"]: t for t in r["tools"]}
    assert by_id["uuid-a"] == {
        "tool_id": "uuid-a",
        "mcp_name": "svc_uuid-a",
        "category": "read",
        "intent": "Get a quote",
        "capability": "get_quote",
        "registered": True,
    }
    assert by_id["uuid-b"]["mcp_name"] == "svc_uuid-b"
    assert by_id["uuid-b"]["capability"] == "post_tweet"
    assert by_id["uuid-b"]["registered"] is True
    assert r["unregistered_count"] == 0
    assert r["unregistered"] == []


def test_mcp_name_resolver_is_the_only_naming_authority():
    # tool_id stays; mcp_name is whatever the resolver returns (slug/func rename).
    registry = {"uuid-a": _identity("read", "x", "cap")}
    renamed = build_canonical_identities(registry, lambda tid: "newslug_cap", "op")
    assert renamed["tools"][0]["tool_id"] == "uuid-a"
    assert renamed["tools"][0]["mcp_name"] == "newslug_cap"


def test_paid_tool_uuid_missing_from_registry_surfaces_as_unregistered():
    """A UUID recorded by @paid_tool but absent from _tool_registry must
    appear as unregistered drift — not be silently omitted (#174).

    Domain handlers often decorate with @paid_tool(capability_uuid("X"))
    without a matching ToolIdentity seed. debit_or_deny then returns
    tool_not_registered, but list_canonical_identities only walked the
    registry — so Pricing Studio Reconcile reported clean and never
    surfaced the drift.
    """
    registry = {
        "uuid-seeded": _identity("read", "Seeded tool", "seeded_tool"),
    }
    paid_only_id = capability_uuid("post_performance")
    paid_func_names = {
        "uuid-seeded": "seeded_tool",
        paid_only_id: "post_performance",
    }

    def mcp_name_for(tid: str) -> str:
        # Mirror runtime.mcp_name_for fallback for unknown registry entries.
        fn = paid_func_names.get(tid)
        if fn:
            return f"excalibur_{fn}"
        return f"excalibur_{tid}"

    r = build_canonical_identities(
        registry,
        mcp_name_for,
        "npub1op",
        paid_tool_names=paid_func_names,
    )

    by_id = {t["tool_id"]: t for t in r["tools"]}
    assert "uuid-seeded" in by_id
    assert by_id["uuid-seeded"].get("registered", True) is True

    assert paid_only_id in by_id, (
        "paid-only UUID must appear in the listing so Reconcile can flag drift"
    )
    drift = by_id[paid_only_id]
    assert drift["registered"] is False
    assert drift["mcp_name"] == "excalibur_post_performance"
    assert drift["capability"] == "post_performance"
    assert r["count"] == 2
    assert r.get("unregistered_count") == 1
    assert paid_only_id in {t["tool_id"] for t in r.get("unregistered", [])}


def test_list_canonical_identities_uuid_is_capability_v5():
    """LIST_CANONICAL_IDENTITIES_UUID must be capability_uuid v5, not a hand-written v4."""
    import uuid as _uuid

    from tollbooth.tool_identity import LIST_CANONICAL_IDENTITIES_UUID

    expected = capability_uuid("list_canonical_identities")
    assert LIST_CANONICAL_IDENTITIES_UUID == expected
    assert _uuid.UUID(LIST_CANONICAL_IDENTITIES_UUID).version == 5


import pytest


@pytest.mark.asyncio
async def test_runtime_list_surfaces_paid_tool_not_in_registry():
    """End-to-end via OperatorRuntime: @paid_tool without a ToolIdentity seed
    must still appear in the canonical listing as unregistered drift (#174).
    """
    from tollbooth.constants import ErrorCode
    from tollbooth.runtime import OperatorRuntime
    from tollbooth.tool_identity import ToolIdentity
    from tollbooth.tools.identities import build_canonical_identities

    seeded_id = capability_uuid("seeded_tool")
    registry = {
        seeded_id: ToolIdentity(
            tool_id=seeded_id,
            category="read",
            intent="Seeded",
            capability="seeded_tool",
        ),
    }
    rt = OperatorRuntime(tool_registry=registry, nsec_env_var="__UNUSED__")
    rt._slug = "excalibur"

    paid_only_id = capability_uuid("post_performance")

    @rt.paid_tool(paid_only_id)
    async def post_performance(npub: str = "", dpop_token: str = "") -> dict:
        return {"ok": True}

    # debit_or_deny still refuses — that is the live failure mode
    denied = await rt.debit_or_deny(paid_only_id, "npub1test")
    assert denied["error_code"] == ErrorCode.TOOL_NOT_REGISTERED

    # But the listing now surfaces it so Reconcile can flag the drift
    assert rt.mcp_name_for(paid_only_id) == "excalibur_post_performance"
    r = build_canonical_identities(
        rt._tool_registry,
        rt.mcp_name_for,
        "npub1op",
        paid_tool_names=rt._tool_func_names,
    )
    by_id = {t["tool_id"]: t for t in r["tools"]}
    assert paid_only_id in by_id
    assert by_id[paid_only_id]["registered"] is False
    assert by_id[paid_only_id]["mcp_name"] == "excalibur_post_performance"
    assert r["unregistered_count"] == 1
