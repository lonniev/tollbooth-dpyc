"""Unit tests for tollbooth.tools.identities.build_canonical_identities (M2.1c)."""

from __future__ import annotations

from types import SimpleNamespace

from tollbooth.tools.identities import build_canonical_identities


def _identity(category, intent, capability):
    return SimpleNamespace(category=category, intent=intent, capability=capability)


def test_empty_registry():
    r = build_canonical_identities({}, lambda tid: tid, "npub1op")
    assert r == {"success": True, "operator_npub": "npub1op", "count": 0, "tools": []}


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
    }
    assert by_id["uuid-b"]["mcp_name"] == "svc_uuid-b"
    assert by_id["uuid-b"]["capability"] == "post_tweet"


def test_mcp_name_resolver_is_the_only_naming_authority():
    # tool_id stays; mcp_name is whatever the resolver returns (slug/func rename).
    registry = {"uuid-a": _identity("read", "x", "cap")}
    renamed = build_canonical_identities(registry, lambda tid: "newslug_cap", "op")
    assert renamed["tools"][0]["tool_id"] == "uuid-a"
    assert renamed["tools"][0]["mcp_name"] == "newslug_cap"
