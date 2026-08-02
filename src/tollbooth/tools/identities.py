"""Canonical tool-identity listing (audit M2.1c).

Extracted from the ``list_canonical_identities`` closure so the registry →
response shaping is testable without a runtime. The shim in ``runtime.py``
passes the tool registry, the runtime's ``mcp_name_for`` resolver, the
operator npub, and TWO independent views of what is actually live: the
``paid_tool`` function-name map (issue #174) and the FastMCP wire surface
(issue #175). They catch different drift and are both needed — see below.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def build_canonical_identities(
    registry: dict[str, Any],
    mcp_name_for: Callable[[str], str],
    operator_npub: str,
    *,
    paid_tool_names: Mapping[str, str] | None = None,
    live_mcp_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Shape the canonical (tool_id, mcp_name, category, intent, capability) list.

    The authoritative mapping a client UUID-joins against the stored pricing
    model — ``tool_id`` is stable, ``mcp_name`` reflects the current slug/func
    name. Pure: ``mcp_name_for`` is the only behavior dependency.

    Drift is detected from two sides, because neither alone is complete:

    ``paid_tool_names`` is the runtime's UUID → function-name map recorded by
    ``@paid_tool`` at decoration time. Any UUID present there but absent from
    ``registry`` is a live tool the dispatcher knows and debit_or_deny will
    refuse — reported with its tool_id and capability, which the wire name
    alone cannot supply.

    ``live_mcp_names`` is what FastMCP currently exposes. It catches anything
    on the wire regardless of how it got there — a plain FastMCP tool, a
    runtime-synthesized one — none of which appear in the paid_tool map.

    The wire pass reports only what the UUID pass did not already explain, so
    one drifting tool is named once, at the richest detail available.
    ``unregistered`` is always present (empty list when unknown or clean), and
    every entry carries ``mcp_name`` and ``reason``.
    """
    items: list[dict[str, Any]] = [
        {
            "tool_id": tool_id,
            "mcp_name": mcp_name_for(tool_id),
            "category": identity.category,
            "intent": identity.intent,
            "capability": identity.capability,
            "registered": True,
        }
        for tool_id, identity in registry.items()
    ]

    unregistered: list[dict[str, Any]] = []

    # Pass 1 — decorated but never seeded. Richest signal: we know the UUID.
    if paid_tool_names:
        for tool_id, func_name in paid_tool_names.items():
            if tool_id in registry:
                continue
            entry = {
                "tool_id": tool_id,
                "mcp_name": mcp_name_for(tool_id),
                "category": "",
                "intent": (
                    "Exposed by @paid_tool but missing from the ToolIdentity "
                    "registry — debit_or_deny will return tool_not_registered "
                    "until a ToolIdentity seed is added."
                ),
                "capability": func_name or "",
                "registered": False,
                "reason": "decorated_but_absent_from_registry",
            }
            items.append(entry)
            unregistered.append(entry)

    # Pass 2 — anything else on the wire. Computed AFTER pass 1 so a tool the
    # UUID pass already named is not reported a second time as a bare name.
    if live_mcp_names is not None:
        known_names = {item["mcp_name"] for item in items}
        seen: set[str] = set()
        for name in live_mcp_names:
            if not name or name in seen:
                continue
            seen.add(name)
            if name not in known_names:
                unregistered.append(
                    {
                        "mcp_name": name,
                        "reason": "exposed_on_wire_but_absent_from_registry",
                    }
                )

    return {
        "success": True,
        "operator_npub": operator_npub,
        "count": len(items),
        "unregistered_count": len(unregistered),
        "tools": items,
        "unregistered": unregistered,
    }
