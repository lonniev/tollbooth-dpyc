"""Canonical tool-identity listing (audit M2.1c).

Extracted from the ``list_canonical_identities`` closure so the registry →
response shaping is testable without a runtime. The shim in ``runtime.py``
passes the tool registry, the runtime's ``mcp_name_for`` resolver, the
operator npub, and the ``paid_tool`` function-name map so live tools that
were decorated but never seeded into ``ToolIdentity`` still surface as
unregistered drift (issue #174).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def build_canonical_identities(
    registry: dict[str, Any],
    mcp_name_for: Callable[[str], str],
    operator_npub: str,
    *,
    paid_tool_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Shape the canonical (tool_id, mcp_name, category, intent, capability) list.

    The authoritative mapping a client UUID-joins against the stored pricing
    model — ``tool_id`` is stable, ``mcp_name`` reflects the current slug/func
    name. Pure: ``mcp_name_for`` is the only behavior dependency.

    ``paid_tool_names`` is the runtime's UUID → function-name map recorded by
    ``@paid_tool`` at decoration time. Any UUID present there but absent from
    ``registry`` is a live tool the dispatcher knows and debit_or_deny will
    refuse — include it as ``registered: false`` so Pricing Studio Reconcile
    can flag the drift instead of silently reporting clean.
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
            }
            items.append(entry)
            unregistered.append(entry)

    return {
        "success": True,
        "operator_npub": operator_npub,
        "count": len(items),
        "unregistered_count": len(unregistered),
        "unregistered": unregistered,
        "tools": items,
    }
