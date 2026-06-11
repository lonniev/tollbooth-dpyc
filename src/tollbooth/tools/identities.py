"""Canonical tool-identity listing (audit M2.1c).

Extracted from the ``list_canonical_identities`` closure so the registry →
response shaping is testable without a runtime. The shim in ``runtime.py``
passes the tool registry, the runtime's ``mcp_name_for`` resolver, and the
operator npub.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_canonical_identities(
    registry: dict[str, Any],
    mcp_name_for: Callable[[str], str],
    operator_npub: str,
) -> dict[str, Any]:
    """Shape the canonical (tool_id, mcp_name, category, intent, capability) list.

    The authoritative mapping a client UUID-joins against the stored pricing
    model — ``tool_id`` is stable, ``mcp_name`` reflects the current slug/func
    name. Pure: ``mcp_name_for`` is the only behavior dependency.
    """
    items: list[dict[str, Any]] = [
        {
            "tool_id": tool_id,
            "mcp_name": mcp_name_for(tool_id),
            "category": identity.category,
            "intent": identity.intent,
            "capability": identity.capability,
        }
        for tool_id, identity in registry.items()
    ]
    return {
        "success": True,
        "operator_npub": operator_npub,
        "count": len(items),
        "tools": items,
    }
