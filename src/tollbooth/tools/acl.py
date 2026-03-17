"""ACL management tools: set, get, and delete tool ACLs."""

from __future__ import annotations

import logging
from typing import Any

from tollbooth.acl_verify import verify_acl_event

logger = logging.getLogger(__name__)


async def set_tool_acl_tool(
    acl_store: Any,
    operator_npub: str,
    signed_acl_event_json: str,
) -> dict[str, Any]:
    """Validate and store a signed ACL event.

    Returns ``{"status": "ok", "tool_pattern": "...", "allowed_count": N}``.
    """
    # Verify the signed event before storing
    content = verify_acl_event(signed_acl_event_json, operator_npub)
    if content is None:
        return {"status": "error", "error": "Invalid or unauthorized ACL event"}

    tool_pattern = content["tool_pattern"]
    allowed_npubs = content["allowed_npubs"]

    try:
        await acl_store.ensure_schema()
        row_id = await acl_store.store_acl(
            operator_npub, tool_pattern, signed_acl_event_json,
        )
        return {
            "status": "ok",
            "tool_pattern": tool_pattern,
            "allowed_count": len(allowed_npubs),
            "acl_id": row_id,
        }
    except Exception as e:
        logger.exception("Failed to store ACL for %s", operator_npub)
        return {"status": "error", "error": str(e)}


async def get_tool_acls_tool(
    acl_store: Any,
    operator_npub: str,
) -> dict[str, Any]:
    """List all ACLs for an operator.

    Returns ``{"status": "ok", "acls": [...]}``.
    """
    try:
        acls = await acl_store.fetch_all_acls(operator_npub)
        return {"status": "ok", "acls": acls}
    except Exception as e:
        logger.exception("Failed to fetch ACLs for %s", operator_npub)
        return {"status": "error", "error": str(e)}


async def delete_tool_acl_tool(
    acl_store: Any,
    operator_npub: str,
    tool_pattern: str,
) -> dict[str, Any]:
    """Delete an ACL.

    Returns ``{"status": "ok", "deleted": true/false}``.
    """
    try:
        deleted = await acl_store.delete_acl(operator_npub, tool_pattern)
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        logger.exception("Failed to delete ACL for %s/%s", operator_npub, tool_pattern)
        return {"status": "error", "error": str(e)}
