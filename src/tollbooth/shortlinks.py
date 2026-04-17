"""Utility for creating short URLs via the tollbooth-shortlinks MCP service."""

from __future__ import annotations

import json
import logging

import httpx

from tollbooth.registry import resolve_service_by_name

logger = logging.getLogger(__name__)

_FALLBACK_URL = "https://tollbooth-shortlinks.fastmcp.app/mcp/"


def _ensure_mcp_path(url: str) -> str:
    """Ensure the URL ends with /mcp/ without doubling it."""
    stripped = url.rstrip("/")
    if stripped.endswith("/mcp"):
        return stripped + "/"
    return stripped + "/mcp/"


async def _resolve_shortlinks_url() -> str:
    """Resolve the shortlinks service URL from the DPYC registry, falling back to hardcoded."""
    try:
        svc = await resolve_service_by_name("tollbooth-shortlinks")
        url = svc.get("url", "")
        if url:
            return _ensure_mcp_path(url)
    except Exception:
        pass
    return _FALLBACK_URL


async def create_shortlink(url: str, slug: str | None = None) -> str | None:
    """Create a short URL via tollbooth-shortlinks. Returns short_url or None on failure.

    Best-effort — returns None silently if the service is unavailable.
    """
    try:
        service_url = await _resolve_shortlinks_url()

        arguments: dict[str, str] = {"url": url}
        if slug is not None:
            arguments["slug"] = slug

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                service_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "create_shortlink",
                        "arguments": arguments,
                    },
                    "id": 1,
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                timeout=10,
            )

        if resp.status_code != 200:
            return None

        # Response may be SSE (text/event-stream) or plain JSON
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                content = data.get("result", {}).get("structuredContent", {})
                if content.get("success") and content.get("short_url"):
                    return str(content["short_url"])

        # Try plain JSON response
        try:
            data = resp.json()
            content = data.get("result", {}).get("structuredContent", {})
            if content.get("success") and content.get("short_url"):
                return str(content["short_url"])
        except (json.JSONDecodeError, ValueError):
            pass

        return None
    except Exception:
        logger.debug("shortlink creation failed for %s", url, exc_info=True)
        return None
