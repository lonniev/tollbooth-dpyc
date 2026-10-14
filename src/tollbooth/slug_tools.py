"""Utility for slug-prefixed MCP tool registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tollbooth.tool_identity import ToolIdentity

F = TypeVar("F", bound=Callable[..., Any])


def make_slug_tool(
    mcp_server: Any,
    slug: str,
    identify: Callable[[str], ToolIdentity],
) -> Callable[[F], F]:
    """Return a decorator that registers MCP tools with a slug prefix.

    Every tool is registered with the display title and behaviour hints its
    :class:`~tollbooth.tool_identity.ToolIdentity` declares. ``identify`` maps
    the decorated function's name to that identity, and raises when no
    identity declares it — a tool never goes on the wire undescribed.

    Usage in server.py::

        tool = make_slug_tool(mcp, "myslug", runtime.identity_for)

        @tool
        async def check_balance():
            ...
        # Registered as "myslug_check_balance", titled and annotated
    """

    def decorator(func: F) -> F:
        identity = identify(func.__name__)
        return mcp_server.tool(  # type: ignore[no-any-return]
            name=f"{slug}_{func.__name__}",
            title=identity.title,
            annotations=identity.behavior.annotations(),
        )(func)

    return decorator  # type: ignore[return-value]
