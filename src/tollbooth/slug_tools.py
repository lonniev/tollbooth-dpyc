"""Utility for slug-prefixed MCP tool registration."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def make_slug_tool(mcp_server: Any, slug: str) -> Callable[[F], F]:
    """Return a decorator that registers MCP tools with a slug prefix.

    Usage in server.py::

        from tollbooth.slug_tools import make_slug_tool
        mcp = FastMCP("my-server")
        tool = make_slug_tool(mcp, "myslug")

        @tool
        async def check_balance():
            ...
        # Registered as "myslug_check_balance"
    """

    def decorator(func: F) -> F:
        prefixed = f"{slug}_{func.__name__}"
        return mcp_server.tool(name=prefixed)(func)

    return decorator  # type: ignore[return-value]
