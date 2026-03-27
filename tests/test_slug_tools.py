"""Tests for slug-prefixed MCP tool registration."""

from __future__ import annotations


from tollbooth.slug_tools import make_slug_tool


class _FakeToolManager:
    """Minimal stand-in for FastMCP tool registration."""

    def __init__(self) -> None:
        self.registered: dict[str, object] = {}

    def tool(self, *, name: str):
        def decorator(func):
            self.registered[name] = func
            return func
        return decorator


class TestMakeSlugTool:
    def test_prefixes_function_name(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "brain")

        @tool
        def check_balance():
            return "ok"

        assert "brain_check_balance" in fake.registered
        assert fake.registered["brain_check_balance"] is check_balance

    def test_different_slug(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "authority")

        @tool
        def certify_credits():
            return "cert"

        assert "authority_certify_credits" in fake.registered

    def test_async_function(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "excalibur")

        @tool
        async def post_tweet():
            return "posted"

        assert "excalibur_post_tweet" in fake.registered

    def test_preserves_original_function(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "brain")

        @tool
        def search_thoughts():
            return "found"

        # The decorated function should still be callable
        assert search_thoughts() == "found"

    def test_multiple_tools_same_slug(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "brain")

        @tool
        def tool_a():
            return "a"

        @tool
        def tool_b():
            return "b"

        assert "brain_tool_a" in fake.registered
        assert "brain_tool_b" in fake.registered
        assert len(fake.registered) == 2
