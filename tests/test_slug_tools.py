"""Tests for slug-prefixed MCP tool registration."""

from __future__ import annotations

import pytest

from tollbooth.slug_tools import make_slug_tool
from tollbooth.tool_identity import READS, ToolIdentity


def _identify(func_name: str) -> ToolIdentity:
    return ToolIdentity(
        tool_id=f"id-{func_name}", category="free", intent="test",
        title=func_name.replace("_", " ").capitalize(), behavior=READS,
        capability=func_name,
    )


class _FakeToolManager:
    """Minimal stand-in for FastMCP tool registration."""

    def __init__(self) -> None:
        self.registered: dict[str, object] = {}

        self.metadata: dict[str, dict] = {}

    def tool(self, *, name: str, title: str, annotations: dict):
        def decorator(func):
            self.registered[name] = func
            self.metadata[name] = {"title": title, "annotations": annotations}
            return func
        return decorator


class TestMakeSlugTool:
    def test_prefixes_function_name(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "brain", _identify)

        @tool
        def check_balance():
            return "ok"

        assert "brain_check_balance" in fake.registered
        assert fake.registered["brain_check_balance"] is check_balance

    def test_different_slug(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "authority", _identify)

        @tool
        def certify_credits():
            return "cert"

        assert "authority_certify_credits" in fake.registered

    def test_async_function(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "excalibur", _identify)

        @tool
        async def post_tweet():
            return "posted"

        assert "excalibur_post_tweet" in fake.registered

    def test_preserves_original_function(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "brain", _identify)

        @tool
        def search_thoughts():
            return "found"

        # The decorated function should still be callable
        assert search_thoughts() == "found"

    def test_multiple_tools_same_slug(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "brain", _identify)

        @tool
        def tool_a():
            return "a"

        @tool
        def tool_b():
            return "b"

        assert "brain_tool_a" in fake.registered
        assert "brain_tool_b" in fake.registered
        assert len(fake.registered) == 2


class TestToolMetadata:
    def test_registers_title_and_behavior(self) -> None:
        fake = _FakeToolManager()
        tool = make_slug_tool(fake, "brain", _identify)

        @tool
        def check_balance():
            return "ok"

        assert fake.metadata["brain_check_balance"] == {
            "title": "Check balance",
            "annotations": READS.annotations(),
        }

    def test_undeclared_tool_is_refused(self) -> None:
        fake = _FakeToolManager()

        def identify(func_name: str) -> ToolIdentity:
            raise LookupError(func_name)

        tool = make_slug_tool(fake, "brain", identify)
        with pytest.raises(LookupError):
            @tool
            def mystery():
                return "?"
        assert fake.registered == {}
