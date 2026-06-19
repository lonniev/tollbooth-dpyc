"""check_price must agree with debit_or_deny on tool_not_priced.

Regression for a false-positive: check_price used to resolve the UUID from the
registry and return flat/0/success for a tool that wasn't in the active pricing
model — while the real paid call (debit_or_deny._resolve_pricing) denied it with
tool_not_priced. The two now agree: a non-free tool absent from (or unpriced in)
the loaded model previews as tool_not_priced.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tollbooth.constants import ErrorCode
from tollbooth.runtime import OperatorRuntime, register_standard_tools
from tollbooth.tool_identity import ToolIdentity, capability_uuid

WIDGET = capability_uuid("widget")
REGISTRY = {
    WIDGET: ToolIdentity(tool_id=WIDGET, capability="widget", category="read",
                         intent="A priced read tool"),
}


class _Resolver:
    """Minimal pricing resolver — check_price's gate returns before the
    constraint-preview block, so only these members are exercised."""

    def __init__(self, *, has: bool, priced: bool, cost: int = 0):
        self._has = has
        self._priced = priced
        self._cost = cost
        self._cached_model = None

    @property
    def neon_available(self) -> bool:
        return True

    async def _ensure_fresh(self) -> None:
        pass

    async def has_tool(self, tool_id: str) -> bool:
        return self._has

    async def is_priced(self, tool_id: str) -> bool:
        return self._priced

    async def get_tool_pricing(self, tool_id: str):
        from tollbooth.pricing import ToolPricing
        return ToolPricing(fixed=self._cost)


def _check_price_tool(resolver: _Resolver):
    rt = OperatorRuntime(tool_registry=REGISTRY, nsec_env_var="__UNUSED__")
    rt._pricing_resolver = resolver
    tools: dict = {}

    def fake_slug_tool(_mcp, _slug):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        register_standard_tools(MagicMock(), "test", rt, service_name="test")
    return tools["check_price"]


@pytest.mark.asyncio
async def test_absent_from_model_previews_not_priced():
    check_price = _check_price_tool(_Resolver(has=False, priced=False))
    r = await check_price(tool_id="widget")
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.TOOL_NOT_PRICED
    assert "not yet in the pricing model" in r["error"]


@pytest.mark.asyncio
async def test_present_but_unpriced_previews_tbd():
    check_price = _check_price_tool(_Resolver(has=True, priced=False))
    r = await check_price(tool_id="widget")
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.TOOL_NOT_PRICED
    assert "not been priced" in r["error"]


@pytest.mark.asyncio
async def test_priced_tool_previews_cost():
    check_price = _check_price_tool(_Resolver(has=True, priced=True, cost=5))
    r = await check_price(tool_id="widget")
    assert r["success"] is True
    assert r["effective_cost_api_sats"] == 5
