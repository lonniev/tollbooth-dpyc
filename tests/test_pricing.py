"""Tests for ToolPricing."""

import pytest

from tollbooth.pricing import ToolPricing


class TestFixedPricing:
    def test_fixed_only(self):
        assert ToolPricing(fixed=5).compute() == 5

    def test_fixed_zero(self):
        assert ToolPricing(fixed=0).compute() == 0

    def test_fixed_with_min_cost(self):
        assert ToolPricing(fixed=1, min_cost=5).compute() == 5

    def test_fixed_above_min_cost(self):
        assert ToolPricing(fixed=10, min_cost=5).compute() == 10

    def test_composable_with_tool_tier(self):
        p = ToolPricing(fixed=5)  # WRITE tier was 5 sats
        assert p.compute() == 5


class TestRatePricing:
    def test_rate_only(self):
        p = ToolPricing(rate_percent=2.0, rate_param="amount_sats", min_cost=10)
        assert p.compute(amount_sats=50_000) == 1000

    def test_rate_with_min_cost_enforcement(self):
        p = ToolPricing(rate_percent=2.0, rate_param="amount_sats", min_cost=10)
        # 2% of 100 = 2, below min_cost of 10
        assert p.compute(amount_sats=100) == 10

    def test_rate_with_max_cost_cap(self):
        p = ToolPricing(rate_percent=10.0, rate_param="size", max_cost=50)
        # 10% of 1000 = 100, capped at 50
        assert p.compute(size=1000) == 50

    def test_rate_rounds_up(self):
        p = ToolPricing(rate_percent=3.0, rate_param="amount")
        # 3% of 10 = 0.3, ceil → 1
        assert p.compute(amount=10) == 1

    def test_zero_amount_returns_min(self):
        p = ToolPricing(rate_percent=2.0, rate_param="amount", min_cost=10)
        assert p.compute(amount=0) == 10

    def test_negative_amount_returns_min(self):
        p = ToolPricing(rate_percent=2.0, rate_param="amount", min_cost=5)
        assert p.compute(amount=-100) == 5


class TestHybridPricing:
    def test_fixed_plus_rate(self):
        p = ToolPricing(fixed=5, rate_percent=1.0, rate_param="size")
        # fixed 5 + 1% of 1000 = 5 + 10 = 15
        assert p.compute(size=1000) == 15

    def test_fixed_plus_rate_with_max(self):
        p = ToolPricing(fixed=5, rate_percent=10.0, rate_param="size", max_cost=20)
        # fixed 5 + 10% of 1000 = 105, capped at 20
        assert p.compute(size=1000) == 20

    def test_fixed_plus_rate_with_min(self):
        p = ToolPricing(fixed=5, rate_percent=1.0, rate_param="size", min_cost=15)
        # 1% of 100 = 1, below min 15, so rate_cost = 15, total = 5 + 15 = 20
        assert p.compute(size=100) == 20


class TestValidation:
    def test_missing_param_raises(self):
        p = ToolPricing(rate_percent=2.0, rate_param="amount_sats")
        with pytest.raises(ValueError, match="Missing required parameter"):
            p.compute()

    def test_non_numeric_param_raises(self):
        p = ToolPricing(rate_percent=2.0, rate_param="amount_sats")
        with pytest.raises(ValueError, match="must be numeric"):
            p.compute(amount_sats="not_a_number")

    def test_none_param_raises(self):
        p = ToolPricing(rate_percent=2.0, rate_param="amount_sats")
        with pytest.raises(ValueError, match="must be numeric"):
            p.compute(amount_sats=None)


class TestImmutability:
    def test_frozen(self):
        p = ToolPricing(fixed=5)
        with pytest.raises(AttributeError):
            p.fixed = 10  # type: ignore[misc]
