"""Tests for SurgePricingConstraint."""

from datetime import UTC, datetime

import pytest

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
    PriceModifier,
)
from tollbooth.constraints.surge import SurgePricingConstraint


def _ctx(tool: str = "fetch", demand: int = 0) -> ConstraintContext:
    return ConstraintContext(
        ledger=LedgerSnapshot(balance_api_sats=100),
        patron=PatronIdentity(npub="npub1test"),
        env=EnvironmentSnapshot(
            utc_now=datetime.now(UTC),
            tool_name=tool,
            global_demand=((tool, demand),),
        ),
    )


class TestSurgePricingConstraint:
    def test_no_surge_below_first_tier(self):
        c = SurgePricingConstraint.from_dict({
            "type": "surge_pricing",
            "max_capacity": 100,
            "tiers": [
                {"capacity_pct": 0.5, "multiplier": 1.5},
            ],
        })
        result = c.evaluate(_ctx(demand=10))
        assert result.allowed
        assert result.price_modifier is None

    def test_surge_at_threshold(self):
        c = SurgePricingConstraint.from_dict({
            "type": "surge_pricing",
            "max_capacity": 100,
            "tiers": [
                {"capacity_pct": 0.5, "multiplier": 1.5},
                {"capacity_pct": 0.8, "multiplier": 2.0},
            ],
        })
        result = c.evaluate(_ctx(demand=50))
        assert result.allowed
        assert result.price_modifier is not None
        assert result.price_modifier.surge_multiplier == 1.5

    def test_surge_highest_tier(self):
        c = SurgePricingConstraint.from_dict({
            "type": "surge_pricing",
            "max_capacity": 100,
            "tiers": [
                {"capacity_pct": 0.5, "multiplier": 1.5},
                {"capacity_pct": 0.8, "multiplier": 2.0},
            ],
        })
        result = c.evaluate(_ctx(demand=90))
        assert result.price_modifier.surge_multiplier == 2.0
        assert result.metadata["surge_utilization"] == 0.9

    def test_surge_over_capacity(self):
        c = SurgePricingConstraint.from_dict({
            "type": "surge_pricing",
            "max_capacity": 100,
            "tiers": [
                {"capacity_pct": 0.5, "multiplier": 1.5},
                {"capacity_pct": 0.95, "multiplier": 3.0},
            ],
        })
        result = c.evaluate(_ctx(demand=120))
        assert result.price_modifier.surge_multiplier == 3.0

    def test_no_demand_data_returns_neutral(self):
        c = SurgePricingConstraint.from_dict({
            "type": "surge_pricing",
            "max_capacity": 100,
            "tiers": [{"capacity_pct": 0.5, "multiplier": 2.0}],
        })
        # No global_demand for this tool
        ctx = ConstraintContext(
            ledger=LedgerSnapshot(),
            patron=PatronIdentity(),
            env=EnvironmentSnapshot(
                utc_now=datetime.now(UTC),
                tool_name="fetch",
            ),
        )
        result = c.evaluate(ctx)
        assert result.allowed
        assert result.price_modifier is None

    def test_never_denies(self):
        c = SurgePricingConstraint.from_dict({
            "type": "surge_pricing",
            "max_capacity": 10,
            "tiers": [{"capacity_pct": 0.1, "multiplier": 10.0}],
        })
        result = c.evaluate(_ctx(demand=1000))
        assert result.allowed is True

    def test_describe(self):
        c = SurgePricingConstraint.from_dict({
            "type": "surge_pricing",
            "max_capacity": 100,
            "window": "1h",
            "tiers": [{"capacity_pct": 0.8, "multiplier": 1.5}],
        })
        desc = c.describe()
        assert "100" in desc
        assert "1h" in desc

    def test_round_trip_serialization(self):
        original = {
            "type": "surge_pricing",
            "max_capacity": 200,
            "window": "15m",
            "tiers": [
                {"capacity_pct": 0.5, "multiplier": 1.0},
                {"capacity_pct": 0.8, "multiplier": 1.5},
            ],
        }
        c = SurgePricingConstraint.from_dict(original)
        d = c.to_dict()
        assert d["type"] == "surge_pricing"
        assert d["max_capacity"] == 200
        assert d["window"] == "15m"
        assert len(d["tiers"]) == 2

    def test_invalid_capacity_pct(self):
        with pytest.raises(ValueError, match="capacity_pct"):
            SurgePricingConstraint.from_dict({
                "type": "surge_pricing",
                "max_capacity": 100,
                "tiers": [{"capacity_pct": 1.5, "multiplier": 2.0}],
            })

    def test_invalid_zero_capacity(self):
        with pytest.raises(ValueError, match="max_capacity"):
            SurgePricingConstraint.from_dict({
                "type": "surge_pricing",
                "max_capacity": 0,
                "tiers": [{"capacity_pct": 0.5, "multiplier": 2.0}],
            })


class TestPriceModifierSurge:
    def test_surge_multiplier_increases_price(self):
        mod = PriceModifier(surge_multiplier=2.0)
        assert mod.apply_to(10) == 20

    def test_surge_with_discount_composes(self):
        # 2x surge then 50% discount = base price
        mod = PriceModifier(surge_multiplier=2.0, discount_percent=50.0)
        assert mod.apply_to(10) == 10

    def test_surge_serialization(self):
        mod = PriceModifier(surge_multiplier=1.5)
        d = mod.to_dict()
        assert d["surge_multiplier"] == 1.5
        restored = PriceModifier.from_dict(d)
        assert restored.surge_multiplier == 1.5

    def test_no_surge_not_serialized(self):
        mod = PriceModifier()
        d = mod.to_dict()
        assert "surge_multiplier" not in d

    def test_free_overrides_surge(self):
        mod = PriceModifier(surge_multiplier=10.0, free=True)
        assert mod.apply_to(100) == 0


class TestEnvironmentSnapshotDemand:
    def test_demand_for_known_tool(self):
        env = EnvironmentSnapshot(
            utc_now=datetime.now(UTC),
            tool_name="fetch",
            global_demand=(("fetch", 42), ("post", 10)),
        )
        assert env.demand_for("fetch") == 42
        assert env.demand_for("post") == 10

    def test_demand_for_unknown_tool(self):
        env = EnvironmentSnapshot(
            utc_now=datetime.now(UTC),
            global_demand=(("fetch", 42),),
        )
        assert env.demand_for("other") == 0

    def test_empty_demand(self):
        env = EnvironmentSnapshot(utc_now=datetime.now(UTC))
        assert env.demand_for("anything") == 0
