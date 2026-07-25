"""Tests for tollbooth.constraints.supply — FiniteSupplyConstraint."""

from datetime import UTC, datetime

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.supply import FiniteSupplyConstraint


def _ctx(invocation_count=0, tool_name="my_tool", global_total=0):
    demand: tuple[tuple[str, int], ...] = ()
    if global_total:
        demand = ((f"{tool_name}:__total__", global_total),)
    return ConstraintContext(
        ledger=LedgerSnapshot(),
        patron=PatronIdentity(),
        env=EnvironmentSnapshot(
            utc_now=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
            tool_name=tool_name,
            invocation_count=invocation_count,
            global_demand=demand,
        ),
    )


class TestGlobalScope:
    def test_within_cap(self):
        c = FiniteSupplyConstraint(max_invocations=100, scope="global")
        result = c.evaluate(_ctx(global_total=50))
        assert result.allowed is True
        assert result.metadata["remaining_invocations"] == 50

    def test_at_cap(self):
        c = FiniteSupplyConstraint(max_invocations=100, scope="global")
        result = c.evaluate(_ctx(global_total=100))
        assert result.allowed is False
        assert result.reason == "supply_exhausted"
        assert result.metadata["remaining_invocations"] == 0

    def test_over_cap(self):
        c = FiniteSupplyConstraint(max_invocations=100, scope="global")
        result = c.evaluate(_ctx(global_total=150))
        assert result.allowed is False

    def test_zero_cap(self):
        c = FiniteSupplyConstraint(max_invocations=0, scope="global")
        result = c.evaluate(_ctx(global_total=0))
        assert result.allowed is False

    def test_one_remaining(self):
        c = FiniteSupplyConstraint(max_invocations=10, scope="global")
        result = c.evaluate(_ctx(global_total=9))
        assert result.allowed is True
        assert result.metadata["remaining_invocations"] == 1

    def test_no_demand_data_defaults_to_zero(self):
        """When no lifetime total is in env, count is 0 — full supply available."""
        c = FiniteSupplyConstraint(max_invocations=100, scope="global")
        result = c.evaluate(_ctx())  # global_total=0, no demand tuple entry
        assert result.allowed is True
        assert result.metadata["remaining_invocations"] == 100


class TestPerPatronScope:
    def test_uses_invocation_count(self):
        c = FiniteSupplyConstraint(max_invocations=5, scope="per_patron")
        result = c.evaluate(_ctx(invocation_count=3))
        assert result.allowed is True
        assert result.metadata["remaining_invocations"] == 2

    def test_patron_at_cap(self):
        c = FiniteSupplyConstraint(max_invocations=5, scope="per_patron")
        result = c.evaluate(_ctx(invocation_count=5))
        assert result.allowed is False
        assert result.reason == "supply_exhausted"

    def test_ignores_global_total_in_per_patron(self):
        c = FiniteSupplyConstraint(max_invocations=5, scope="per_patron")
        result = c.evaluate(_ctx(invocation_count=2, global_total=999))
        assert result.allowed is True
        assert result.metadata["remaining_invocations"] == 3


class TestSerialization:
    def test_to_dict(self):
        c = FiniteSupplyConstraint(max_invocations=50, scope="global")
        d = c.to_dict()
        assert d["type"] == "finite_supply"
        assert d["max_invocations"] == 50
        assert d["scope"] == "global"
        assert "current_count" not in d

    def test_round_trip(self):
        c = FiniteSupplyConstraint(max_invocations=25, scope="per_patron")
        restored = FiniteSupplyConstraint.from_dict(c.to_dict())
        assert restored.max_invocations == 25
        assert restored.scope == "per_patron"

    def test_from_dict_defaults(self):
        c = FiniteSupplyConstraint.from_dict({"max_invocations": 10})
        assert c.scope == "global"

    def test_from_dict_ignores_legacy_current_count(self):
        """Legacy pricing models with current_count are parsed without error."""
        c = FiniteSupplyConstraint.from_dict(
            {"max_invocations": 100, "current_count": 42, "scope": "global"}
        )
        assert c.max_invocations == 100
        assert c.scope == "global"

    def test_describe(self):
        c = FiniteSupplyConstraint(max_invocations=100, scope="global")
        desc = c.describe()
        assert "100" in desc
        assert "global" in desc
