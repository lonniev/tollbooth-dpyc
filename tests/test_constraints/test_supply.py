"""Tests for tollbooth.constraints.supply — FiniteSupplyConstraint."""

from datetime import datetime, timezone

import pytest

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.supply import FiniteSupplyConstraint


def _ctx(invocation_count=0):
    return ConstraintContext(
        ledger=LedgerSnapshot(),
        patron=PatronIdentity(),
        env=EnvironmentSnapshot(
            utc_now=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
            invocation_count=invocation_count,
        ),
    )


class TestGlobalScope:
    def test_within_cap(self):
        c = FiniteSupplyConstraint(max_invocations=100, current_count=50, scope="global")
        result = c.evaluate(_ctx())
        assert result.allowed is True
        assert result.metadata["remaining_invocations"] == 50

    def test_at_cap(self):
        c = FiniteSupplyConstraint(max_invocations=100, current_count=100, scope="global")
        result = c.evaluate(_ctx())
        assert result.allowed is False
        assert result.reason == "supply_exhausted"
        assert result.metadata["remaining_invocations"] == 0

    def test_over_cap(self):
        c = FiniteSupplyConstraint(max_invocations=100, current_count=150, scope="global")
        result = c.evaluate(_ctx())
        assert result.allowed is False

    def test_zero_cap(self):
        c = FiniteSupplyConstraint(max_invocations=0, current_count=0, scope="global")
        result = c.evaluate(_ctx())
        assert result.allowed is False

    def test_one_remaining(self):
        c = FiniteSupplyConstraint(max_invocations=10, current_count=9, scope="global")
        result = c.evaluate(_ctx())
        assert result.allowed is True
        assert result.metadata["remaining_invocations"] == 1


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

    def test_ignores_current_count_in_per_patron(self):
        # current_count should be ignored when scope is per_patron
        c = FiniteSupplyConstraint(
            max_invocations=5, current_count=999, scope="per_patron"
        )
        result = c.evaluate(_ctx(invocation_count=2))
        assert result.allowed is True
        assert result.metadata["remaining_invocations"] == 3


class TestSerialization:
    def test_to_dict(self):
        c = FiniteSupplyConstraint(max_invocations=50, current_count=10, scope="global")
        d = c.to_dict()
        assert d["type"] == "finite_supply"
        assert d["max_invocations"] == 50
        assert d["current_count"] == 10
        assert d["scope"] == "global"

    def test_round_trip(self):
        c = FiniteSupplyConstraint(max_invocations=25, current_count=5, scope="per_patron")
        restored = FiniteSupplyConstraint.from_dict(c.to_dict())
        assert restored.max_invocations == 25
        assert restored.current_count == 5
        assert restored.scope == "per_patron"

    def test_from_dict_defaults(self):
        c = FiniteSupplyConstraint.from_dict({"max_invocations": 10})
        assert c.current_count == 0
        assert c.scope == "global"

    def test_describe(self):
        c = FiniteSupplyConstraint(max_invocations=100, scope="global")
        desc = c.describe()
        assert "100" in desc
        assert "global" in desc
