"""Tests for tollbooth.constraints.engine — ConstraintEngine logic modes."""

from datetime import datetime, timezone

import pytest

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
    PriceModifier,
    ToolConstraint,
)
from tollbooth.constraints.engine import ConstraintEngine, _stack_modifiers


# ---------------------------------------------------------------------------
# Stub constraints for testing
# ---------------------------------------------------------------------------


class AlwaysAllow(ToolConstraint):
    """Stub that always allows."""

    def __init__(self, modifier=None, metadata=None):
        self._modifier = modifier
        self._metadata = metadata or {}

    def evaluate(self, context):
        return ConstraintResult(
            allowed=True,
            price_modifier=self._modifier,
            metadata=self._metadata,
        )

    def describe(self):
        return "always allow"

    def to_dict(self):
        return {"type": "always_allow"}

    @classmethod
    def from_dict(cls, data):
        return cls()


class AlwaysDeny(ToolConstraint):
    """Stub that always denies."""

    def __init__(self, reason="denied", message="Denied."):
        self._reason = reason
        self._message = message

    def evaluate(self, context):
        return ConstraintResult(
            allowed=False,
            reason=self._reason,
            message=self._message,
        )

    def describe(self):
        return "always deny"

    def to_dict(self):
        return {"type": "always_deny"}

    @classmethod
    def from_dict(cls, data):
        return cls()


def _ctx():
    return ConstraintContext(
        ledger=LedgerSnapshot(),
        patron=PatronIdentity(),
        env=EnvironmentSnapshot(
            utc_now=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
        ),
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestEngineConstruction:
    def test_valid_logic_modes(self):
        for mode in ("ALL_MUST_PASS", "ANY_MUST_PASS", "FIRST_MATCH"):
            engine = ConstraintEngine(constraints={}, logic=mode)
            assert engine.logic == mode

    def test_invalid_logic_mode(self):
        with pytest.raises(ValueError, match="Invalid logic mode"):
            ConstraintEngine(constraints={}, logic="INVALID")


# ---------------------------------------------------------------------------
# No constraints
# ---------------------------------------------------------------------------


class TestNoConstraints:
    def test_no_constraints_for_tool(self):
        engine = ConstraintEngine(constraints={})
        result = engine.evaluate("any_tool", _ctx())
        assert result.allowed is True

    def test_constraints_for_other_tool_only(self):
        engine = ConstraintEngine(
            constraints={"other_tool": [AlwaysDeny()]}
        )
        result = engine.evaluate("my_tool", _ctx())
        assert result.allowed is True


# ---------------------------------------------------------------------------
# ALL_MUST_PASS
# ---------------------------------------------------------------------------


class TestAllMustPass:
    def test_all_allow(self):
        engine = ConstraintEngine(
            constraints={"tool": [AlwaysAllow(), AlwaysAllow()]},
            logic="ALL_MUST_PASS",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.allowed is True

    def test_one_denies(self):
        engine = ConstraintEngine(
            constraints={"tool": [AlwaysAllow(), AlwaysDeny(), AlwaysAllow()]},
            logic="ALL_MUST_PASS",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.allowed is False
        assert result.reason == "denied"

    def test_first_denies_short_circuits(self):
        engine = ConstraintEngine(
            constraints={"tool": [AlwaysDeny(reason="first"), AlwaysDeny(reason="second")]},
            logic="ALL_MUST_PASS",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.reason == "first"

    def test_price_modifiers_stack(self):
        m1 = PriceModifier(discount_percent=10.0)
        m2 = PriceModifier(discount_percent=20.0)
        engine = ConstraintEngine(
            constraints={"tool": [AlwaysAllow(modifier=m1), AlwaysAllow(modifier=m2)]},
            logic="ALL_MUST_PASS",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.allowed is True
        assert result.price_modifier is not None
        # Compound: 1 - (1-0.1)(1-0.2) = 1 - 0.72 = 0.28 = 28%
        assert abs(result.price_modifier.discount_percent - 28.0) < 0.01

    def test_metadata_merged(self):
        engine = ConstraintEngine(
            constraints={
                "tool": [
                    AlwaysAllow(metadata={"a": 1}),
                    AlwaysAllow(metadata={"b": 2}),
                ]
            },
            logic="ALL_MUST_PASS",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.metadata["a"] == 1
        assert result.metadata["b"] == 2


# ---------------------------------------------------------------------------
# ANY_MUST_PASS
# ---------------------------------------------------------------------------


class TestAnyMustPass:
    def test_one_allows(self):
        engine = ConstraintEngine(
            constraints={"tool": [AlwaysDeny(), AlwaysAllow(), AlwaysDeny()]},
            logic="ANY_MUST_PASS",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.allowed is True

    def test_all_deny(self):
        engine = ConstraintEngine(
            constraints={"tool": [AlwaysDeny(reason="a"), AlwaysDeny(reason="b")]},
            logic="ANY_MUST_PASS",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.allowed is False
        # Returns last denial
        assert result.reason == "b"

    def test_first_allows_short_circuits(self):
        engine = ConstraintEngine(
            constraints={"tool": [AlwaysAllow(metadata={"first": True}), AlwaysDeny()]},
            logic="ANY_MUST_PASS",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.allowed is True
        assert result.metadata.get("first") is True


# ---------------------------------------------------------------------------
# FIRST_MATCH
# ---------------------------------------------------------------------------


class TestFirstMatch:
    def test_first_deny(self):
        engine = ConstraintEngine(
            constraints={"tool": [AlwaysDeny(reason="first"), AlwaysAllow()]},
            logic="FIRST_MATCH",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.allowed is False
        assert result.reason == "first"

    def test_first_modifier(self):
        m = PriceModifier(discount_percent=50.0)
        engine = ConstraintEngine(
            constraints={
                "tool": [
                    AlwaysAllow(modifier=m),
                    AlwaysAllow(modifier=PriceModifier(free=True)),
                ]
            },
            logic="FIRST_MATCH",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.price_modifier.discount_percent == 50.0
        assert result.price_modifier.free is not True

    def test_all_neutral_allows(self):
        engine = ConstraintEngine(
            constraints={"tool": [AlwaysAllow(), AlwaysAllow()]},
            logic="FIRST_MATCH",
        )
        result = engine.evaluate("tool", _ctx())
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Wildcard constraints
# ---------------------------------------------------------------------------


class TestWildcard:
    def test_wildcard_applies_to_all_tools(self):
        engine = ConstraintEngine(
            constraints={"*": [AlwaysDeny(reason="wildcard")]},
            logic="ALL_MUST_PASS",
        )
        result = engine.evaluate("any_tool_name", _ctx())
        assert result.allowed is False
        assert result.reason == "wildcard"

    def test_wildcard_combined_with_tool_specific(self):
        engine = ConstraintEngine(
            constraints={
                "my_tool": [AlwaysAllow(metadata={"specific": True})],
                "*": [AlwaysAllow(metadata={"wildcard": True})],
            },
            logic="ALL_MUST_PASS",
        )
        result = engine.evaluate("my_tool", _ctx())
        assert result.allowed is True
        assert result.metadata["specific"] is True
        assert result.metadata["wildcard"] is True

    def test_wildcard_can_deny_specific_tool(self):
        engine = ConstraintEngine(
            constraints={
                "my_tool": [AlwaysAllow()],
                "*": [AlwaysDeny(reason="global_block")],
            },
            logic="ALL_MUST_PASS",
        )
        result = engine.evaluate("my_tool", _ctx())
        assert result.allowed is False


# ---------------------------------------------------------------------------
# _stack_modifiers
# ---------------------------------------------------------------------------


class TestStackModifiers:
    def test_none_plus_none(self):
        assert _stack_modifiers(None, None) is None

    def test_base_plus_none(self):
        m = PriceModifier(discount_percent=10.0)
        result = _stack_modifiers(m, None)
        assert result is m

    def test_none_plus_addition(self):
        m = PriceModifier(discount_sats=50)
        result = _stack_modifiers(None, m)
        assert result is m

    def test_compound_percentages(self):
        m1 = PriceModifier(discount_percent=10.0)
        m2 = PriceModifier(discount_percent=20.0)
        result = _stack_modifiers(m1, m2)
        # 1 - (1-0.1)(1-0.2) = 1 - 0.72 = 0.28 = 28%
        assert abs(result.discount_percent - 28.0) < 0.01

    def test_additive_sats(self):
        m1 = PriceModifier(discount_sats=100)
        m2 = PriceModifier(discount_sats=50)
        result = _stack_modifiers(m1, m2)
        assert result.discount_sats == 150

    def test_multiplicative_bonus(self):
        m1 = PriceModifier(bonus_multiplier=1.1)
        m2 = PriceModifier(bonus_multiplier=1.2)
        result = _stack_modifiers(m1, m2)
        assert abs(result.bonus_multiplier - 1.32) < 0.01

    def test_free_propagates(self):
        m1 = PriceModifier(free=True)
        m2 = PriceModifier(discount_percent=50.0)
        result = _stack_modifiers(m1, m2)
        assert result.free is True

    def test_free_from_addition(self):
        m1 = PriceModifier(discount_percent=50.0)
        m2 = PriceModifier(free=True)
        result = _stack_modifiers(m1, m2)
        assert result.free is True
