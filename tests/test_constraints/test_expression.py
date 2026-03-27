"""Tests for tollbooth.constraints.expression — JsonExpressionConstraint + safety."""

import ast
import inspect
from datetime import datetime, timezone

import pytest

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.expression import (
    JsonExpressionConstraint,
    _eval_node,
    _eval_op,
    _resolve_field,
)


def _ctx(
    balance=500,
    total_consumed=200,
    npub="npub1test",
    membership_tier="default",
    tool_name="my_tool",
    invocation_count=5,
):
    return ConstraintContext(
        ledger=LedgerSnapshot(
            balance_api_sats=balance,
            total_consumed_api_sats=total_consumed,
        ),
        patron=PatronIdentity(npub=npub, membership_tier=membership_tier),
        env=EnvironmentSnapshot(
            utc_now=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
            tool_name=tool_name,
            invocation_count=invocation_count,
        ),
    )


# ---------------------------------------------------------------------------
# Field resolution
# ---------------------------------------------------------------------------


class TestFieldResolution:
    def test_ledger_field(self):
        ctx = _ctx(balance=999)
        assert _resolve_field(ctx, "ledger.balance_api_sats") == 999

    def test_patron_field(self):
        ctx = _ctx(npub="npub1abc")
        assert _resolve_field(ctx, "patron.npub") == "npub1abc"

    def test_env_field(self):
        ctx = _ctx(tool_name="test_tool")
        assert _resolve_field(ctx, "env.tool_name") == "test_tool"

    def test_invalid_section(self):
        ctx = _ctx()
        with pytest.raises(ValueError, match="Unknown section"):
            _resolve_field(ctx, "invalid.field")

    def test_invalid_attribute(self):
        ctx = _ctx()
        with pytest.raises(ValueError, match="Unknown attribute"):
            _resolve_field(ctx, "ledger.nonexistent")

    def test_no_dot(self):
        ctx = _ctx()
        with pytest.raises(ValueError, match="Invalid field path"):
            _resolve_field(ctx, "nodot")


# ---------------------------------------------------------------------------
# Operator evaluation
# ---------------------------------------------------------------------------


class TestOperators:
    def test_eq(self):
        assert _eval_op(5, "==", 5) is True
        assert _eval_op(5, "==", 6) is False

    def test_ne(self):
        assert _eval_op(5, "!=", 6) is True
        assert _eval_op(5, "!=", 5) is False

    def test_gt(self):
        assert _eval_op(10, ">", 5) is True
        assert _eval_op(5, ">", 5) is False

    def test_lt(self):
        assert _eval_op(3, "<", 5) is True
        assert _eval_op(5, "<", 5) is False

    def test_gte(self):
        assert _eval_op(5, ">=", 5) is True
        assert _eval_op(6, ">=", 5) is True
        assert _eval_op(4, ">=", 5) is False

    def test_lte(self):
        assert _eval_op(5, "<=", 5) is True
        assert _eval_op(4, "<=", 5) is True
        assert _eval_op(6, "<=", 5) is False

    def test_in(self):
        assert _eval_op("gold", "in", ["gold", "platinum"]) is True
        assert _eval_op("bronze", "in", ["gold", "platinum"]) is False

    def test_contains(self):
        assert _eval_op("hello world", "contains", "world") is True
        assert _eval_op("hello", "contains", "xyz") is False

    def test_starts_with(self):
        assert _eval_op("npub1abc", "starts_with", "npub1") is True
        assert _eval_op("npub1abc", "starts_with", "nsec") is False

    def test_matches_regex(self):
        assert _eval_op("tool_v2", "matches", r"tool_v\d+") is True
        assert _eval_op("other", "matches", r"tool_v\d+") is False

    def test_unknown_operator(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            _eval_op(1, "xor", 2)


# ---------------------------------------------------------------------------
# Logic combinators
# ---------------------------------------------------------------------------


class TestLogicCombinators:
    def test_and_all_true(self):
        ctx = _ctx(balance=1000)
        node = {
            "and": [
                {"field": "ledger.balance_api_sats", "op": ">=", "value": 500},
                {"field": "ledger.balance_api_sats", "op": "<=", "value": 2000},
            ]
        }
        assert _eval_node(ctx, node) is True

    def test_and_one_false(self):
        ctx = _ctx(balance=100)
        node = {
            "and": [
                {"field": "ledger.balance_api_sats", "op": ">=", "value": 500},
                {"field": "ledger.balance_api_sats", "op": "<=", "value": 2000},
            ]
        }
        assert _eval_node(ctx, node) is False

    def test_or_one_true(self):
        ctx = _ctx(membership_tier="gold")
        node = {
            "or": [
                {"field": "patron.membership_tier", "op": "==", "value": "gold"},
                {"field": "patron.membership_tier", "op": "==", "value": "platinum"},
            ]
        }
        assert _eval_node(ctx, node) is True

    def test_or_none_true(self):
        ctx = _ctx(membership_tier="bronze")
        node = {
            "or": [
                {"field": "patron.membership_tier", "op": "==", "value": "gold"},
                {"field": "patron.membership_tier", "op": "==", "value": "platinum"},
            ]
        }
        assert _eval_node(ctx, node) is False

    def test_not(self):
        ctx = _ctx(balance=100)
        node = {"not": {"field": "ledger.balance_api_sats", "op": ">", "value": 500}}
        assert _eval_node(ctx, node) is True

    def test_nested(self):
        ctx = _ctx(balance=1000, membership_tier="gold")
        node = {
            "and": [
                {"field": "patron.membership_tier", "op": "==", "value": "gold"},
                {
                    "or": [
                        {"field": "ledger.balance_api_sats", "op": ">=", "value": 500},
                        {"field": "env.invocation_count", "op": "<", "value": 3},
                    ]
                },
            ]
        }
        assert _eval_node(ctx, node) is True

    def test_invalid_node(self):
        ctx = _ctx()
        with pytest.raises(ValueError, match="Invalid expression node"):
            _eval_node(ctx, {"invalid": "structure"})


# ---------------------------------------------------------------------------
# JsonExpressionConstraint integration
# ---------------------------------------------------------------------------


class TestJsonExpressionConstraint:
    def test_allow_on_match(self):
        c = JsonExpressionConstraint(
            expression={"field": "ledger.balance_api_sats", "op": ">=", "value": 100},
            on_match="allow",
        )
        result = c.evaluate(_ctx(balance=500))
        assert result.allowed is True

    def test_deny_when_not_matched(self):
        c = JsonExpressionConstraint(
            expression={"field": "ledger.balance_api_sats", "op": ">=", "value": 1000},
            on_match="allow",
            deny_reason="insufficient_balance",
            deny_message="Need at least 1000 sats.",
        )
        result = c.evaluate(_ctx(balance=500))
        assert result.allowed is False
        assert result.reason == "insufficient_balance"

    def test_deny_on_match(self):
        c = JsonExpressionConstraint(
            expression={"field": "patron.membership_tier", "op": "==", "value": "banned"},
            on_match="deny",
            deny_reason="banned",
            deny_message="Account is banned.",
        )
        result = c.evaluate(_ctx(membership_tier="banned"))
        assert result.allowed is False
        assert result.reason == "banned"

    def test_deny_mode_allows_when_not_matched(self):
        c = JsonExpressionConstraint(
            expression={"field": "patron.membership_tier", "op": "==", "value": "banned"},
            on_match="deny",
        )
        result = c.evaluate(_ctx(membership_tier="gold"))
        assert result.allowed is True

    def test_with_price_modifier(self):
        c = JsonExpressionConstraint(
            expression={"field": "patron.membership_tier", "op": "==", "value": "gold"},
            on_match="allow",
            price_modifier={"discount_percent": 20.0},
        )
        result = c.evaluate(_ctx(membership_tier="gold"))
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.discount_percent == 20.0

    def test_no_modifier_when_none(self):
        c = JsonExpressionConstraint(
            expression={"field": "ledger.balance_api_sats", "op": ">", "value": 0},
        )
        result = c.evaluate(_ctx(balance=100))
        assert result.price_modifier is None


class TestJsonExpressionSerialization:
    def test_to_dict(self):
        c = JsonExpressionConstraint(
            expression={"field": "ledger.balance_api_sats", "op": ">", "value": 0},
            on_match="allow",
            deny_reason="no_balance",
            deny_message="No balance.",
        )
        d = c.to_dict()
        assert d["type"] == "json_expression"
        assert d["expression"]["field"] == "ledger.balance_api_sats"
        assert d["on_match"] == "allow"

    def test_round_trip(self):
        c = JsonExpressionConstraint(
            expression={
                "and": [
                    {"field": "patron.npub", "op": "starts_with", "value": "npub1"},
                    {"field": "env.invocation_count", "op": "<", "value": 100},
                ]
            },
            on_match="deny",
            deny_reason="test",
            price_modifier={"discount_sats": 50},
        )
        restored = JsonExpressionConstraint.from_dict(c.to_dict())
        assert restored.expression == c.expression
        assert restored.on_match == "deny"
        assert restored._price_modifier_dict == {"discount_sats": 50}

    def test_describe(self):
        c = JsonExpressionConstraint(
            expression={"field": "x.y", "op": "==", "value": 1},
            on_match="deny",
        )
        assert "deny" in c.describe()


# ---------------------------------------------------------------------------
# INJECTION SAFETY TESTS
# ---------------------------------------------------------------------------


class TestInjectionSafety:
    """Verify that the expression module never uses eval(), exec(), or compile()."""

    def test_no_eval_in_source(self):
        """Statically verify no eval/exec/compile calls in expression.py."""
        import tollbooth.constraints.expression as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)

        dangerous_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr

                if name in ("eval", "exec", "compile"):
                    dangerous_calls.append(name)

        assert dangerous_calls == [], (
            f"Found dangerous calls in expression.py: {dangerous_calls}"
        )

    def test_malicious_field_path(self):
        """Ensure crafted field paths cannot escape the sandbox."""
        ctx = _ctx()
        with pytest.raises(ValueError):
            _resolve_field(ctx, "__class__.__bases__")

    def test_malicious_operator(self):
        """Ensure unknown operators raise, not execute."""
        with pytest.raises(ValueError, match="Unknown operator"):
            _eval_op("x", "__import__('os').system('ls')", "y")

    def test_deeply_nested_expression(self):
        """Ensure deeply nested expressions don't cause issues."""
        ctx = _ctx(balance=100)
        # Build a deeply nested AND (should still work)
        node: dict = {"field": "ledger.balance_api_sats", "op": ">", "value": 0}
        for _ in range(50):
            node = {"and": [node]}
        result = _eval_node(ctx, node)
        assert result is True

    def test_no_code_execution_in_matches(self):
        """Ensure regex 'matches' operator doesn't allow code injection."""
        ctx = _ctx(tool_name="safe_name")
        # This regex pattern should be treated as a pattern, not code
        node = {
            "field": "env.tool_name",
            "op": "matches",
            "value": "(?P<x>.*)",  # named groups are valid regex, not injection
        }
        result = _eval_node(ctx, node)
        assert result is True  # matches (it's valid regex)
