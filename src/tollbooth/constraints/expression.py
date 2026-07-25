"""Safe JSON expression constraint — evaluates structured expressions without eval().

Supports a tree of conditions expressed as JSON dicts.  The evaluator
walks the tree manually (no ``eval``, ``exec``, or ``compile``).

Expression grammar (JSON)
-------------------------

Leaf condition::

    {"field": "ledger.balance_api_sats", "op": ">=", "value": 1000}

Supported operators:
    ==, !=, >, <, >=, <=, in, contains, starts_with, matches

Logic combinators::

    {"and": [<condition>, ...]}
    {"or":  [<condition>, ...]}
    {"not": <condition>}

Full example::

    {
        "and": [
            {"field": "patron.membership_tier", "op": "==", "value": "gold"},
            {"or": [
                {"field": "ledger.total_consumed_api_sats", "op": ">=", "value": 5000},
                {"field": "env.invocation_count", "op": "<", "value": 10}
            ]}
        ]
    }
"""

from __future__ import annotations

import re
from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ConstraintSchema,
    ParamSchema,
    PriceModifier,
    ToolConstraint,
)

# ---------------------------------------------------------------------------
# Field resolver
# ---------------------------------------------------------------------------


def _resolve_field(context: ConstraintContext, field_path: str) -> Any:
    """Resolve a dotted field path like ``ledger.balance_api_sats``."""
    parts = field_path.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid field path: {field_path!r} (expected 'section.field')")

    section, attr = parts

    section_map = {
        "ledger": context.ledger,
        "patron": context.patron,
        "env": context.env,
    }

    obj = section_map.get(section)
    if obj is None:
        raise ValueError(
            f"Unknown section {section!r} in field path {field_path!r}. "
            f"Valid sections: {list(section_map)}"
        )

    if not hasattr(obj, attr):
        raise ValueError(
            f"Unknown attribute {attr!r} on {section!r} "
            f"(available: {[a for a in dir(obj) if not a.startswith('_')]})"
        )

    return getattr(obj, attr)


# ---------------------------------------------------------------------------
# Operator evaluator
# ---------------------------------------------------------------------------


def _eval_op(field_value: Any, op: str, test_value: Any) -> bool:
    """Evaluate a single comparison."""
    if op == "==":
        return field_value == test_value
    if op == "!=":
        return field_value != test_value
    if op == ">":
        return field_value > test_value
    if op == "<":
        return field_value < test_value
    if op == ">=":
        return field_value >= test_value
    if op == "<=":
        return field_value <= test_value
    if op == "in":
        # field_value in test_value  (test_value should be a list)
        return field_value in test_value
    if op == "contains":
        # test_value in field_value  (field_value is a string or collection)
        return test_value in field_value
    if op == "starts_with":
        return str(field_value).startswith(str(test_value))
    if op == "matches":
        return bool(re.search(str(test_value), str(field_value)))

    raise ValueError(f"Unknown operator: {op!r}")


# ---------------------------------------------------------------------------
# Tree evaluator
# ---------------------------------------------------------------------------


def _eval_node(context: ConstraintContext, node: dict[str, Any]) -> bool:
    """Recursively evaluate an expression node.  No eval() involved."""
    # Logic combinators
    if "and" in node:
        return all(_eval_node(context, child) for child in node["and"])
    if "or" in node:
        return any(_eval_node(context, child) for child in node["or"])
    if "not" in node:
        return not _eval_node(context, node["not"])

    # Leaf condition
    if "field" in node and "op" in node and "value" in node:
        field_value = _resolve_field(context, node["field"])
        return _eval_op(field_value, node["op"], node["value"])

    raise ValueError(f"Invalid expression node: {node!r}")


# ---------------------------------------------------------------------------
# JsonExpressionConstraint
# ---------------------------------------------------------------------------


class JsonExpressionConstraint(ToolConstraint):
    """Evaluate a safe JSON expression tree against the constraint context.

    Parameters
    ----------
    expression:
        JSON-compatible dict representing the condition tree.
    on_match:
        Action when the expression evaluates to ``True``.
        ``"allow"`` (default) — allow with optional price modifier.
        ``"deny"`` — deny the call.
    deny_reason:
        Machine-readable reason code when denying.
    deny_message:
        Human-readable message when denying.
    price_modifier:
        Optional :class:`PriceModifier` dict applied on match
        (only meaningful when *on_match* is ``"allow"``).
    """

    def __init__(
        self,
        expression: dict[str, Any],
        on_match: str = "allow",
        deny_reason: str = "expression_denied",
        deny_message: str = "Constraint expression not satisfied.",
        price_modifier: dict[str, Any] | None = None,
    ) -> None:
        self.expression = expression
        self.on_match = on_match
        self.deny_reason = deny_reason
        self.deny_message = deny_message
        self._price_modifier_dict = price_modifier

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        matched = _eval_node(context, self.expression)

        if self.on_match == "deny":
            # Expression defines when to DENY
            if matched:
                return ConstraintResult(
                    allowed=False,
                    reason=self.deny_reason,
                    message=self.deny_message,
                )
            return ConstraintResult(allowed=True)

        # on_match == "allow" (default):
        # Expression defines when to ALLOW (with optional modifier)
        if matched:
            modifier = (
                PriceModifier.from_dict(self._price_modifier_dict)
                if self._price_modifier_dict
                else None
            )
            return ConstraintResult(
                allowed=True,
                price_modifier=modifier,
            )

        # Expression not matched — deny by default
        return ConstraintResult(
            allowed=False,
            reason=self.deny_reason,
            message=self.deny_message,
        )

    def describe(self) -> str:
        return f"JSON expression constraint ({self.on_match} on match)"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "json_expression",
            "expression": self.expression,
            "on_match": self.on_match,
            "deny_reason": self.deny_reason,
            "deny_message": self.deny_message,
        }
        if self._price_modifier_dict:
            d["price_modifier"] = self._price_modifier_dict
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JsonExpressionConstraint:
        return cls(
            expression=data["expression"],
            on_match=data.get("on_match", "allow"),
            deny_reason=data.get("deny_reason", "expression_denied"),
            deny_message=data.get("deny_message", "Constraint expression not satisfied."),
            price_modifier=data.get("price_modifier"),
        )

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="json_expression",
            category="Dynamic",
            description="Evaluate a safe JSON expression tree against the constraint context. Supports and/or/not logic with field comparisons.",
            params=[
                ParamSchema(name="expression", type="string", description="JSON dict representing the condition tree (and/or/not combinators with field/op/value leaves)."),
                ParamSchema(name="on_match", type="string", required=False, default="allow", description="Action when expression is true: 'allow' or 'deny'."),
                ParamSchema(name="deny_reason", type="string", required=False, default="expression_denied", description="Machine-readable denial reason code."),
                ParamSchema(name="deny_message", type="string", required=False, default="Constraint expression not satisfied.", description="Human-readable denial message."),
                ParamSchema(name="price_modifier", type="string", required=False, description="Optional PriceModifier dict applied on match (only for on_match='allow')."),
            ],
        )
