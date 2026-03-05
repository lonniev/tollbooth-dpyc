"""Constraint engine — evaluates a list of constraints for a tool call."""

from __future__ import annotations

from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    PriceModifier,
    ToolConstraint,
)


class ConstraintEngine:
    """Evaluate a set of constraints against a tool call context.

    Parameters
    ----------
    constraints:
        Mapping of tool name to list of constraints.
        The wildcard key ``"*"`` applies to all tools.
    logic:
        Default logic mode for combining constraints:

        ``"ALL_MUST_PASS"``
            Every constraint must return ``allowed=True``.
            First denial short-circuits.
        ``"ANY_MUST_PASS"``
            At least one constraint must return ``allowed=True``.
        ``"FIRST_MATCH"``
            Use the result of the first constraint that produces
            a non-neutral result (denial or price modifier).
    """

    VALID_LOGIC = {"ALL_MUST_PASS", "ANY_MUST_PASS", "FIRST_MATCH"}

    def __init__(
        self,
        constraints: dict[str, list[ToolConstraint]],
        logic: str = "ALL_MUST_PASS",
    ) -> None:
        if logic not in self.VALID_LOGIC:
            raise ValueError(
                f"Invalid logic mode {logic!r}. Must be one of {self.VALID_LOGIC}"
            )
        self.constraints = constraints
        self.logic = logic

    def evaluate(self, tool_name: str, context: ConstraintContext) -> ConstraintResult:
        """Evaluate all constraints applicable to *tool_name*.

        Applicable constraints are those registered under *tool_name*
        plus any registered under the wildcard ``"*"`` key.

        When multiple pricing constraints pass, their
        :class:`PriceModifier` values stack (discounts compound).
        """
        applicable = list(self.constraints.get(tool_name, []))
        applicable.extend(self.constraints.get("*", []))

        if not applicable:
            return ConstraintResult(allowed=True)

        if self.logic == "ALL_MUST_PASS":
            return self._eval_all_must_pass(applicable, context)
        elif self.logic == "ANY_MUST_PASS":
            return self._eval_any_must_pass(applicable, context)
        else:  # FIRST_MATCH
            return self._eval_first_match(applicable, context)

    # ---- logic modes ----

    def _eval_all_must_pass(
        self,
        constraints: list[ToolConstraint],
        context: ConstraintContext,
    ) -> ConstraintResult:
        """All constraints must allow.  Price modifiers stack."""
        combined_modifier: PriceModifier | None = None
        combined_metadata: dict[str, Any] = {}

        for constraint in constraints:
            result = constraint.evaluate(context)
            if not result.allowed:
                return result  # short-circuit on first denial
            combined_metadata.update(result.metadata)
            combined_modifier = _stack_modifiers(combined_modifier, result.price_modifier)

        return ConstraintResult(
            allowed=True,
            price_modifier=combined_modifier,
            metadata=combined_metadata,
        )

    def _eval_any_must_pass(
        self,
        constraints: list[ToolConstraint],
        context: ConstraintContext,
    ) -> ConstraintResult:
        """At least one constraint must allow."""
        last_denial: ConstraintResult | None = None

        for constraint in constraints:
            result = constraint.evaluate(context)
            if result.allowed:
                return result
            last_denial = result

        # None passed
        return last_denial or ConstraintResult(
            allowed=False,
            reason="no_constraint_passed",
            message="No constraint allowed this invocation.",
        )

    def _eval_first_match(
        self,
        constraints: list[ToolConstraint],
        context: ConstraintContext,
    ) -> ConstraintResult:
        """Use the first constraint that produces a non-neutral result."""
        for constraint in constraints:
            result = constraint.evaluate(context)
            if not result.allowed or result.price_modifier is not None:
                return result

        # All neutral — allow
        return ConstraintResult(allowed=True)


def _stack_modifiers(
    base: PriceModifier | None,
    addition: PriceModifier | None,
) -> PriceModifier | None:
    """Stack two price modifiers (discounts compound)."""
    if addition is None:
        return base
    if base is None:
        return addition

    # If either is free, result is free
    free = base.free or addition.free

    # Compound percentage discounts:
    # After base discount, remaining fraction = (1 - base/100)
    # After addition discount, remaining fraction = (1 - base/100) * (1 - addition/100)
    # Total discount = 1 - remaining = 1 - (1-a)(1-b)
    if base.discount_percent > 0 or addition.discount_percent > 0:
        remaining = (1 - base.discount_percent / 100) * (
            1 - addition.discount_percent / 100
        )
        combined_pct = (1 - remaining) * 100
    else:
        combined_pct = 0.0

    # Absolute discounts add
    combined_sats = base.discount_sats + addition.discount_sats

    # Multipliers multiply
    combined_mult = base.bonus_multiplier * addition.bonus_multiplier
    combined_surge = base.surge_multiplier * addition.surge_multiplier

    return PriceModifier(
        discount_percent=combined_pct,
        discount_sats=combined_sats,
        free=free,
        bonus_multiplier=combined_mult,
        surge_multiplier=combined_surge,
    )
