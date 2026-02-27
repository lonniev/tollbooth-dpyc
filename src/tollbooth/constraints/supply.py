"""Finite supply constraint — cap total invocations globally or per-patron."""

from __future__ import annotations

from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ToolConstraint,
)


class FiniteSupplyConstraint(ToolConstraint):
    """Enforce a finite invocation cap.

    Parameters
    ----------
    max_invocations:
        Total allowed invocations.
    current_count:
        Externally tracked current usage (or ``context.env.invocation_count``
        when *scope* is ``"per_patron"``).
    scope:
        ``"global"`` — *current_count* is the global total.
        ``"per_patron"`` — uses ``context.env.invocation_count`` instead.
    """

    def __init__(
        self,
        max_invocations: int,
        current_count: int = 0,
        scope: str = "global",
    ) -> None:
        self.max_invocations = max_invocations
        self.current_count = current_count
        self.scope = scope

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        count = (
            context.env.invocation_count
            if self.scope == "per_patron"
            else self.current_count
        )
        remaining = max(0, self.max_invocations - count)

        if remaining <= 0:
            return ConstraintResult(
                allowed=False,
                reason="supply_exhausted",
                message=f"Supply exhausted ({self.max_invocations} invocations used).",
                metadata={"remaining_invocations": 0},
            )

        return ConstraintResult(
            allowed=True,
            metadata={"remaining_invocations": remaining},
        )

    def describe(self) -> str:
        return (
            f"Limited to {self.max_invocations} invocations ({self.scope})."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "finite_supply",
            "max_invocations": self.max_invocations,
            "current_count": self.current_count,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FiniteSupplyConstraint:
        return cls(
            max_invocations=int(data["max_invocations"]),
            current_count=int(data.get("current_count", 0)),
            scope=str(data.get("scope", "global")),
        )
