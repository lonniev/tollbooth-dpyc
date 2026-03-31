"""Demurrage constraint — credit decay encourages velocity of circulation."""

from __future__ import annotations

from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ConstraintSchema,
    ParamSchema,
    ToolConstraint,
)


class DemurrageConstraint(ToolConstraint):
    """Apply demurrage to credit tranches — unused credits expire.

    In the Austrian economic tradition, demurrage encourages the velocity
    of circulation by giving credits a finite shelf life. This is not a
    penalty — it is a natural property of the credit that aligns patron
    incentives with operator sustainability.

    The TTL is computed once at credit-deposit time and stamped on each
    tranche. This constraint always allows tool calls — it is
    informational, not a gate. The runtime reads ``ttl_seconds`` from
    the evaluation metadata when depositing credits.

    Parameters
    ----------
    ttl_days:
        Days until a credit tranche expires. Clamped to [min_days, max_days].
    target_usage_pct:
        Target percentage of tranche used before expiration. Used by the
        pricing interview to recommend a ttl_days value.
    min_days:
        Floor for ttl_days (default 3).
    max_days:
        Ceiling for ttl_days (default 90).
    """

    def __init__(
        self,
        ttl_days: int = 15,
        target_usage_pct: float = 0.75,
        min_days: int = 3,
        max_days: int = 90,
    ) -> None:
        self.ttl_days = max(min_days, min(max_days, ttl_days))
        self.target_usage_pct = target_usage_pct
        self.min_days = min_days
        self.max_days = max_days

    # ---- ToolConstraint interface ----

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        return ConstraintResult(
            allowed=True,
            reason="demurrage",
            message=f"Credits expire after {self.ttl_days} days.",
            metadata={
                "ttl_seconds": self.ttl_days * 86400,
                "ttl_days": self.ttl_days,
            },
        )

    def describe(self) -> str:
        return f"Demurrage: credit tranches expire after {self.ttl_days} days"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "demurrage",
            "ttl_days": self.ttl_days,
            "target_usage_pct": self.target_usage_pct,
            "min_days": self.min_days,
            "max_days": self.max_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DemurrageConstraint:
        return cls(
            ttl_days=int(data.get("ttl_days", 15)),
            target_usage_pct=float(data.get("target_usage_pct", 0.75)),
            min_days=int(data.get("min_days", 3)),
            max_days=int(data.get("max_days", 90)),
        )

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="demurrage",
            category="Credit Terms",
            description=(
                "Demurrage: credit tranches expire after a configurable number "
                "of days. This encourages healthy velocity of circulation — "
                "patrons are motivated to use their credits rather than hoard "
                "them, and operators maintain predictable revenue flow. The "
                "pricing interview recommends a TTL based on expected daily "
                "usage so patrons use ~75% of each tranche before expiration."
            ),
            params=[
                ParamSchema(
                    name="ttl_days", type="int",
                    description="Days until a credit tranche expires.",
                ),
                ParamSchema(
                    name="target_usage_pct", type="float", required=False, default=0.75,
                    description="Target usage percentage before expiration (for interview recommendations).",
                ),
                ParamSchema(
                    name="min_days", type="int", required=False, default=3,
                    description="Minimum allowed TTL in days.",
                ),
                ParamSchema(
                    name="max_days", type="int", required=False, default=90,
                    description="Maximum allowed TTL in days.",
                ),
            ],
        )
