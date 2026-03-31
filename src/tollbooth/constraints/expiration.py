"""Tranche expiration constraint — set credit demurrage as a pipeline step."""

from __future__ import annotations

from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ConstraintSchema,
    ParamSchema,
    ToolConstraint,
)


class TrancheExpirationConstraint(ToolConstraint):
    """Set per-tranche credit expiration (demurrage).

    Credits not used within the TTL are forfeited. The TTL is computed
    once at credit-deposit time and stamped on each tranche.

    This constraint always allows tool calls — it is informational,
    not a gate. The runtime reads ``ttl_seconds`` from the evaluation
    metadata when depositing credits.

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
            reason="tranche_expiration",
            message=f"Credits expire after {self.ttl_days} days.",
            metadata={
                "ttl_seconds": self.ttl_days * 86400,
                "ttl_days": self.ttl_days,
            },
        )

    def describe(self) -> str:
        return f"Credit tranches expire after {self.ttl_days} days"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tranche_expiration",
            "ttl_days": self.ttl_days,
            "target_usage_pct": self.target_usage_pct,
            "min_days": self.min_days,
            "max_days": self.max_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrancheExpirationConstraint:
        return cls(
            ttl_days=int(data.get("ttl_days", 15)),
            target_usage_pct=float(data.get("target_usage_pct", 0.75)),
            min_days=int(data.get("min_days", 3)),
            max_days=int(data.get("max_days", 90)),
        )

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="tranche_expiration",
            category="Credit Terms",
            description=(
                "Set credit tranche expiration (demurrage). Credits not used "
                "within the TTL are forfeited. The pricing interview recommends "
                "a TTL based on expected daily usage so patrons use ~75% of "
                "each tranche before expiration."
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
