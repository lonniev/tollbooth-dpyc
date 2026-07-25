"""Surge pricing constraint — demand-elastic pricing from global counters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ConstraintSchema,
    ParamSchema,
    PriceModifier,
    ToolConstraint,
)


@dataclass(frozen=True)
class _Tier:
    """A single surge tier: when utilization >= capacity_pct, apply multiplier."""

    capacity_pct: float
    multiplier: float


class SurgePricingConstraint(ToolConstraint):
    """Demand-elastic pricing derived from global demand counters.

    Never denies — only reprices.  Reads ``env.global_demand`` to determine
    current utilization as a fraction of ``max_capacity``, then walks the
    tier list to find the applicable surge multiplier.

    Config example::

        {
            "type": "surge_pricing",
            "max_capacity": 100,
            "window": "1h",
            "tiers": [
                {"capacity_pct": 0.5, "multiplier": 1.0},
                {"capacity_pct": 0.8, "multiplier": 1.5},
                {"capacity_pct": 0.95, "multiplier": 2.0}
            ]
        }

    ``window`` is advisory metadata — the operator's server code is
    responsible for passing the correct demand count for the matching
    time window via ``EnvironmentSnapshot.global_demand``.
    """

    def __init__(
        self,
        max_capacity: int,
        tiers: list[_Tier],
        window: str = "1h",
    ) -> None:
        if max_capacity <= 0:
            raise ValueError("max_capacity must be positive")
        if not tiers:
            raise ValueError("At least one tier is required")

        self.max_capacity = max_capacity
        self.tiers = sorted(tiers, key=lambda t: t.capacity_pct)
        self.window = window

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        demand = context.env.demand_for(context.env.tool_name)
        utilization = demand / self.max_capacity

        # Walk tiers — last tier whose threshold is met wins
        multiplier = 1.0
        for tier in self.tiers:
            if utilization >= tier.capacity_pct:
                multiplier = tier.multiplier

        if multiplier == 1.0:
            return ConstraintResult(allowed=True)

        return ConstraintResult(
            allowed=True,
            price_modifier=PriceModifier(surge_multiplier=multiplier),
            metadata={
                "surge_utilization": round(utilization, 3),
                "surge_multiplier": multiplier,
                "demand": demand,
                "max_capacity": self.max_capacity,
            },
        )

    def describe(self) -> str:
        tiers_desc = ", ".join(
            f"{t.capacity_pct:.0%}={t.multiplier}x" for t in self.tiers
        )
        return (
            f"Surge pricing: {self.max_capacity} calls/{self.window} capacity, "
            f"tiers [{tiers_desc}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "surge_pricing",
            "max_capacity": self.max_capacity,
            "window": self.window,
            "tiers": [
                {"capacity_pct": t.capacity_pct, "multiplier": t.multiplier}
                for t in self.tiers
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurgePricingConstraint:
        raw_tiers = data.get("tiers", [])
        if not isinstance(raw_tiers, list):
            raise TypeError("'tiers' must be a list")

        tiers = []
        for t in raw_tiers:
            pct = float(t["capacity_pct"])
            mult = float(t["multiplier"])
            if not (0 < pct <= 1.0):
                raise ValueError(f"capacity_pct must be in (0, 1], got {pct}")
            if mult <= 0:
                raise ValueError(f"multiplier must be positive, got {mult}")
            tiers.append(_Tier(capacity_pct=pct, multiplier=mult))

        return cls(
            max_capacity=int(data["max_capacity"]),
            tiers=tiers,
            window=str(data.get("window", "1h")),
        )

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="surge_pricing",
            category="Dynamic",
            description="Demand-elastic pricing derived from global demand counters. Never denies — only reprices via surge multiplier tiers.",
            params=[
                ParamSchema(name="max_capacity", type="int", description="Maximum invocations per window (capacity baseline)."),
                ParamSchema(name="tiers", type="tiers", description="List of {capacity_pct: float (0-1], multiplier: float} tier objects."),
                ParamSchema(name="window", type="string", required=False, default="1h", description="Advisory time window label (e.g. '1h', '24h')."),
            ],
        )
