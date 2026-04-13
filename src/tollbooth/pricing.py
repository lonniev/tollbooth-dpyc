"""Dynamic tool pricing for Tollbooth micropayment gating."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class ToolPricing:
    """Computes the api_sat cost for a tool call.

    Supports fixed costs, percentage-based costs keyed off a tool parameter,
    or both combined.
    """

    fixed: int = 0
    rate_percent: float = 0.0
    rate_param: str = ""
    min_cost: int = 0
    max_cost: int | None = None

    def compute(self, **params: object) -> int:
        """Return the api_sat cost given tool call parameters."""
        cost = self.fixed

        if self.rate_percent > 0 and self.rate_param:
            if self.rate_param not in params:
                raise ValueError(
                    f"Missing required parameter '{self.rate_param}' for rate pricing"
                )
            raw = params[self.rate_param]
            if not isinstance(raw, (int, float)):
                raise ValueError(
                    f"Parameter '{self.rate_param}' must be numeric, got {type(raw).__name__}"
                )
            amount = float(raw)
            if amount < 0:
                raise ValueError(
                    f"Parameter '{self.rate_param}' must be non-negative, got {amount}"
                )
            rate_cost = ceil(amount * self.rate_percent / 100) if amount > 0 else 0
            rate_cost = max(rate_cost, self.min_cost)
            cost += rate_cost
        else:
            cost = max(cost, self.min_cost)

        if self.max_cost is not None:
            cost = min(cost, self.max_cost)

        return cost
