"""Dynamic tool pricing for Tollbooth micropayment gating."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class ToolPricing:
    """Computes the api_sat cost for a tool call.

    Supports fixed costs, percentage-based costs keyed off a numeric tool
    parameter, and **categorical multipliers** keyed off enum-style
    parameters (e.g. ``difficulty`` × ``mode``). All forms compose:

    1. ``fixed`` is the base cost.
    2. ``rate_percent`` × ``rate_param`` adds a percent-of-numeric component.
    3. ``multipliers`` multiplies the result by a per-parameter lookup.

    Multipliers are stored as an immutable nested tuple to keep ``ToolPricing``
    a hashable, frozen dataclass:
        ``(("difficulty", (("apprentice", 1.0), ("journeyman", 2.0), …)), …)``
    Missing param values resolve to multiplier 1.0 (no surcharge). Use this
    for Optionality-style "difficulty × historicity" pricing where neither
    axis is a dollar amount.
    """

    fixed: int = 0
    rate_percent: float = 0.0
    rate_param: str = ""
    min_cost: int = 0
    max_cost: int | None = None
    multipliers: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()

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

        # Apply categorical multipliers (Optionality: difficulty × mode).
        # Each unmatched value contributes 1.0 (no surcharge); a missing
        # param key means the table doesn't apply to this call.
        if self.multipliers:
            product = 1.0
            for param_name, lookup in self.multipliers:
                value = params.get(param_name)
                if not isinstance(value, str):
                    continue
                for key, mult in lookup:
                    if key == value:
                        product *= float(mult)
                        break
            cost = ceil(cost * product)

        if self.max_cost is not None:
            cost = min(cost, self.max_cost)

        return cost
