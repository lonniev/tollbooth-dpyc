"""Base classes for the Tollbooth Constraint Engine.

Defines the ToolConstraint ABC, ConstraintContext (with its three frozen
sub-snapshots), ConstraintResult, and PriceModifier.  Every baked-in
constraint inherits from ToolConstraint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Context snapshots — frozen, read-only views of patron/env state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerSnapshot:
    """Read-only snapshot of a patron's ledger state."""

    balance_api_sats: int = 0
    total_deposited_api_sats: int = 0
    total_consumed_api_sats: int = 0
    total_expired_api_sats: int = 0


@dataclass(frozen=True)
class PatronIdentity:
    """Patron identification context."""

    npub: str = ""
    membership_tier: str = "default"
    join_date: str | None = None  # ISO-8601 datetime string


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Runtime environment context."""

    utc_now: datetime  # Current UTC time
    tool_name: str = ""
    invocation_count: int = 0  # Per-patron total invocations for this tool
    global_demand: tuple[tuple[str, int], ...] = ()  # (tool_name, count) pairs

    def demand_for(self, tool_name: str) -> int:
        """Return the global demand count for *tool_name*, or 0 if unknown."""
        for name, count in self.global_demand:
            if name == tool_name:
                return count
        return 0


@dataclass(frozen=True)
class ConstraintContext:
    """All context available to constraint evaluation."""

    ledger: LedgerSnapshot
    patron: PatronIdentity
    env: EnvironmentSnapshot


# ---------------------------------------------------------------------------
# PriceModifier
# ---------------------------------------------------------------------------


@dataclass
class PriceModifier:
    """Price adjustment produced by a pricing constraint."""

    discount_percent: float = 0.0  # e.g. 50.0 for 50% off
    discount_sats: int = 0  # absolute discount in api-sats
    free: bool = False  # override to zero cost
    bonus_multiplier: float = 1.0  # e.g. 1.25 for 25% more value
    surge_multiplier: float = 1.0  # e.g. 1.5 for 50% surge markup

    def apply_to(self, base_price: int) -> int:
        """Apply this modifier to *base_price* and return the adjusted price.

        Order: free → surge markup → percent discount → absolute discount.
        Surge increases the base before discounts are applied, so a 2x surge
        with a 50% happy-hour discount nets out to 1x base price.
        """
        if self.free:
            return 0
        price = base_price
        if self.surge_multiplier != 1.0:
            price = int(price * self.surge_multiplier)
        if self.discount_percent > 0:
            price = max(0, price - int(price * self.discount_percent / 100))
        if self.discount_sats > 0:
            price = max(0, price - self.discount_sats)
        return price

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.discount_percent:
            d["discount_percent"] = self.discount_percent
        if self.discount_sats:
            d["discount_sats"] = self.discount_sats
        if self.free:
            d["free"] = True
        if self.bonus_multiplier != 1.0:
            d["bonus_multiplier"] = self.bonus_multiplier
        if self.surge_multiplier != 1.0:
            d["surge_multiplier"] = self.surge_multiplier
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriceModifier:
        return cls(
            discount_percent=float(data.get("discount_percent", 0.0)),
            discount_sats=int(data.get("discount_sats", 0)),
            free=bool(data.get("free", False)),
            bonus_multiplier=float(data.get("bonus_multiplier", 1.0)),
            surge_multiplier=float(data.get("surge_multiplier", 1.0)),
        )


# ---------------------------------------------------------------------------
# ConstraintResult
# ---------------------------------------------------------------------------


@dataclass
class ConstraintResult:
    """Result of evaluating a single constraint."""

    allowed: bool = True
    reason: str = ""  # machine-readable denial code
    message: str = ""  # human-readable explanation
    retry_after: datetime | None = None
    price_modifier: PriceModifier | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ToolConstraint ABC
# ---------------------------------------------------------------------------


class ToolConstraint(ABC):
    """Abstract base for all constraints."""

    @abstractmethod
    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        """Evaluate this constraint against *context*."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this constraint."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (including ``type`` key)."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolConstraint:
        """Deserialize from a JSON-compatible dict."""
        ...
