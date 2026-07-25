"""Constraint type registry and per-step loader.

Maps the JSON ``type`` strings operators put in their pricing model's
``PipelineStep`` objects to the concrete :class:`ToolConstraint`
classes that implement them.  Each step in a tool's chain is
instantiated by :func:`load_constraint` at evaluation time.
"""

from __future__ import annotations

from typing import Any

from tollbooth.constraints.base import ConstraintSchema, ToolConstraint
from tollbooth.constraints.expression import JsonExpressionConstraint
from tollbooth.constraints.patron_proof import PatronProofConstraint
from tollbooth.constraints.periodic import PeriodicRefreshConstraint
from tollbooth.constraints.pricing import (
    BulkBonusConstraint,
    CouponConstraint,
    FreeTrialConstraint,
    HappyHourConstraint,
    LoyaltyDiscountConstraint,
)
from tollbooth.constraints.supply import FiniteSupplyConstraint
from tollbooth.constraints.surge import SurgePricingConstraint
from tollbooth.constraints.temporal import TemporalWindowConstraint

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONSTRAINT_REGISTRY: dict[str, type[ToolConstraint]] = {
    "temporal_window": TemporalWindowConstraint,
    "finite_supply": FiniteSupplyConstraint,
    "periodic_refresh": PeriodicRefreshConstraint,
    "coupon": CouponConstraint,
    "free_trial": FreeTrialConstraint,
    "loyalty_discount": LoyaltyDiscountConstraint,
    "bulk_bonus": BulkBonusConstraint,
    "happy_hour": HappyHourConstraint,
    "json_expression": JsonExpressionConstraint,
    "surge_pricing": SurgePricingConstraint,
    "patron_proof": PatronProofConstraint,
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when the operator config is invalid."""


def load_constraint(data: dict[str, Any]) -> ToolConstraint:
    """Instantiate a single constraint from its JSON dict.

    The dict **must** contain a ``"type"`` key matching a registry entry.
    """
    ctype = data.get("type")
    if not ctype:
        raise ConfigError(f"Constraint dict missing 'type' key: {data!r}")

    cls = CONSTRAINT_REGISTRY.get(ctype)
    if cls is None:
        raise ConfigError(
            f"Unknown constraint type {ctype!r}. "
            f"Known types: {sorted(CONSTRAINT_REGISTRY)}"
        )

    # Patron-group scoping lives on PricingStep.patron_npubs (see
    # constraints/gate.py), which enforces its own max-10 rule. The former
    # constraint-level `_patron_npubs` handling here was dead: it read a key
    # (`_patron_npubs`) nothing ever serialized, and nothing read the attribute
    # back. Removed in M2.5.
    return cls.from_dict(data)


def validate_step(data: dict[str, Any]) -> list[str]:
    """Validate one constraint-step dict without instantiating it.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("Step must be a dict.")
        return errors
    ctype = data.get("type")
    if not ctype:
        errors.append("Step missing 'type'.")
        return errors
    if ctype not in CONSTRAINT_REGISTRY:
        errors.append(f"Unknown constraint type {ctype!r}.")
    return errors


def get_all_schemas() -> list[ConstraintSchema]:
    """Return the schema for every registered constraint type."""
    return [cls.schema() for cls in CONSTRAINT_REGISTRY.values()]
