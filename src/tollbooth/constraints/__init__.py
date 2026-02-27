"""Tollbooth Constraint Engine — conditional access and pricing for MCP tools.

Public API
----------
All classes are re-exported here for convenience::

    from tollbooth.constraints import ConstraintEngine, ConstraintContext, ...
"""

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
    PriceModifier,
    ToolConstraint,
)
from tollbooth.constraints.config import (
    CONSTRAINT_REGISTRY,
    ConfigError,
    load_constraint,
    load_constraints,
    validate_config,
)
from tollbooth.constraints.engine import ConstraintEngine
from tollbooth.constraints.expression import JsonExpressionConstraint
from tollbooth.constraints.periodic import PeriodicRefreshConstraint, parse_iso_duration
from tollbooth.constraints.pricing import (
    BulkBonusConstraint,
    CouponConstraint,
    FreeTrialConstraint,
    HappyHourConstraint,
    LoyaltyDiscountConstraint,
)
from tollbooth.constraints.supply import FiniteSupplyConstraint
from tollbooth.constraints.temporal import TemporalWindowConstraint

__all__ = [
    # Base
    "ToolConstraint",
    "ConstraintContext",
    "ConstraintResult",
    "PriceModifier",
    "LedgerSnapshot",
    "PatronIdentity",
    "EnvironmentSnapshot",
    # Constraints
    "TemporalWindowConstraint",
    "FiniteSupplyConstraint",
    "PeriodicRefreshConstraint",
    "CouponConstraint",
    "FreeTrialConstraint",
    "LoyaltyDiscountConstraint",
    "BulkBonusConstraint",
    "HappyHourConstraint",
    "JsonExpressionConstraint",
    # Engine
    "ConstraintEngine",
    # Config
    "CONSTRAINT_REGISTRY",
    "ConfigError",
    "load_constraint",
    "load_constraints",
    "validate_config",
    # Utilities
    "parse_iso_duration",
]
