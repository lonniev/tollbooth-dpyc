"""Tollbooth constraint chains — per-tool conditional access and pricing.

Each tool in an operator's pricing model owns an ordered chain of
constraint steps.  The :class:`ConstraintGate` walks that chain at
debit / preview time, applying each step's price modifier in turn.
"""

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ConstraintSchema,
    EnvironmentSnapshot,
    LedgerSnapshot,
    ParamSchema,
    PatronIdentity,
    PriceModifier,
    ToolConstraint,
)
from tollbooth.constraints.config import (
    CONSTRAINT_REGISTRY,
    ConfigError,
    get_all_schemas,
    load_constraint,
    validate_step,
)
from tollbooth.constraints.expression import JsonExpressionConstraint
from tollbooth.constraints.gate import ConstraintGate
from tollbooth.constraints.periodic import PeriodicRefreshConstraint, parse_iso_duration
from tollbooth.constraints.pricing import (
    BulkBonusConstraint,
    CouponConstraint,
    FreeTrialConstraint,
    HappyHourConstraint,
    LoyaltyDiscountConstraint,
)
from tollbooth.constraints.patron_proof import PatronProofConstraint
from tollbooth.constraints.supply import FiniteSupplyConstraint
from tollbooth.constraints.surge import SurgePricingConstraint
from tollbooth.constraints.temporal import TemporalWindowConstraint

__all__ = [
    # Base
    "ToolConstraint",
    "ConstraintContext",
    "ConstraintResult",
    "ConstraintSchema",
    "ParamSchema",
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
    "SurgePricingConstraint",
    "PatronProofConstraint",
    # Gate
    "ConstraintGate",
    # Config
    "CONSTRAINT_REGISTRY",
    "ConfigError",
    "get_all_schemas",
    "load_constraint",
    "validate_step",
    # Utilities
    "parse_iso_duration",
]
