"""Tollbooth Constraint Engine — conditional access and pricing for MCP tools.

Public API
----------
All classes are re-exported here for convenience::

    from tollbooth.constraints import ConstraintEngine, ConstraintContext, ...
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
    load_constraints,
    validate_config,
)
from tollbooth.constraints.engine import ConstraintEngine
from tollbooth.constraints.demurrage import DemurrageConstraint
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
    "DemurrageConstraint",
    # Engine
    "ConstraintEngine",
    # Gate
    "ConstraintGate",
    # Config
    "CONSTRAINT_REGISTRY",
    "ConfigError",
    "get_all_schemas",
    "load_constraint",
    "load_constraints",
    "validate_config",
    # Utilities
    "parse_iso_duration",
]
