"""Tests for tollbooth.constraints.config — type registry + per-step loader/validator."""

import pytest

from tollbooth.constraints.config import (
    CONSTRAINT_REGISTRY,
    ConfigError,
    load_constraint,
    validate_step,
)
from tollbooth.constraints.temporal import TemporalWindowConstraint
from tollbooth.constraints.supply import FiniteSupplyConstraint
from tollbooth.constraints.pricing import FreeTrialConstraint


# ---------------------------------------------------------------------------
# CONSTRAINT_REGISTRY
# ---------------------------------------------------------------------------


class TestConstraintRegistry:
    def test_all_types_registered(self):
        # Every constraint class we ship should be reachable via the registry.
        expected = {
            "temporal_window",
            "finite_supply",
            "periodic_refresh",
            "coupon",
            "free_trial",
            "loyalty_discount",
            "bulk_bonus",
            "happy_hour",
            "json_expression",
            "surge_pricing",
            "patron_proof",
        }
        assert expected.issubset(set(CONSTRAINT_REGISTRY))

    def test_registry_values_are_classes(self):
        for cls in CONSTRAINT_REGISTRY.values():
            assert isinstance(cls, type)


# ---------------------------------------------------------------------------
# load_constraint — single-step loader
# ---------------------------------------------------------------------------


class TestLoadConstraint:
    def test_temporal_window(self):
        c = load_constraint({
            "type": "temporal_window",
            "schedule_start": "08:00",
            "schedule_end": "21:00",
        })
        assert isinstance(c, TemporalWindowConstraint)

    def test_finite_supply(self):
        c = load_constraint({
            "type": "finite_supply",
            "max_invocations": 100,
        })
        assert isinstance(c, FiniteSupplyConstraint)

    def test_free_trial(self):
        c = load_constraint({"type": "free_trial", "first_n_free": 3})
        assert isinstance(c, FreeTrialConstraint)

    def test_missing_type(self):
        with pytest.raises(ConfigError, match="missing 'type'"):
            load_constraint({"schedule_start": "08:00", "schedule_end": "21:00"})

    def test_unknown_type(self):
        with pytest.raises(ConfigError, match="Unknown constraint type"):
            load_constraint({"type": "nonexistent"})


# ---------------------------------------------------------------------------
# validate_step — per-step validator (no instantiation)
# ---------------------------------------------------------------------------


class TestValidateStep:
    def test_valid_step(self):
        assert validate_step({"type": "free_trial", "first_n_free": 5}) == []

    def test_not_a_dict(self):
        errors = validate_step("not a dict")  # type: ignore[arg-type]
        assert errors and "dict" in errors[0]

    def test_missing_type(self):
        errors = validate_step({"first_n_free": 5})
        assert errors and "type" in errors[0]

    def test_unknown_type(self):
        errors = validate_step({"type": "made_up_constraint"})
        assert errors and "Unknown constraint type" in errors[0]
