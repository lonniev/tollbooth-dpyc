"""Tests for tollbooth.constraints.config — JSON config loader and validator."""

from datetime import datetime, timezone

import pytest

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.config import (
    CONSTRAINT_REGISTRY,
    ConfigError,
    load_constraint,
    load_constraints,
    validate_config,
)
from tollbooth.constraints.engine import ConstraintEngine
from tollbooth.constraints.temporal import TemporalWindowConstraint
from tollbooth.constraints.supply import FiniteSupplyConstraint
from tollbooth.constraints.pricing import FreeTrialConstraint


def _ctx(hour=12, invocation_count=0):
    return ConstraintContext(
        ledger=LedgerSnapshot(),
        patron=PatronIdentity(),
        env=EnvironmentSnapshot(
            utc_now=datetime(2026, 3, 2, hour, 0, tzinfo=timezone.utc),  # Monday
            invocation_count=invocation_count,
        ),
    )


# ---------------------------------------------------------------------------
# CONSTRAINT_REGISTRY
# ---------------------------------------------------------------------------


class TestConstraintRegistry:
    def test_all_types_registered(self):
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
            "demurrage",
            "patron_proof",
        }
        assert set(CONSTRAINT_REGISTRY.keys()) == expected

    def test_registry_values_are_classes(self):
        for name, cls in CONSTRAINT_REGISTRY.items():
            assert hasattr(cls, "evaluate"), f"{name} class missing evaluate()"
            assert hasattr(cls, "from_dict"), f"{name} class missing from_dict()"


# ---------------------------------------------------------------------------
# load_constraint
# ---------------------------------------------------------------------------


class TestLoadConstraint:
    def test_temporal_window(self):
        c = load_constraint({
            "type": "temporal_window",
            "schedule": "08:00-21:00",
            "timezone": "UTC",
        })
        assert isinstance(c, TemporalWindowConstraint)

    def test_finite_supply(self):
        c = load_constraint({
            "type": "finite_supply",
            "max_invocations": 100,
        })
        assert isinstance(c, FiniteSupplyConstraint)

    def test_free_trial(self):
        c = load_constraint({
            "type": "free_trial",
            "first_n_free": 5,
        })
        assert isinstance(c, FreeTrialConstraint)

    def test_missing_type(self):
        with pytest.raises(ConfigError, match="missing 'type'"):
            load_constraint({"schedule": "08:00-21:00"})

    def test_unknown_type(self):
        with pytest.raises(ConfigError, match="Unknown constraint type"):
            load_constraint({"type": "nonexistent"})


# ---------------------------------------------------------------------------
# load_constraints (full engine)
# ---------------------------------------------------------------------------


class TestLoadConstraints:
    def test_basic_config(self):
        config = {
            "tool_constraints": {
                "fetch_whitepaper": {
                    "constraints": [
                        {"type": "temporal_window", "schedule": "08:00-21:00"},
                        {"type": "free_trial", "first_n_free": 3},
                    ],
                    "logic": "ALL_MUST_PASS",
                }
            }
        }
        engine = load_constraints(config)
        assert isinstance(engine, ConstraintEngine)
        assert "fetch_whitepaper" in engine.constraints
        assert len(engine.constraints["fetch_whitepaper"]) == 2

    def test_multi_tool_config(self):
        config = {
            "tool_constraints": {
                "tool_a": {
                    "constraints": [
                        {"type": "finite_supply", "max_invocations": 50},
                    ],
                },
                "tool_b": {
                    "constraints": [
                        {"type": "free_trial", "first_n_free": 10},
                    ],
                },
            }
        }
        engine = load_constraints(config)
        assert len(engine.constraints) == 2

    def test_missing_tool_constraints_key(self):
        with pytest.raises(ConfigError, match="tool_constraints"):
            load_constraints({})

    def test_tool_constraints_not_dict(self):
        with pytest.raises(ConfigError):
            load_constraints({"tool_constraints": "bad"})

    def test_tool_config_not_dict(self):
        with pytest.raises(ConfigError, match="must be a dict"):
            load_constraints({"tool_constraints": {"tool": "bad"}})

    def test_missing_constraints_list(self):
        with pytest.raises(ConfigError, match="constraints"):
            load_constraints({"tool_constraints": {"tool": {}}})

    def test_constraints_not_list(self):
        with pytest.raises(ConfigError, match="constraints"):
            load_constraints({
                "tool_constraints": {"tool": {"constraints": "bad"}}
            })

    def test_invalid_constraint_in_list(self):
        with pytest.raises(ConfigError, match="constraint #0"):
            load_constraints({
                "tool_constraints": {
                    "tool": {
                        "constraints": [
                            {"type": "unknown_type"}
                        ]
                    }
                }
            })

    def test_end_to_end_evaluation(self):
        config = {
            "tool_constraints": {
                "my_tool": {
                    "constraints": [
                        {"type": "temporal_window", "schedule": "08:00-21:00"},
                        {"type": "free_trial", "first_n_free": 3},
                    ],
                    "logic": "ALL_MUST_PASS",
                }
            }
        }
        engine = load_constraints(config)
        result = engine.evaluate("my_tool", _ctx(hour=12, invocation_count=1))
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.free is True

    def test_end_to_end_denied(self):
        config = {
            "tool_constraints": {
                "my_tool": {
                    "constraints": [
                        {"type": "temporal_window", "schedule": "08:00-10:00"},
                    ],
                    "logic": "ALL_MUST_PASS",
                }
            }
        }
        engine = load_constraints(config)
        result = engine.evaluate("my_tool", _ctx(hour=12))
        assert result.allowed is False


# ---------------------------------------------------------------------------
# Round-trip serialization
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_serialize_and_reload(self):
        config = {
            "tool_constraints": {
                "tool_x": {
                    "constraints": [
                        {
                            "type": "coupon",
                            "code": "EARLYBIRD",
                            "discount_percent": 50,
                            "expires_at": "2026-04-01T00:00:00Z",
                        },
                        {
                            "type": "finite_supply",
                            "max_invocations": 100,
                            "scope": "global",
                        },
                    ],
                    "logic": "ALL_MUST_PASS",
                }
            }
        }

        # Load
        engine = load_constraints(config)
        assert len(engine.constraints["tool_x"]) == 2

        # Serialize back
        rebuilt_list = [c.to_dict() for c in engine.constraints["tool_x"]]
        assert rebuilt_list[0]["type"] == "coupon"
        assert rebuilt_list[0]["code"] == "EARLYBIRD"
        assert rebuilt_list[1]["type"] == "finite_supply"

        # Re-load from serialized
        config2 = {
            "tool_constraints": {
                "tool_x": {
                    "constraints": rebuilt_list,
                    "logic": "ALL_MUST_PASS",
                }
            }
        }
        engine2 = load_constraints(config2)
        assert len(engine2.constraints["tool_x"]) == 2


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config(self):
        config = {
            "tool_constraints": {
                "tool": {
                    "constraints": [
                        {"type": "free_trial", "first_n_free": 5},
                    ]
                }
            }
        }
        errors = validate_config(config)
        assert errors == []

    def test_missing_tool_constraints(self):
        errors = validate_config({})
        assert len(errors) == 1
        assert "tool_constraints" in errors[0]

    def test_tool_constraints_not_dict(self):
        errors = validate_config({"tool_constraints": []})
        assert len(errors) == 1

    def test_tool_config_not_dict(self):
        errors = validate_config({"tool_constraints": {"t": "bad"}})
        assert any("must be a dict" in e for e in errors)

    def test_missing_constraints_key(self):
        errors = validate_config({"tool_constraints": {"t": {}}})
        assert any("constraints" in e for e in errors)

    def test_constraints_not_list(self):
        errors = validate_config({
            "tool_constraints": {"t": {"constraints": "bad"}}
        })
        assert any("list" in e for e in errors)

    def test_constraint_not_dict(self):
        errors = validate_config({
            "tool_constraints": {"t": {"constraints": ["bad"]}}
        })
        assert any("dict" in e for e in errors)

    def test_missing_type(self):
        errors = validate_config({
            "tool_constraints": {"t": {"constraints": [{}]}}
        })
        assert any("type" in e for e in errors)

    def test_unknown_type(self):
        errors = validate_config({
            "tool_constraints": {"t": {"constraints": [{"type": "xyz"}]}}
        })
        assert any("unknown" in e.lower() for e in errors)

    def test_invalid_logic(self):
        errors = validate_config({
            "tool_constraints": {
                "t": {
                    "constraints": [{"type": "free_trial"}],
                    "logic": "BAD_LOGIC",
                }
            }
        })
        assert any("logic" in e.lower() for e in errors)

    def test_multiple_errors(self):
        errors = validate_config({
            "tool_constraints": {
                "t1": {"constraints": [{"type": "bad1"}]},
                "t2": "not_a_dict",
            }
        })
        assert len(errors) >= 2
