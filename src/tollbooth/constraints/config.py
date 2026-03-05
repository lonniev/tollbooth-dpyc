"""JSON config loader for the Constraint Engine.

Loads an operator-provided JSON dict into a :class:`ConstraintEngine`,
mapping constraint type strings to their concrete classes via
``CONSTRAINT_REGISTRY``.
"""

from __future__ import annotations

from typing import Any

from tollbooth.constraints.base import ToolConstraint
from tollbooth.constraints.engine import ConstraintEngine
from tollbooth.constraints.expression import JsonExpressionConstraint
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

    return cls.from_dict(data)


def load_constraints(config: dict[str, Any]) -> ConstraintEngine:
    """Build a :class:`ConstraintEngine` from an operator config dict.

    Expected schema::

        {
            "tool_constraints": {
                "<tool_name>": {
                    "constraints": [ {type: ..., ...}, ... ],
                    "logic": "ALL_MUST_PASS"   // optional, default ALL_MUST_PASS
                },
                ...
            }
        }

    Raises :class:`ConfigError` on invalid input.
    """
    tool_constraints_section = config.get("tool_constraints")
    if not isinstance(tool_constraints_section, dict):
        raise ConfigError(
            "'tool_constraints' key must be a dict mapping tool names to config."
        )

    # Determine a global default logic, if given
    global_logic = config.get("logic", "ALL_MUST_PASS")

    constraints_map: dict[str, list[ToolConstraint]] = {}

    for tool_name, tool_cfg in tool_constraints_section.items():
        if not isinstance(tool_cfg, dict):
            raise ConfigError(
                f"Config for tool {tool_name!r} must be a dict, got {type(tool_cfg).__name__}."
            )

        raw_list = tool_cfg.get("constraints")
        if not isinstance(raw_list, list):
            raise ConfigError(
                f"Tool {tool_name!r} config must contain a 'constraints' list."
            )

        parsed: list[ToolConstraint] = []
        for i, item in enumerate(raw_list):
            try:
                parsed.append(load_constraint(item))
            except (ConfigError, KeyError, ValueError, TypeError) as exc:
                raise ConfigError(
                    f"Error in {tool_name!r} constraint #{i}: {exc}"
                ) from exc

        constraints_map[tool_name] = parsed

    # Per-tool logic overrides — we use the first one found (or global default).
    # In practice the engine uses a single logic for all, so we pick the
    # most common or the explicit global.
    logic = global_logic
    for tool_cfg in tool_constraints_section.values():
        if isinstance(tool_cfg, dict) and "logic" in tool_cfg:
            logic = tool_cfg["logic"]
            break  # use the first explicitly set logic

    return ConstraintEngine(constraints=constraints_map, logic=logic)


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a config dict without building the engine.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    if "tool_constraints" not in config:
        errors.append("Missing 'tool_constraints' key.")
        return errors

    tc = config["tool_constraints"]
    if not isinstance(tc, dict):
        errors.append("'tool_constraints' must be a dict.")
        return errors

    for tool_name, tool_cfg in tc.items():
        if not isinstance(tool_cfg, dict):
            errors.append(f"Config for tool {tool_name!r} must be a dict.")
            continue

        if "constraints" not in tool_cfg:
            errors.append(f"Tool {tool_name!r}: missing 'constraints' list.")
            continue

        raw_list = tool_cfg["constraints"]
        if not isinstance(raw_list, list):
            errors.append(f"Tool {tool_name!r}: 'constraints' must be a list.")
            continue

        for i, item in enumerate(raw_list):
            if not isinstance(item, dict):
                errors.append(f"Tool {tool_name!r} constraint #{i}: must be a dict.")
                continue

            ctype = item.get("type")
            if not ctype:
                errors.append(f"Tool {tool_name!r} constraint #{i}: missing 'type'.")
            elif ctype not in CONSTRAINT_REGISTRY:
                errors.append(
                    f"Tool {tool_name!r} constraint #{i}: unknown type {ctype!r}."
                )

        if "logic" in tool_cfg:
            if tool_cfg["logic"] not in ConstraintEngine.VALID_LOGIC:
                errors.append(
                    f"Tool {tool_name!r}: invalid logic {tool_cfg['logic']!r}."
                )

    return errors
