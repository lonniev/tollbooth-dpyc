"""Pricing model tools: get and set operator pricing models."""

from __future__ import annotations

import json
import logging
from typing import Any

from tollbooth.constraints.config import CONSTRAINT_REGISTRY, get_all_schemas
from tollbooth.pricing_model import PricingModel

logger = logging.getLogger(__name__)


def list_constraint_types() -> list[dict[str, Any]]:
    """Return schema dicts for all registered constraint types."""
    return [s.to_dict() for s in get_all_schemas()]


def build_pricing_preview(
    tool_id: str,
    tool_name: str,
    pricing: Any,
    parsed_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the base ``check_price`` result (pricing type + costs), pre-constraints.

    Pure: takes a resolved ``ToolPricing`` and parsed kwargs, returns the
    result skeleton the constraint pass then augments. Extracted from the
    ``check_price`` closure (audit M2.1b) so the percent / flat / flat+multipliers
    branching is unit-testable without a runtime.
    """
    result: dict[str, Any] = {
        "success": True,
        "tool_id": tool_id,
        "tool_name": tool_name,
        "constraints_enabled": False,
        "constraint_effects": [],
    }

    if pricing.rate_percent > 0:
        result["pricing_type"] = "percent"
        result["rate_percent"] = pricing.rate_percent
        result["rate_param"] = pricing.rate_param
        result["min_cost_sats"] = pricing.min_cost
        if parsed_kwargs:
            base_cost = pricing.compute(**parsed_kwargs)
            result["base_cost_api_sats"] = base_cost
            result["effective_cost_api_sats"] = base_cost
        else:
            result["base_cost_api_sats"] = None
            result["effective_cost_api_sats"] = None
            result["hint"] = (
                f"Pass tool_kwargs with '{pricing.rate_param}' "
                f"to preview the cost (e.g. '{{\"{pricing.rate_param}\": 1000}}')."
            )
    else:
        # Flat pricing may still scale by categorical multipliers — pass
        # parsed_kwargs through so lookup tables resolve against the preview.
        base_cost = pricing.compute(**parsed_kwargs)
        result["pricing_type"] = "flat" if not pricing.multipliers else "flat+multipliers"
        result["base_cost_api_sats"] = base_cost
        result["effective_cost_api_sats"] = base_cost
        if pricing.multipliers:
            result["multipliers"] = {
                param: {k: v for k, v in lookup}
                for param, lookup in pricing.multipliers
            }

    return result


def apply_constraint_preview(
    result: dict[str, Any],
    base_cost: int,
    effective: int,
    denial: dict[str, Any] | None,
    demand_count: int,
) -> None:
    """Augment a ``build_pricing_preview`` result with constraint-chain effects.

    Pure: mutates ``result`` in place from a ConstraintGate evaluation's
    ``(denial, effective)`` plus the tool's current demand. Extracted so the
    denied / discount / credit formatting is unit-testable.
    """
    if demand_count > 0:
        result["current_demand"] = demand_count
    if denial:
        result["effective_cost_api_sats"] = 0
        result["constraint_effects"].append({
            "type": "denied",
            "reason": denial.get("constraint_reason", "blocked"),
        })
    else:
        result["effective_cost_api_sats"] = effective
        if effective != base_cost:
            effect_type = "credit" if effective < 0 else "discount"
            result["constraint_effects"].append({
                "type": effect_type,
                "from": int(base_cost),
                "to": effective,
            })


def _validate_tool_chains(model: PricingModel) -> list[str]:
    """Validate every tool's constraint chain against the registry.

    Returns a list of error strings (empty = valid).  Each error names
    the tool and step position so the operator can find it in the
    Pricing Studio.
    """
    errors: list[str] = []
    schemas_by_type = {s.type: s for s in get_all_schemas()}

    for tool in model.tools:
        for i, step in enumerate(tool.chain):
            ctype = step.type
            if ctype not in CONSTRAINT_REGISTRY:
                errors.append(
                    f"Tool {tool.tool_name!r} step #{i} ({step.id!r}): "
                    f"unknown constraint type {ctype!r}. "
                    f"Valid types: {sorted(CONSTRAINT_REGISTRY)}"
                )
                continue

            schema = schemas_by_type.get(ctype)
            if schema is None:
                continue  # no schema to validate against

            for param in schema.params:
                if param.required and param.name not in step.params:
                    errors.append(
                        f"Tool {tool.tool_name!r} step #{i} ({step.id!r}): "
                        f"missing required param {param.name!r} "
                        f"for constraint type {ctype!r}."
                    )

    return errors


def _model_to_response(model: PricingModel) -> dict[str, Any]:
    """Convert a PricingModel to a flat response dict for the Studio."""
    d: dict[str, Any] = {
        "status": "ok",
        "model_id": model.model_id,
        "name": model.name,
        "is_active": model.is_active,
        "tools": [tp.to_dict() for tp in model.tools],
    }
    if model.tranche_lifetime is not None:
        d["tranche_lifetime"] = model.tranche_lifetime.to_dict()
    return d


async def get_pricing_model_tool(
    store: Any,
    operator: str,
) -> dict[str, Any]:
    """Fetch the active pricing model for an operator.

    Returns the model as a flat dict suitable for JSON serialization,
    or ``{"status": "ok"}`` with null fields if no active model exists.

    Free tool (operator self-service).
    """
    try:
        model = await store.fetch_active_model(operator)
    except Exception as e:
        logger.exception("Failed to fetch pricing model for %s", operator)
        return {"status": "error", "error": str(e)}

    if model is None:
        return {"status": "ok", "model_id": None, "name": None, "is_active": None, "tools": None}

    return _model_to_response(model)


async def set_pricing_model_tool(
    store: Any,
    operator: str,
    model_json: str,
) -> dict[str, Any]:
    """Create or update the active pricing model for an operator.

    If the parsed model has a ``model_id`` and that model exists in the store,
    updates the existing model and activates it.  Otherwise creates a new model
    and activates it.

    Returns ``{"status": "ok", "model_id": "...", "tools_count": N}``.

    Free tool (operator self-service).
    """
    try:
        data = json.loads(model_json)
        model_id = data.get("model_id", "") or ""
        model = PricingModel.from_json(model_json, model_id=model_id, operator=operator)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {"status": "error", "error": f"Invalid model_json: {e}"}

    # Validate every tool's chain against the constraint registry.
    validation_errors = _validate_tool_chains(model)
    if validation_errors:
        return {"status": "error", "error": "Tool chain validation failed", "details": validation_errors}

    try:
        await store.ensure_schema()

        # Check if this is an update to an existing model
        if model.model_id:
            existing = await store.fetch_active_model(operator)
            if existing and existing.model_id == model.model_id:
                await store.update_model(model.model_id, model_json)
                await store.activate_model(model.model_id, operator)
                return {
                    "status": "ok",
                    "model_id": model.model_id,
                    "tools_count": len(model.tools),
                    "action": "updated",
                }

        # Create new model and activate it
        model.operator = operator
        if not model.name:
            model.name = "Pricing Model"
        new_id = await store.create_model(model)
        await store.activate_model(new_id, operator)
        return {
            "status": "ok",
            "model_id": new_id,
            "tools_count": len(model.tools),
            "action": "created",
        }
    except Exception as e:
        logger.exception("Failed to set pricing model for %s", operator)
        return {"status": "error", "error": str(e)}
