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


def _validate_pipeline(pipeline: list) -> list[str]:
    """Validate pipeline steps against the constraint registry.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []
    schemas_by_type = {s.type: s for s in get_all_schemas()}

    for i, step in enumerate(pipeline):
        ctype = step.type
        if ctype not in CONSTRAINT_REGISTRY:
            errors.append(
                f"Step #{i} ({step.id!r}): unknown constraint type {ctype!r}. "
                f"Valid types: {sorted(CONSTRAINT_REGISTRY)}"
            )
            continue

        schema = schemas_by_type.get(ctype)
        if schema is None:
            continue  # no schema to validate against

        # Check required params are present
        for param in schema.params:
            if param.required and param.name not in step.params:
                errors.append(
                    f"Step #{i} ({step.id!r}): missing required param {param.name!r} "
                    f"for constraint type {ctype!r}."
                )

    return errors


def _model_to_response(model: PricingModel) -> dict[str, Any]:
    """Convert a PricingModel to a flat response dict matching PricingStudio's PricingModelResponse."""
    return {
        "status": "ok",
        "model_id": model.model_id,
        "name": model.name,
        "is_active": model.is_active,
        "tools": [tp.to_dict() for tp in model.tools],
        "pipeline": [ps.to_dict() for ps in model.pipeline],
    }


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
        return {"status": "ok", "model_id": None, "name": None, "is_active": None, "tools": None, "pipeline": None}

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

    # Validate pipeline steps against constraint registry
    validation_errors = _validate_pipeline(model.pipeline)
    if validation_errors:
        return {"status": "error", "error": "Pipeline validation failed", "details": validation_errors}

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
