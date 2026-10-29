"""The standard property tools' bodies. Who may is decided here; what is kept,
in ``tollbooth.properties``."""

from __future__ import annotations

import logging
from typing import Any

from tollbooth.properties import PropertyError, check_name, listing, validate

logger = logging.getLogger(__name__)


def _account(specs: dict[str, Any]) -> dict[str, Any]:
    return {n: s for n, s in specs.items() if s.follows == "account"}


async def get_operator_properties_tool(rt: Any) -> dict[str, Any]:
    """Every operator property, its spec and value. Anyone may read them."""
    try:
        stored = await (await rt.property_store()).values(rt.operator_npub())
    except Exception as exc:  # noqa: BLE001
        logger.warning("operator properties unreadable: %s", exc)
        return {"success": False, "error": "The operator's properties could not be read just now."}
    return {"success": True, "properties": listing(rt.operator_properties, stored)}


async def set_operator_property_tool(rt: Any, name: str, value: Any) -> dict[str, Any]:
    """Set one operator property, or return it to its default with an empty value."""
    if err := await rt.require_caller_proof(rt.operator_npub()):
        return err
    return await _set(rt, rt.operator_npub(), rt.operator_properties, name, value)


async def get_my_properties_tool(rt: Any, npub: str) -> dict[str, Any]:
    """A patron's properties. Device-following ones carry their spec and default
    only: their value lives in the patron's browser, never here."""
    if err := await rt.require_caller_proof(npub):
        return err
    try:
        stored = await (await rt.property_store()).values(npub)
    except Exception as exc:  # noqa: BLE001
        logger.warning("patron properties unreadable: %s", exc)
        return {"success": False, "error": "Your properties could not be read just now."}
    return {"success": True, "properties": listing(rt.patron_properties,
                                                   {n: v for n, v in stored.items()
                                                    if n in _account(rt.patron_properties)})}


async def set_my_property_tool(rt: Any, npub: str, name: str, value: Any) -> dict[str, Any]:
    """Set one of a patron's own account properties."""
    if err := await rt.require_caller_proof(npub):
        return err
    specs = rt.patron_properties
    if name in specs and specs[name].follows == "device":
        return {"success": False,
                "error": f"{specs[name].label} is kept on each device, so it is set there."}
    return await _set(rt, npub, _account(specs), name, value)


async def _set(rt: Any, npub: str, specs: dict[str, Any], name: str, value: Any) -> dict[str, Any]:
    try:
        spec = check_name(name, specs)
        store = await rt.property_store()
        if value is None or value == "":
            await store.clear(npub, name)
            return {"success": True, "name": name, "value": spec.default, "is_default": True}
        kept = validate(spec, value)
        await store.set(npub, name, kept)
    except PropertyError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("property %s not kept: %s", name, exc)
        return {"success": False, "error": f"{name} could not be kept just now."}
    return {"success": True, "name": name, "value": kept, "is_default": False}
