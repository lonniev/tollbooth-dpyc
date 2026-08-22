"""Accept a JSON string where a tool declared an object or an array.

An MCP client sends tool arguments as JSON. A client that sends
``{"patch": {"doc": …}}`` is understood; a client that sends
``{"patch": "{\\"doc\\": …}"}`` — the same value, serialised one extra time —
is not. FastMCP hands the string to pydantic, which reports::

    Input should be a valid dictionary [type=dict_type, input_type=str]

That message sends the caller hunting for a malformed payload, when the
payload is fine and merely arrived one encoding too deep.

Clients do this unevenly, and the unevenness is what makes it expensive to
diagnose. The same client will send a small argument as an object and a large
one as a string, so the failure looks like a size limit; or send one tool an
object and its sibling a string, so it looks like the tools disagree. Both
readings send an investigator after a server bug that does not exist. Observed
on excalibur's ``update_post`` (10 KB patch rejected, the identical document
accepted by ``create_post``) and on roastify's ``update_design_text``.

**What this does NOT do.** It never touches a parameter the tool declared as a
string, however JSON-shaped its text. It never touches a union such as
``str | dict``, where a string is a legitimate value — those declare their
types under ``anyOf``, which is deliberately not matched. Coercion applies only
where the schema says object-or-array and the value is a string, which is a
combination that was already guaranteed to fail validation. So the change can
turn a certain failure into a success and cannot turn a success into anything
else.

A string that does not parse is left exactly as it arrived, so validation still
reports it — the caller genuinely did send something malformed, and that is
worth saying plainly rather than papering over.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_COERCIBLE = ("object", "array")


def _declared_types(schema: dict[str, Any]) -> list[str]:
    """The JSON-schema ``type`` for one property, normalised to a list.

    A union (``anyOf``) returns nothing: if a string is one of the accepted
    types then a string is not a mistake, and parsing it would silently change
    a caller's meaning.
    """
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list):
        return [t for t in declared if isinstance(t, str)]
    return []


def coerce_json_arguments(
    arguments: dict[str, Any], properties: dict[str, Any],
) -> list[str]:
    """Parse stringified object/array arguments in place.

    Args:
        arguments: The incoming tool arguments. Mutated in place.
        properties: The tool's JSON-schema ``properties`` block.

    Returns:
        The names of the arguments that were parsed, for logging.
    """
    coerced: list[str] = []
    for name, value in list(arguments.items()):
        if not isinstance(value, str):
            continue
        schema = properties.get(name)
        if not isinstance(schema, dict):
            continue
        if not any(t in _COERCIBLE for t in _declared_types(schema)):
            continue
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Leave it. The caller really did send something unparseable, and
            # validation's complaint about it is the honest one.
            continue
        if isinstance(parsed, (dict, list)):
            arguments[name] = parsed
            coerced.append(name)
    return coerced


def build_json_arg_middleware() -> Any:
    """A FastMCP middleware that applies :func:`coerce_json_arguments`.

    Returns ``None`` when the installed FastMCP predates middleware support, so
    an operator on an older pin keeps working rather than failing to import.
    """
    try:
        from fastmcp.server.middleware import Middleware, MiddlewareContext
    except ImportError:  # pragma: no cover - depends on the installed FastMCP
        return None

    class JsonArgumentMiddleware(Middleware):  # type: ignore[misc,valid-type]
        """Parse arguments a client serialised one time too many."""

        async def on_call_tool(self, context: MiddlewareContext, call_next):  # type: ignore[no-untyped-def]
            arguments = getattr(context.message, "arguments", None)
            if isinstance(arguments, dict) and arguments:
                properties = await self._properties(context)
                if properties:
                    coerced = coerce_json_arguments(arguments, properties)
                    if coerced:
                        logger.info(
                            "Parsed JSON-string argument(s) %s for %s — the client "
                            "serialised a declared object/array.",
                            ", ".join(coerced), getattr(context.message, "name", "?"),
                        )
            return await call_next(context)

        @staticmethod
        async def _properties(context: Any) -> dict[str, Any]:
            """The called tool's schema properties, or empty if unavailable.

            Best-effort by construction: a lookup failure must degrade to the
            old behaviour, never take the call down.
            """
            try:
                server = context.fastmcp_context.fastmcp
                tool = await server.get_tool(context.message.name)
                params = getattr(tool, "parameters", None) or {}
                props = params.get("properties")
                return props if isinstance(props, dict) else {}
            except Exception as exc:  # noqa: BLE001 — never break a tool call
                logger.debug("Could not read tool schema for arg coercion: %s", exc)
                return {}

    return JsonArgumentMiddleware()
