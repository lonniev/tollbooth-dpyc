"""Runtime tool synthesis — build typed MCP tools from a declarative spec.

Generic, domain-agnostic machinery: given a parameter schema plus a ``runner``
callback, produce a fully-typed MCP tool *at runtime*. The "named Cypher query"
in cypher-mcp is one consumer; any operator can back a synthesized tool with a
REST call, SQL, a stored prompt, etc., and reuse the identical primitive.

Schema-to-signature note: FastMCP builds a tool's input schema from the
function's ``__annotations__`` (via pydantic's ``TypeAdapter``), **not** from
``__signature__`` alone — a ``**kwargs`` body with only a synthetic signature
fails. So ``build_dynamic_handler`` sets BOTH a real ``__signature__`` and a
matching ``__annotations__`` (including ``"return"``). ``OperatorRuntime.
paid_tool``'s ``functools.wraps`` copies both onto its wrapper (``__annotations__``
is in ``WRAPPER_ASSIGNMENTS``; ``__signature__`` rides ``__dict__``), so the
typed surface survives the billing decorator.
"""

from __future__ import annotations

import inspect
import re
from typing import Annotated, Any, Awaitable, Callable

from pydantic import Field

# The param-schema language for dynamic tools (and operator catalogs).
# A richer language (jsonschema / nested objects) is a later upgrade.
PYTHON_TYPES: dict[str, type] = {
    "string": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
}

# A published tool name becomes a wire identifier: ``{slug}_{name}``.
VALID_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

# runner(params, npub, dpop_token) -> awaitable[dict]
Runner = Callable[[dict[str, Any], str, str], Awaitable[dict[str, Any]]]


def validate_param_schema(param_schema: dict[str, Any]) -> list[str]:
    """Validate the shape of an author-supplied param schema. Returns errors.

    ``param_schema`` maps a param name to a spec ``{"type": ..., "required":
    true|false, "description": ...}``; ``type`` is one of :data:`PYTHON_TYPES`.
    """
    errors: list[str] = []
    if not isinstance(param_schema, dict):
        return ["param_schema must be an object mapping param name -> spec"]
    for name, spec in param_schema.items():
        if not isinstance(spec, dict):
            errors.append(f"param '{name}' spec must be an object")
            continue
        t = spec.get("type", "string")
        if t not in PYTHON_TYPES:
            errors.append(
                f"param '{name}' has unknown type '{t}' "
                f"(allowed: {', '.join(sorted(PYTHON_TYPES))})"
            )
    return errors


def validate_params(
    param_schema: dict[str, Any], params: dict[str, Any] | None
) -> list[str]:
    """Validate incoming params against the schema. Returns errors.

    Fails cheap, before any side effect. Rejects missing required params,
    type mismatches, and unexpected params (tight surface). ``bool`` is a
    subtype of ``int`` in Python, so the numeric checks guard against it.
    """
    errors: list[str] = []
    params = params or {}
    schema = param_schema or {}

    for name, spec in schema.items():
        required = spec.get("required", True)
        if name not in params:
            if required:
                errors.append(f"missing required param '{name}'")
            continue
        t = spec.get("type", "string")
        py = PYTHON_TYPES.get(t)
        val = params[name]
        if py is int:
            ok = isinstance(val, int) and not isinstance(val, bool)
        elif py is float:
            ok = isinstance(val, (int, float)) and not isinstance(val, bool)
        elif py is not None:
            ok = isinstance(val, py)
        else:
            ok = True
        if not ok:
            errors.append(f"param '{name}' must be of type {t}")

    for name in params:
        if name not in schema:
            errors.append(f"unexpected param '{name}'")

    return errors


def build_dynamic_handler(
    name: str,
    param_schema: dict[str, Any],
    runner: Runner,
    *,
    intent: str = "",
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Build a typed async MCP handler that delegates to ``runner``.

    The handler exposes one flat keyword param per ``param_schema`` entry
    (typed via :data:`PYTHON_TYPES`) plus ``npub`` and ``dpop_token``. Its body
    collects the supplied declared params into a dict — omitted optional
    params are dropped, never passed as ``None`` — and awaits
    ``runner(params, npub, dpop_token)``.
    """
    schema = param_schema or {}

    async def handler(**kwargs: Any) -> dict[str, Any]:
        params = {k: v for k, v in kwargs.items() if k in schema and v is not None}
        return await runner(
            params, kwargs.get("npub") or "", kwargs.get("dpop_token") or ""
        )

    sig_params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for pname, spec in schema.items():
        py = PYTHON_TYPES.get(spec.get("type", "string"), str)
        # Runtime-constructed annotation; mypy can't see `py` as a type.
        ann = Annotated[py, Field(description=spec.get("description", ""))]  # type: ignore[valid-type]
        required = spec.get("required", True)
        sig_params.append(
            inspect.Parameter(
                pname,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=ann,
                default=inspect.Parameter.empty if required else None,
            )
        )
        annotations[pname] = ann
    for extra in ("npub", "dpop_token"):
        sig_params.append(
            inspect.Parameter(
                extra, inspect.Parameter.KEYWORD_ONLY, annotation=str, default=""
            )
        )
        annotations[extra] = str
    annotations["return"] = dict

    handler.__name__ = name
    handler.__qualname__ = name
    handler.__doc__ = intent or f"Run the '{name}' operation."
    handler.__signature__ = inspect.Signature(sig_params)  # type: ignore[attr-defined]
    handler.__annotations__ = annotations
    return handler
