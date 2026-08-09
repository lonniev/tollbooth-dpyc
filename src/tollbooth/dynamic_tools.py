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
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

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

# Distinguishes "no default declared" from "default declared as None", which are
# different instructions: drop the param, versus bind an explicit null.
_UNSET = object()


def _is_of_type(value: Any, py: type | None) -> bool:
    """Does *value* satisfy the declared python type?

    ``bool`` is a subtype of ``int`` in Python, so the numeric checks exclude it
    explicitly — otherwise ``True`` would pass as an int and reach a query as 1.
    """
    if py is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if py is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if py is not None:
        return isinstance(value, py)
    return True


def validate_param_schema(param_schema: dict[str, Any]) -> list[str]:
    """Validate the shape of an author-supplied param schema. Returns errors.

    ``param_schema`` maps a param name to a spec ``{"type": ..., "required":
    true|false, "default": ..., "description": ...}``; ``type`` is one of
    :data:`PYTHON_TYPES`.

    A declared ``default`` must match the declared ``type``. It is what an
    omitted optional param binds to — see :func:`build_dynamic_handler` for why
    an optional param that binds to nothing is a latent break.
    """
    errors: list[str] = []
    if not isinstance(param_schema, dict):
        return ["param_schema must be an object mapping param name -> spec"]
    for name, spec in param_schema.items():
        if not isinstance(spec, dict):
            errors.append(f"param '{name}' spec must be an object")
            continue
        t = spec.get("type", "string")
        # An explicit ``None`` default means "bind null" — the idiom a backend
        # uses to branch on absence itself, e.g. Cypher's
        # ``coalesce($title, i.title)``. Everything else must type-match.
        if (
            t in PYTHON_TYPES
            and spec.get("default", _UNSET) not in (_UNSET, None)
            and not _is_of_type(spec["default"], PYTHON_TYPES[t])
        ):
            errors.append(f"param '{name}' default must be of type {t}")
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
        val = params[name]
        # A param whose schema declares a ``None`` default is nullable: null is
        # the value its backend expects when the caller said nothing.
        if val is None and spec.get("default", _UNSET) is None:
            continue
        if not _is_of_type(val, PYTHON_TYPES.get(t)):
            errors.append(f"param '{name}' must be of type {t}")

    for name in params:
        if name not in schema:
            errors.append(f"unexpected param '{name}'")

    return errors


def apply_param_defaults(
    param_schema: dict[str, Any], supplied: dict[str, Any] | None
) -> dict[str, Any]:
    """Materialize a schema's declared defaults into the params actually bound.

    A ``default`` in a param schema is a promise to the caller: omit this and the
    declared value is used. Validation alone cannot keep that promise — it accepts
    the omission and returns no error, and whatever binds the params afterwards
    sees nothing. A backend that interpolates every declared name then fails on the
    one that was never bound, and the caller gets an opaque execution error for a
    param they were told was optional.

    That is not hypothetical. cypher-mcp declares ``as_at_ms`` as
    ``{required: False, default: 0}``; calling the query through the named tool
    worked, while the same query by key failed with "Tool execution failed. Check
    operator logs." — because only the named-tool path filled the default. Two
    routes to one query disagreed about the contract.

    Only declared params survive: an unexpected key is dropped rather than bound,
    which keeps the surface tight the way ``validate_params`` intends. An optional
    param with no declared default stays absent, since some backends legitimately
    branch on absence.

    ``None`` counts as not supplied — a keyword-argument handler cannot distinguish
    "omitted" from "passed as null", and a declared default is the better answer for
    both. A param meant to accept null declares ``default: None`` and gets it.
    """
    schema = param_schema or {}
    given = supplied or {}
    params: dict[str, Any] = {}
    for name, spec in schema.items():
        value = given.get(name)
        if value is not None:
            params[name] = value
        elif "default" in spec:
            params[name] = spec["default"]
    return params


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
    collects the supplied declared params into a dict and awaits
    ``runner(params, npub, dpop_token)``.

    **An omitted optional param binds its declared ``default`` if it has one.**
    Dropping it instead is what broke cypher-mcp's ``list_capabilities`` for four
    days: a backend that interpolates every declared name — a Cypher template
    referencing ``$since_ms``, a SQL statement, a prompt slot — cannot run at all
    when the binding simply isn't there, and the caller sees a bare execution
    failure with nothing naming the missing parameter. Never passing ``None`` was
    right (a null would have reached the query as a silent wrong answer); the
    mistake was having nothing to pass in its place.

    An optional param with no declared default is still dropped, since some
    backends legitimately branch on absence — but authors of interpolating
    backends should declare one.
    """
    schema = param_schema or {}

    async def handler(**kwargs: Any) -> dict[str, Any]:
        return await runner(
            apply_param_defaults(schema, kwargs),
            kwargs.get("npub") or "",
            kwargs.get("dpop_token") or "",
        )

    sig_params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for pname, spec in schema.items():
        py = PYTHON_TYPES.get(spec.get("type", "string"), str)
        # Runtime-constructed annotation; mypy can't see `py` as a type.
        ann = Annotated[py, Field(description=spec.get("description", ""))]  # type: ignore[valid-type]
        required = spec.get("required", True)
        if required:
            sig_default: Any = inspect.Parameter.empty
        else:
            # Surface the author's default in the tool's published schema, so a
            # caller reading tools/list sees the value it will actually get
            # rather than a null the declared type does not even admit.
            sig_default = spec.get("default", None)
        sig_params.append(
            inspect.Parameter(
                pname,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=ann,
                default=sig_default,
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
