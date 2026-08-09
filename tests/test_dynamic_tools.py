"""Runtime tool synthesis — param-schema language, handler builder, and the
OperatorRuntime register/unregister surface.

The wheel test env has no FastMCP (consumers bring it), so registration is
exercised against a minimal fake MCP — the same approach as test_slug_tools.
One opt-in test confirms real FastMCP schema generation when fastmcp is present.
"""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

import pytest

from tollbooth.dynamic_tools import (
    apply_param_defaults,
    build_dynamic_handler,
    validate_param_schema,
    validate_params,
)
from tollbooth.runtime import OperatorRuntime
from tollbooth.slug_tools import make_slug_tool
from tollbooth.tool_identity import capability_uuid

# --------------------------------------------------------------------------
# Param-schema language
# --------------------------------------------------------------------------


class TestValidateParamSchema:
    def test_accepts_known_types(self) -> None:
        schema = {
            "a": {"type": "string"}, "b": {"type": "int"},
            "c": {"type": "float"}, "d": {"type": "bool"}, "e": {"type": "list"},
        }
        assert validate_param_schema(schema) == []

    def test_rejects_unknown_type(self) -> None:
        errs = validate_param_schema({"a": {"type": "datetime"}})
        assert any("unknown type 'datetime'" in e for e in errs)

    def test_rejects_non_object_spec(self) -> None:
        assert validate_param_schema({"a": "string"})  # spec must be a dict

    def test_rejects_non_dict_schema(self) -> None:
        assert validate_param_schema(["a", "b"])  # type: ignore[arg-type]


class TestDefaultTypeChecking:
    def test_default_must_match_declared_type(self) -> None:
        errs = validate_param_schema(
            {"since_ms": {"type": "int", "required": False, "default": "soon"}},
        )
        assert any("default must be of type int" in e for e in errs)

    def test_matching_default_is_accepted(self) -> None:
        assert validate_param_schema(
            {"since_ms": {"type": "int", "required": False, "default": 0}},
        ) == []

    def test_bool_is_not_an_int_default(self) -> None:
        """bool subclasses int in Python; a True default would reach a query as 1."""
        errs = validate_param_schema(
            {"since_ms": {"type": "int", "required": False, "default": True}},
        )
        assert any("default must be of type int" in e for e in errs)

    def test_none_default_declares_a_nullable_param(self) -> None:
        """``coalesce($title, i.title)`` wants a bound null, not a dropped name."""
        assert validate_param_schema(
            {"title": {"type": "string", "required": False, "default": None}},
        ) == []

    def test_nullable_param_accepts_null_at_call_time(self) -> None:
        schema = {"title": {"type": "string", "required": False, "default": None}}
        assert validate_params(schema, {"title": None}) == []

    def test_non_nullable_param_still_rejects_null(self) -> None:
        schema = {"title": {"type": "string", "required": False}}
        assert validate_params(schema, {"title": None}) != []


class TestValidateParams:
    schema = {  # noqa: RUF012
        "from_city": {"type": "string"},
        "to_city": {"type": "string"},
        "max_stops": {"type": "int", "required": False},
    }

    def test_all_good(self) -> None:
        assert validate_params(self.schema, {"from_city": "JFK", "to_city": "LHR"}) == []

    def test_missing_required(self) -> None:
        errs = validate_params(self.schema, {"from_city": "JFK"})
        assert any("missing required param 'to_city'" in e for e in errs)

    def test_optional_omitted_is_ok(self) -> None:
        assert validate_params(self.schema, {"from_city": "JFK", "to_city": "LHR"}) == []

    def test_type_mismatch(self) -> None:
        errs = validate_params(self.schema, {"from_city": "JFK", "to_city": "LHR", "max_stops": "two"})
        assert any("max_stops" in e and "int" in e for e in errs)

    def test_bool_is_not_int(self) -> None:
        # bool is a Python subtype of int; the language treats them distinctly.
        errs = validate_params({"n": {"type": "int"}}, {"n": True})
        assert any("must be of type int" in e for e in errs)

    def test_unexpected_param(self) -> None:
        errs = validate_params(self.schema, {"from_city": "JFK", "to_city": "LHR", "bogus": 1})
        assert any("unexpected param 'bogus'" in e for e in errs)


# --------------------------------------------------------------------------
# Handler builder
# --------------------------------------------------------------------------


SCHEMA = {
    "from_city": {"type": "string", "description": "origin"},
    "to_city": {"type": "string", "description": "destination"},
    "max_stops": {"type": "int", "description": "max layovers", "required": False},
}


class TestBuildDynamicHandler:
    def test_signature_and_annotations(self) -> None:
        async def runner(params: dict, npub: str, dpop_token: str) -> dict:
            return {}

        h = build_dynamic_handler("find_airline_flights", SCHEMA, runner, intent="Find flights.")
        assert h.__name__ == "find_airline_flights"
        assert h.__doc__ == "Find flights."

        params = inspect.signature(h).parameters
        assert list(params) == ["from_city", "to_city", "max_stops", "npub", "dpop_token"]
        # required params have no default; optional + npub/dpop_token do.
        assert params["from_city"].default is inspect.Parameter.empty
        assert params["max_stops"].default is None
        assert params["npub"].default == ""

        hints = get_type_hints(h)  # resolves Annotated → base type
        assert hints["from_city"] is str
        assert hints["max_stops"] is int
        assert hints["return"] is dict

    async def test_delegates_and_drops_omitted_optionals(self) -> None:
        captured: dict[str, Any] = {}

        async def runner(params: dict, npub: str, dpop_token: str) -> dict:
            captured["params"] = params
            captured["npub"] = npub
            captured["dpop_token"] = dpop_token
            return {"ok": True}

        h = build_dynamic_handler("find_airline_flights", SCHEMA, runner)
        # max_stops omitted → must NOT appear as None in the runner's params.
        out = await h(from_city="JFK", to_city="LHR", max_stops=None, npub="np", dpop_token="pf")
        assert out == {"ok": True}
        assert captured["params"] == {"from_city": "JFK", "to_city": "LHR"}
        assert captured["npub"] == "np" and captured["dpop_token"] == "pf"

    async def test_passes_supplied_optionals(self) -> None:
        captured: dict[str, Any] = {}

        async def runner(params: dict, npub: str, dpop_token: str) -> dict:
            captured["params"] = params
            return {}

        h = build_dynamic_handler("find_airline_flights", SCHEMA, runner)
        await h(from_city="JFK", to_city="LHR", max_stops=1)
        assert captured["params"] == {"from_city": "JFK", "to_city": "LHR", "max_stops": 1}

    async def test_omitted_optional_binds_its_declared_default(self) -> None:
        """The four-day cypher-mcp outage, in miniature.

        ``list_capabilities`` declared ``since_ms`` optional and its Cypher
        referenced ``$since_ms`` unconditionally. Omitting it dropped the name
        entirely, so the query ran against a parameter that was never bound and
        died — every no-argument call failing, refunded, for four days. A backend
        that interpolates each declared name needs *something* to bind.
        """
        captured: dict[str, Any] = {}

        async def runner(params: dict, npub: str, dpop_token: str) -> dict:
            captured["params"] = params
            return {}

        schema = {
            "since_ms": {"type": "int", "required": False, "default": 0},
            "note": {"type": "string", "required": False, "default": ""},
        }
        h = build_dynamic_handler("list_capabilities", schema, runner)
        await h()
        assert captured["params"] == {"since_ms": 0, "note": ""}

        # A supplied value still wins over the default.
        await h(since_ms=123)
        assert captured["params"]["since_ms"] == 123

    async def test_declared_default_is_published_in_the_signature(self) -> None:
        """A caller reading tools/list must see the value it will actually get,
        not a null that the declared type does not even admit."""

        async def runner(params: dict, npub: str, dpop_token: str) -> dict:
            return {}

        schema = {"since_ms": {"type": "int", "required": False, "default": 0}}
        h = build_dynamic_handler("list_capabilities", schema, runner)
        assert inspect.signature(h).parameters["since_ms"].default == 0

    async def test_optional_without_default_is_still_dropped(self) -> None:
        """Absence stays expressible — some backends branch on it."""
        captured: dict[str, Any] = {}

        async def runner(params: dict, npub: str, dpop_token: str) -> dict:
            captured["params"] = params
            return {}

        h = build_dynamic_handler("find_airline_flights", SCHEMA, runner)
        await h(from_city="JFK", to_city="LHR")
        assert "max_stops" not in captured["params"]


# --------------------------------------------------------------------------
# OperatorRuntime.register/unregister_dynamic_tool (fake MCP)
# --------------------------------------------------------------------------


class _FakeMCP:
    """Minimal stand-in: a slug-prefixed tool() decorator + remove_tool()."""

    def __init__(self) -> None:
        self.registered: dict[str, Any] = {}
        self.removed: list[str] = []

    def tool(self, *, name: str):
        def decorator(func):
            self.registered[name] = func
            return func
        return decorator

    def remove_tool(self, name: str, version: str | None = None) -> None:
        self.removed.append(name)
        self.registered.pop(name, None)


def _wired_runtime(slug: str = "cypher") -> tuple[OperatorRuntime, _FakeMCP]:
    rt = OperatorRuntime()
    fake = _FakeMCP()
    rt._slug = slug
    rt._mcp = fake
    rt._tool = make_slug_tool(fake, slug)
    return rt, fake


async def _runner(params: dict, npub: str, dpop_token: str) -> dict:
    return {"params": params}


class TestRegisterDynamicTool:
    def test_registers_typed_unpriced_tool(self) -> None:
        rt, fake = _wired_runtime()
        name = rt.register_dynamic_tool(
            name="find_airline_flights", param_schema=SCHEMA, runner=_runner,
            intent="Find flights.",
        )
        assert name == "cypher_find_airline_flights"
        assert "cypher_find_airline_flights" in fake.registered

        tool_id = capability_uuid("dyn:find_airline_flights")
        ident = rt._tool_registry[tool_id]
        assert ident.capability == "find_airline_flights"
        assert ident.category == "read"
        assert ident.pricing_hint_value == 0  # unpriced until Studio prices it
        assert rt._tool_func_names[tool_id] == "find_airline_flights"
        assert rt.mcp_name_for(tool_id) == "cypher_find_airline_flights"

    def test_idempotent_replace(self) -> None:
        rt, fake = _wired_runtime()
        rt.register_dynamic_tool(name="q", param_schema={"a": {"type": "string"}}, runner=_runner)
        rt.register_dynamic_tool(name="q", param_schema={"b": {"type": "int"}}, runner=_runner)
        # one identity, the prior wire tool was removed before re-adding.
        ids = [tid for tid, i in rt._tool_registry.items() if i.capability == "q"]
        assert len(ids) == 1
        assert "cypher_q" in fake.removed
        assert "cypher_q" in fake.registered


class TestRuntimeNameFrozenUuidRename:
    """A tool renamed after launch keeps its ORIGINAL (frozen) UUID so pricing
    and identity proofs stay stable. ``runtime_name`` must resolve such a tool
    through its registered identity, NOT by re-hashing the current capability
    string — otherwise the expected proof ``u`` tag becomes the raw UUID and
    every owner-consent proof fails (masked as authority_consent_required).
    Regression for the 0.71.0 network_books_health → network_persistence_health
    rename.
    """

    def test_renamed_capability_resolves_via_frozen_id(self) -> None:
        from tollbooth.tool_identity import ToolIdentity

        rt, _ = _wired_runtime(slug="authority")
        # tool_id frozen to the OLD name's hash; capability is the NEW string.
        frozen = capability_uuid("network_books_health")
        rt._tool_registry[frozen] = ToolIdentity(
            tool_id=frozen,
            capability="network_persistence_health",
            category="restricted",
            intent="Owner persistence health.",
        )

        # Resolves to the real wire name (registered identity), not the raw
        # hash of the renamed string.
        assert (
            rt.runtime_name("network_persistence_health")
            == "authority_network_persistence_health"
        )
        # And critically NOT the recomputed-hash miss (the pre-fix bug).
        assert rt.runtime_name("network_persistence_health") != capability_uuid(
            "network_persistence_health"
        )

    def test_unrenamed_capability_still_resolves(self) -> None:
        """The common case — capability string == the UUID's seed — is unchanged."""
        rt, _ = _wired_runtime()
        rt.register_dynamic_tool(
            name="find_airline_flights", param_schema=SCHEMA, runner=_runner,
            intent="Find flights.",
        )
        assert rt.runtime_name("find_airline_flights") == "cypher_find_airline_flights"

    def test_rejects_bad_name(self) -> None:
        rt, _ = _wired_runtime()
        with pytest.raises(ValueError, match="must match"):
            rt.register_dynamic_tool(name="Find Flights", param_schema={}, runner=_runner)

    def test_rejects_bad_schema(self) -> None:
        rt, _ = _wired_runtime()
        with pytest.raises(ValueError, match="invalid param_schema"):
            rt.register_dynamic_tool(
                name="q", param_schema={"a": {"type": "datetime"}}, runner=_runner,
            )

    def test_requires_register_standard_tools(self) -> None:
        rt = OperatorRuntime()  # _mcp / _tool still None
        with pytest.raises(RuntimeError, match="register_standard_tools"):
            rt.register_dynamic_tool(name="q", param_schema={}, runner=_runner)

    def test_custom_uuid_is_honored(self) -> None:
        rt, _ = _wired_runtime()
        rt.register_dynamic_tool(
            name="q", param_schema={}, runner=_runner, uuid="11111111-1111-5111-8111-111111111111",
        )
        assert "11111111-1111-5111-8111-111111111111" in rt._tool_registry


class TestUnregisterDynamicTool:
    def test_removes_by_name(self) -> None:
        rt, fake = _wired_runtime()
        rt.register_dynamic_tool(name="find_airline_flights", param_schema=SCHEMA, runner=_runner)
        assert rt.unregister_dynamic_tool("find_airline_flights") is True
        assert "cypher_find_airline_flights" in fake.removed
        assert capability_uuid("dyn:find_airline_flights") not in rt._tool_registry

    def test_removes_by_uuid(self) -> None:
        rt, _fake = _wired_runtime()
        rt.register_dynamic_tool(name="q", param_schema={}, runner=_runner)
        tid = capability_uuid("dyn:q")
        assert rt.unregister_dynamic_tool(tid) is True
        assert tid not in rt._tool_registry

    def test_missing_raises(self) -> None:
        rt, _ = _wired_runtime()
        with pytest.raises(ValueError, match="no dynamic tool"):
            rt.unregister_dynamic_tool("nope")

    def test_missing_quiet_is_false(self) -> None:
        rt, _ = _wired_runtime()
        assert rt.unregister_dynamic_tool("nope", _quiet=True) is False


# --------------------------------------------------------------------------
# Real FastMCP schema generation (opt-in — only where fastmcp is installed)
# --------------------------------------------------------------------------


def test_fastmcp_generates_typed_schema() -> None:
    pytest.importorskip("fastmcp")
    import functools

    from fastmcp.tools import Tool

    async def runner(params: dict, npub: str, dpop_token: str) -> dict:
        return {}

    h = build_dynamic_handler("find_airline_flights", SCHEMA, runner)

    # Simulate paid_tool's functools.wraps wrapper preserving the typed surface.
    @functools.wraps(h)
    async def wrapped(*a, **k):  # pragma: no cover - schema-gen path only
        return await h(*a, **k)

    t = Tool.from_function(wrapped, name="cypher_find_airline_flights")
    props = t.parameters["properties"]
    assert props["from_city"]["type"] == "string"
    assert props["max_stops"]["type"] == "integer"
    assert set(t.parameters["required"]) == {"from_city", "to_city"}


# ---------------------------------------------------------------------------
# apply_param_defaults — validation accepts an omission; something must still bind it
# ---------------------------------------------------------------------------


_AS_AT = {
    "name": {"type": "string", "required": True},
    "as_at_ms": {"type": "int", "required": False, "default": 0},
}


def test_an_omitted_optional_param_binds_its_declared_default() -> None:
    """The cypher-mcp regression: validation passes, then the bind has no value.

    `validate_params` returns no error for an omitted optional param, so a backend
    that interpolates every declared name fails on the one nothing bound — and the
    caller sees an opaque execution error for a param they were told was optional.
    """
    assert validate_params(_AS_AT, {"name": "cap"}) == [], "omission must be legal"
    assert apply_param_defaults(_AS_AT, {"name": "cap"}) == {"name": "cap", "as_at_ms": 0}


def test_a_supplied_value_beats_the_default() -> None:
    assert apply_param_defaults(_AS_AT, {"name": "c", "as_at_ms": 42})["as_at_ms"] == 42


def test_an_optional_param_with_no_default_stays_absent() -> None:
    """Some backends branch on absence; inventing a value would change behaviour."""
    schema = {"q": {"type": "string", "required": False}}
    assert apply_param_defaults(schema, {}) == {}


def test_undeclared_params_are_dropped_not_bound() -> None:
    """Keeps the surface as tight as validate_params intends."""
    assert apply_param_defaults(_AS_AT, {"name": "c", "sneaky": "x"}) == {
        "name": "c",
        "as_at_ms": 0,
    }


def test_null_is_treated_as_omitted_and_takes_the_default() -> None:
    """A kwargs handler cannot tell 'omitted' from 'passed None'."""
    assert apply_param_defaults(_AS_AT, {"name": "c", "as_at_ms": None})["as_at_ms"] == 0


def test_a_param_declaring_a_none_default_still_binds_none() -> None:
    schema = {"cursor": {"type": "string", "required": False, "default": None}}
    assert apply_param_defaults(schema, {}) == {"cursor": None}


@pytest.mark.asyncio
async def test_the_named_tool_path_still_fills_defaults_after_the_extraction() -> None:
    """Behaviour-preserving: build_dynamic_handler delegates now, same result."""
    seen: dict[str, Any] = {}

    async def runner(params: dict, npub: str, dpop_token: str) -> dict:
        seen.update(params)
        return {"ok": True}

    handler = build_dynamic_handler("t", _AS_AT, runner)
    await handler(name="cap", npub="npub1", dpop_token="tok")
    assert seen == {"name": "cap", "as_at_ms": 0}
