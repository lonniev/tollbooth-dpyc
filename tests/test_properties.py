"""Properties: what may be tuned, who may tune it, and what is kept."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tollbooth.properties import (
    STANDARD_PATRON_PROPERTIES,
    PropertyError,
    PropertySpec,
    PropertyStore,
    check_declarations,
    check_name,
    listing,
    llm_model_property,
    validate,
)
from tollbooth.runtime import OperatorRuntime
from tollbooth.tools.properties import (
    get_my_properties_tool,
    get_operator_properties_tool,
    set_my_property_tool,
    set_operator_property_tool,
)

OPERATOR = "npub1operator"
PATRON = "npub1patron"
OTHER = "npub1other"

MODEL = llm_model_property("Reference edit", suggestions=("google/gemini-2.5-flash",))
UNITS = PropertySpec(kind="choice", default="F", label="Units", choices=("F", "C"))
ORDER = PropertySpec(kind="list", default=[], label="Chart order", choices=("gdd", "frost", "rain"))
KWH = PropertySpec(kind="number", default=15, label="kWh price", minimum=0, maximum=200)
BEES = PropertySpec(kind="flag", default=True, label="Bees", follows="device")
NOTE = PropertySpec(kind="text", default="", label="Note")


# ── What a value may be ──────────────────────────────────────────────────


@pytest.mark.parametrize("raw,kept", [
    ("writer", "writer"), ("READER", "reader"), (" google/gemini-2.5-flash ", "google/gemini-2.5-flash"),
    ("openai/gpt-5-mini", "openai/gpt-5-mini"), ("anthropic/claude-sonnet-4.6:thinking", "anthropic/claude-sonnet-4.6:thinking"),
])
def test_a_model_is_a_tier_or_a_slug(raw, kept):
    assert validate(MODEL, raw) == kept


@pytest.mark.parametrize("raw", [
    "", "gpt", "../etc/passwd", "a/b c", "x/<script>", "a/b\nc", "A" * 60 + "/" + "b" * 70, 7, None,
])
def test_a_model_refuses_anything_else(raw):
    with pytest.raises(PropertyError):
        validate(MODEL, raw)


def test_each_kind_keeps_only_what_it_declares():
    assert validate(UNITS, "C") == "C"
    assert validate(ORDER, '["rain", "gdd", "rain"]') == ["rain", "gdd"]
    assert validate(KWH, "12.5") == 12.5 and validate(KWH, 12) == 12
    assert validate(BEES, "false") is False and validate(BEES, True) is True
    assert validate(NOTE, "  plain words ") == "plain words"
    for spec, raw in [(UNITS, "K"), (ORDER, '["hail"]'), (ORDER, "not json"), (KWH, "-1"),
                      (KWH, "nan"), (KWH, True), (BEES, "yes"), (NOTE, "a\x00b"), (NOTE, "x" * 201)]:
        with pytest.raises(PropertyError):
            validate(spec, raw)


def test_a_declaration_that_could_never_be_kept_fails_at_start():
    with pytest.raises(ValueError):
        PropertySpec(kind="choice", default="K", label="Units", choices=("F", "C"))
    with pytest.raises(ValueError):
        PropertySpec(kind="dial", default=1, label="?")
    with pytest.raises(ValueError):
        check_declarations({"Bad-Name": UNITS})


def test_an_unknown_name_is_refused_with_the_names_there_are():
    with pytest.raises(PropertyError, match="units"):
        check_name("unts", {"units": UNITS})


def test_listing_shows_defaults_and_reads_a_stale_value_as_the_default():
    rows = listing({"units": UNITS, "model": MODEL},
                   {"units": ("C", "2026-10-29"), "model": ("no slug", "2026-10-28")})
    by = {r["name"]: r for r in rows}
    assert (by["units"]["value"], by["units"]["is_default"]) == ("C", False)
    assert (by["model"]["value"], by["model"]["is_default"]) == ("writer", True)
    assert by["model"]["suggestions"] == ["google/gemini-2.5-flash"]


# ── What is kept ─────────────────────────────────────────────────────────


class FakeVault:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], Any] = {}
        self.reads = 0

    def _t(self, table: str) -> str:
        return f"op.{table}"

    async def _execute(self, sql: str, params: list[Any] | None = None) -> dict[str, Any]:
        p = params or []
        if sql.startswith("SELECT"):
            self.reads += 1
            return {"rows": [{"name": n, "value": v, "updated_at": "2026-10-29"}
                             for (o, n), v in self.rows.items() if o == p[0]]}
        if sql.startswith("INSERT"):
            self.rows[(p[0], p[1])] = p[2]
            return {"rowCount": 1}
        if sql.startswith("DELETE"):
            gone = [k for k in self.rows if k[0] == p[0] and (len(p) == 1 or k[1] == p[1])]
            for k in gone:
                del self.rows[k]
            return {"rowCount": len(gone)}
        raise AssertionError(sql)


def fake_rt(vault: FakeVault, *, proves: set[str]) -> SimpleNamespace:
    store = PropertyStore(vault)

    async def require_caller_proof(npub):
        return None if npub in proves else {"success": False, "error_code": "proof_required"}

    async def property_store():
        return store

    return SimpleNamespace(
        operator_npub=lambda: OPERATOR,
        operator_properties={"reference_edit_model": MODEL},
        patron_properties={**STANDARD_PATRON_PROPERTIES, "units": UNITS, "bees": BEES},
        require_caller_proof=require_caller_proof,
        property_store=property_store,
    )


async def test_the_operator_sets_a_model_and_anyone_reads_it():
    vault = FakeVault()
    rt = fake_rt(vault, proves={OPERATOR})
    out = await set_operator_property_tool(rt, "reference_edit_model", "google/gemini-2.5-flash")
    assert out == {"success": True, "name": "reference_edit_model",
                   "value": "google/gemini-2.5-flash", "is_default": False}
    read = await get_operator_properties_tool(fake_rt(vault, proves=set()))
    assert read["properties"][0]["value"] == "google/gemini-2.5-flash"

    back = await set_operator_property_tool(rt, "reference_edit_model", "")
    assert back["is_default"] is True and not vault.rows


async def test_only_the_operator_sets_an_operator_property():
    vault = FakeVault()
    out = await set_operator_property_tool(fake_rt(vault, proves={PATRON}), "reference_edit_model", "x/y")
    assert out["error_code"] == "proof_required" and not vault.rows


async def test_a_bad_value_is_refused_and_nothing_is_kept():
    vault = FakeVault()
    out = await set_operator_property_tool(fake_rt(vault, proves={OPERATOR}), "reference_edit_model", "x/<b>")
    assert out["success"] is False and "slug" in out["error"] and not vault.rows


async def test_a_patron_sets_and_reads_only_their_own():
    vault = FakeVault()
    mine = fake_rt(vault, proves={PATRON})
    assert (await set_my_property_tool(mine, PATRON, "units", "C"))["value"] == "C"
    read = {p["name"]: p for p in (await get_my_properties_tool(mine, PATRON))["properties"]}
    assert read["units"]["value"] == "C" and read["tour"]["is_default"]

    theirs = fake_rt(vault, proves={OTHER})
    assert (await get_my_properties_tool(theirs, PATRON))["error_code"] == "proof_required"
    assert (await set_my_property_tool(theirs, PATRON, "units", "F"))["error_code"] == "proof_required"
    assert json.loads(vault.rows[(PATRON, "units")]) == "C"


async def test_a_device_property_is_never_kept_here():
    vault = FakeVault()
    rt = fake_rt(vault, proves={PATRON})
    out = await set_my_property_tool(rt, PATRON, "bees", "false")
    assert out["success"] is False and "device" in out["error"] and not vault.rows


async def test_a_patron_cannot_set_an_operator_property_through_their_own():
    vault = FakeVault()
    out = await set_my_property_tool(fake_rt(vault, proves={PATRON}), PATRON, "reference_edit_model", "x/y")
    assert out["success"] is False and not vault.rows


# ── The runtime's read ───────────────────────────────────────────────────


def runtime(vault: Any) -> OperatorRuntime:
    rt = OperatorRuntime(nsec_env_var="__UNUSED__",
                         operator_properties={"reference_edit_model": MODEL})
    rt.operator_npub = lambda: OPERATOR  # type: ignore[method-assign]
    rt._property_store = PropertyStore(vault)
    return rt


async def test_an_unset_operator_property_is_its_default_and_a_set_one_is_cached():
    vault = FakeVault()
    rt = runtime(vault)
    assert await rt.operator_property("reference_edit_model") == "writer"
    await rt._property_store.set(OPERATOR, "reference_edit_model", "openai/gpt-5-mini")
    assert await rt.operator_property("reference_edit_model") == "openai/gpt-5-mini"
    reads = vault.reads
    await rt.operator_property("reference_edit_model")
    assert vault.reads == reads, "read from the per-process cache"


async def test_an_unreadable_property_answers_its_default():
    class Down(FakeVault):
        async def _execute(self, sql, params=None):
            raise RuntimeError("neon is asleep")
    assert await runtime(Down()).operator_property("reference_edit_model") == "writer"


def test_every_patron_has_the_standard_properties():
    rt = OperatorRuntime(nsec_env_var="__UNUSED__", patron_properties={"units": UNITS})
    assert {"theme", "timezone", "tour", "units"} <= set(rt.patron_properties)


async def test_operator_property_tools_appear_only_when_something_is_declared(monkeypatch):
    from fastmcp import Client, FastMCP

    from tollbooth.runtime import register_standard_tools
    from tollbooth.tool_identity import STANDARD_IDENTITIES

    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)

    async def names(**kw) -> set[str]:
        mcp = FastMCP("t")
        rt = OperatorRuntime(tool_registry=dict(STANDARD_IDENTITIES), nsec_env_var="__UNUSED__", **kw)
        register_standard_tools(mcp, "x", rt)
        async with Client(mcp) as c:
            return {t.name for t in await c.list_tools()}

    plain = await names()
    assert "x_set_my_property" in plain and "x_set_operator_property" not in plain
    assert "x_set_operator_property" in await names(operator_properties={"m": MODEL})


async def test_forgetting_a_patron_clears_their_properties_and_not_the_operators():
    from tollbooth.tools.courier import forget_credentials_tool

    vault = FakeVault()
    store = PropertyStore(vault)
    await store.set(PATRON, "units", "C")
    await store.set(OPERATOR, "reference_edit_model", "x/y")

    async def ok(*a, **kw):
        return None

    class Courier:
        async def forget(self, npub, service):
            return {"success": True}

    class Revocations:
        async def revoke_all(self, npub):
            return None

    async def courier():
        return Courier()

    async def revocations():
        return Revocations()

    async def property_store():
        return store

    rt = SimpleNamespace(
        operator_credential_service="op-svc", patron_credential_service="patron-svc",
        require_caller_proof=ok, courier=courier, proof_grant_revocations=revocations,
        property_store=property_store, holder_auth_enabled=False, _on_forget=None, _cashier=None,
    )
    out = await forget_credentials_tool(rt, "patron-svc", PATRON)
    assert out["properties_cleared"] == 1
    assert list(vault.rows) == [(OPERATOR, "reference_edit_model")]
