"""Every wheel-provided tool reaches the wire titled, annotated and documented."""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from tollbooth.authority.tools import AUTHORITY_TOOL_REGISTRY, register_authority_tools
from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.params import PatronNpub
from tollbooth.runtime import OperatorRuntime, register_standard_tools
from tollbooth.tool_identity import READS, READS_REMOTE, STANDARD_IDENTITIES, ToolIdentity


@pytest.fixture
async def listed_tools(monkeypatch):
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    mcp = FastMCP("metadata-test")
    # Every conditional tool on: patron credentials and OAuth register their own.
    rt = OperatorRuntime(
        tool_registry={**STANDARD_IDENTITIES, **AUTHORITY_TOOL_REGISTRY},
        nsec_env_var="__UNUSED__",
        ots_enabled=True,
        patron_credential_template=CredentialTemplate(
            service="test-patron", version=1, description="Test.",
            fields={"api_key": FieldSpec(required=True, sensitive=True, description="Key.")},
        ),
        oauth_provider=object(),
    )
    register_standard_tools(mcp, "x", rt)
    register_authority_tools(mcp, rt, slug="x")

    async def runner(params: dict, npub: str) -> dict:
        return {}

    rt.register_dynamic_tool(
        name="published_query", title="Published query", behavior=READS_REMOTE,
        param_schema={"city": {"type": "string", "description": "A city."}},
        runner=runner,
    )
    async with Client(mcp) as client:
        return await client.list_tools()


@pytest.mark.asyncio
async def test_every_tool_has_a_title_and_behavior(listed_tools):
    names = {t.name for t in listed_tools}
    assert {"x_receive_patron_credentials", "x_begin_oauth", "x_published_query"} <= names
    for t in listed_tools:
        assert t.title, t.name
        hints = t.annotations
        assert hints is not None, t.name
        assert None not in (
            hints.readOnlyHint, hints.destructiveHint,
            hints.idempotentHint, hints.openWorldHint,
        ), t.name


@pytest.mark.asyncio
async def test_every_parameter_is_described(listed_tools):
    undocumented = [
        f"{t.name}.{name}"
        for t in listed_tools
        for name, schema in t.inputSchema.get("properties", {}).items()
        if not schema.get("description")
    ]
    assert undocumented == []


def test_identities_carry_their_own_title():
    for identity in (*STANDARD_IDENTITIES.values(), *AUTHORITY_TOOL_REGISTRY.values()):
        assert identity.title.strip(), identity.capability


def test_identity_for_finds_paid_tools_by_uuid_and_others_by_capability():
    paid = ToolIdentity(
        tool_id="uuid-paid", category="read", intent="Paid.",
        title="Paid tool", behavior=READS, capability="renamed_capability",
    )
    rt = OperatorRuntime(tool_registry={paid.tool_id: paid}, nsec_env_var="__UNUSED__")
    rt._tool_func_names[paid.tool_id] = "paid_fn"
    assert rt.identity_for("paid_fn") is paid
    assert rt.identity_for("check_balance").title == "Check balance"
    with pytest.raises(LookupError):
        rt.identity_for("nobody_declared_me")


def test_shared_params_describe_themselves():
    (field,) = PatronNpub.__metadata__
    assert field.description
