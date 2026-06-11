"""Tests for tollbooth.authority.tenant_provisioner pure helpers + role/config
ops (audit M2.4). Complements test_tenant_provisioner_grants.py. Was ~51%.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from unittest.mock import AsyncMock

from tollbooth.authority import tenant_provisioner as tp

BASE = "postgresql://authrole:authpw@db.host:5432/neondb?sslmode=require"


# ── pure helpers ──────────────────────────────────────────────────────

def test_schema_name_for_npub_is_deterministic_and_safe():
    a = tp.schema_name_for_npub("npub1abc")
    assert a == tp.schema_name_for_npub("npub1abc")          # deterministic
    assert a.startswith("op_") and len(a) == len("op_") + 16
    assert tp.schema_name_for_npub("npub1xyz") != a          # collision-resistant
    # the derived name passes the safe-identifier gate
    tp._validate_schema_name(a)


def test_validate_schema_name_rejects_injection():
    for bad in ("Op_x", "1op", "op-x", 'op";DROP', "op x", ""):
        with pytest.raises(ValueError, match="Unsafe schema name"):
            tp._validate_schema_name(bad)


def test_neon_url_with_schema_sets_search_path():
    url = tp.neon_url_with_schema(BASE, "op_abc")
    q = parse_qs(urlparse(url).query)
    assert q["options"] == ["-c search_path=op_abc"]
    assert q["sslmode"] == ["require"]  # existing params preserved


def test_neon_url_with_schema_rejects_unsafe():
    with pytest.raises(ValueError):
        tp.neon_url_with_schema(BASE, "bad-schema")


def test_extract_authority_role():
    assert tp.extract_authority_role(BASE) == "authrole"
    assert tp.extract_authority_role("postgresql://db.host/neondb") == ""


def test_neon_url_for_operator_swaps_creds_and_encodes_password():
    url = tp.neon_url_for_operator(BASE, "op_abc", "p@ss/w+rd")
    parsed = urlparse(url)
    assert parsed.username == "op_abc"
    # password is percent-encoded (no raw / @ + in the netloc)
    assert "p%40ss%2Fw%2Brd" in url
    assert parsed.hostname == "db.host" and parsed.port == 5432
    assert parse_qs(parsed.query)["options"] == ["-c search_path=op_abc"]


def test_generate_operator_password_is_urlsafe_and_nonempty():
    pw = tp.generate_operator_password()
    assert pw and tp._SAFE_PASSWORD.match(pw)
    assert pw != tp.generate_operator_password()  # random


# ── create_operator_role ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_role_rejects_unsafe_inputs():
    vault = AsyncMock()
    with pytest.raises(ValueError, match="Unsafe schema"):
        await tp.create_operator_role(vault, "bad-schema", "goodpw")
    with pytest.raises(ValueError, match="unsafe characters"):
        await tp.create_operator_role(vault, "op_abc", "bad pw!")


@pytest.mark.asyncio
async def test_create_role_happy_creates_and_grants():
    vault = AsyncMock()
    await tp.create_operator_role(vault, "op_abc", "genpw")
    sqls = " | ".join(c.args[0] for c in vault._execute.await_args_list)
    assert 'CREATE ROLE "op_abc" WITH LOGIN PASSWORD' in sqls
    assert 'GRANT "op_abc" TO CURRENT_USER' in sqls
    assert 'GRANT USAGE ON SCHEMA "op_abc"' in sqls


@pytest.mark.asyncio
async def test_create_role_resets_password_when_exists():
    calls = []

    async def _exec(sql, *a):
        calls.append(sql)
        if sql.startswith("CREATE ROLE"):
            raise RuntimeError("role already exists")
        return {}

    vault = AsyncMock()
    vault._execute = _exec
    await tp.create_operator_role(vault, "op_abc", "genpw")
    assert any(s.startswith('ALTER ROLE "op_abc" WITH PASSWORD') for s in calls)


@pytest.mark.asyncio
async def test_create_role_reraises_non_exists_error():
    async def _exec(sql, *a):
        if sql.startswith("CREATE ROLE"):
            raise RuntimeError("permission denied")
        return {}

    vault = AsyncMock()
    vault._execute = _exec
    with pytest.raises(RuntimeError, match="permission denied"):
        await tp.create_operator_role(vault, "op_abc", "genpw")


# ── config CRUD row parsing ───────────────────────────────────────────

def _vault_no_t():
    """AsyncMock vault with a sync _t identity (so _t() doesn't auto-await)."""
    vault = AsyncMock()
    vault._t = lambda t: t
    return vault


@pytest.mark.asyncio
async def test_get_config_value_list_and_dict_rows():
    vault = _vault_no_t()
    vault._execute = AsyncMock(return_value={"rows": [["the-value"]]})
    assert await tp.get_operator_config_value(vault, "npub", "k") == "the-value"

    vault._execute = AsyncMock(return_value={"rows": [{"value": "dv"}]})
    assert await tp.get_operator_config_value(vault, "npub", "k") == "dv"

    vault._execute = AsyncMock(return_value={"rows": []})
    assert await tp.get_operator_config_value(vault, "npub", "k") is None


@pytest.mark.asyncio
async def test_get_all_config_merges_rows():
    vault = _vault_no_t()
    vault._execute = AsyncMock(return_value={"rows": [["schema", "op_abc"], {"key": "neon_database_url", "value": "url"}]})
    cfg = await tp.get_all_operator_config(vault, "npub")
    assert cfg == {"schema": "op_abc", "neon_database_url": "url"}


@pytest.mark.asyncio
async def test_store_operator_config_upserts():
    vault = _vault_no_t()
    await tp.store_operator_config(vault, "npub", "schema", "op_abc")
    sql, params = vault._execute.await_args.args[0], vault._execute.await_args.args[1]
    assert "ON CONFLICT (npub, key)" in sql
    assert params == ["npub", "schema", "op_abc"]
