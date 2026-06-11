"""Tests for tollbooth.authority.role_migration (audit M2.4).

Per-operator Postgres role isolation: create a LOGIN role, transfer schema/table
ownership, revoke Authority DML, store operator-scoped credentials. Was 0%
covered. These exercise the orchestration with the tenant_provisioner functions
patched; the CLI _main is left out (env-driven entry point).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

import tollbooth.authority.role_migration as rm

BASE_URL = "postgresql://authrole:pw@host/db"
NSEC_HEX = PrivateKey().hex()


def _provisioner_mocks(**over):
    """Build a dict of mocks for the tenant_provisioner functions, keeping
    references so tests can assert on them (patch.multiple's context return
    only includes DEFAULT-created mocks)."""
    mocks = {
        "get_all_operator_config": AsyncMock(return_value={"schema": "op_xyz"}),
        "generate_operator_password": MagicMock(return_value="genpw"),
        "create_operator_role": AsyncMock(),
        "transfer_schema_ownership": AsyncMock(),
        "extract_authority_role": MagicMock(return_value="authrole"),
        "revoke_authority_access": AsyncMock(),
        "neon_url_for_operator": MagicMock(return_value="postgresql://op_xyz:genpw@host/db"),
        "store_operator_config": AsyncMock(),
    }
    mocks.update(over)
    return mocks


@pytest.mark.asyncio
async def test_migrate_single_no_schema_errors():
    mocks = _provisioner_mocks(get_all_operator_config=AsyncMock(return_value={}))
    with patch.multiple(rm, **mocks):
        r = await rm.migrate_single_operator(MagicMock(), "npub1op", BASE_URL, NSEC_HEX)
    assert r == {"success": False, "npub": "npub1op", "error": "No schema in bootstrap_config"}


@pytest.mark.asyncio
async def test_migrate_single_happy_path_calls_steps_and_stores_creds():
    mocks = _provisioner_mocks()
    vault = MagicMock()
    with patch.multiple(rm, **mocks):
        r = await rm.migrate_single_operator(vault, "npub1op", BASE_URL, NSEC_HEX)

    assert r == {"success": True, "npub": "npub1op", "schema": "op_xyz"}
    mocks["create_operator_role"].assert_awaited_once_with(vault, "op_xyz", "genpw")
    mocks["transfer_schema_ownership"].assert_awaited_once_with(vault, "op_xyz")
    mocks["revoke_authority_access"].assert_awaited_once_with(vault, "op_xyz", "authrole")
    # two stores: encrypted role_password + new operator URL
    store = mocks["store_operator_config"]
    keys_stored = {c.args[2] for c in store.await_args_list}
    assert keys_stored == {"role_password", "neon_database_url"}
    # the password was encrypted (not the raw "genpw")
    pw_call = next(c for c in store.await_args_list if c.args[2] == "role_password")
    assert pw_call.args[3] != "genpw" and pw_call.args[3]


@pytest.mark.asyncio
async def test_migrate_single_step_failure_returns_error():
    mocks = _provisioner_mocks(create_operator_role=AsyncMock(side_effect=RuntimeError("role exists")))
    with patch.multiple(rm, **mocks):
        r = await rm.migrate_single_operator(MagicMock(), "npub1op", BASE_URL, NSEC_HEX)
    assert r["success"] is False
    assert r["schema"] == "op_xyz"
    assert "role exists" in r["error"]


@pytest.mark.asyncio
async def test_migrate_all_enumerates_and_batches():
    vault = MagicMock()
    # rows in list form + dict form + an empty npub that should be skipped
    vault._execute = AsyncMock(return_value={"rows": [["npub_a"], {"npub": "npub_b"}, [""]]})
    del vault._t  # force the identity table-name fallback

    single = AsyncMock(side_effect=lambda v, npub, *a: {"success": True, "npub": npub})
    with patch.object(rm, "migrate_single_operator", single):
        results = await rm.migrate_all_operators(vault, BASE_URL, NSEC_HEX)

    assert [r["npub"] for r in results] == ["npub_a", "npub_b"]  # empty skipped
    assert single.await_count == 2
