"""Tests for ToolAclStore — CRUD via Neon HTTP API.

Follows the same mock pattern as test_pricing_store.py: patches
``vault._client.post`` with AsyncMock.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from tollbooth.acl_store import ToolAclStore
from tollbooth.vaults.neon import NeonVault


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_pricing_store.py)
# ---------------------------------------------------------------------------

DATABASE_URL = "postgresql://user:password@ep-test.us-east-2.aws.neon.tech/testdb"
HTTP_ENDPOINT = "https://ep-test.us-east-2.aws.neon.tech/sql"


def _vault() -> NeonVault:
    return NeonVault(database_url=DATABASE_URL)


def _store(vault: NeonVault | None = None) -> ToolAclStore:
    return ToolAclStore(neon_vault=vault or _vault())


def _response(
    status_code: int = 200, json_data: dict | list | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", HTTP_ENDPOINT),
    )


def _sql_result(
    rows: list[dict] | None = None,
    row_count: int | None = None,
    command: str = "SELECT",
) -> dict:
    result: dict = {"command": command}
    if rows is not None:
        result["rows"] = rows
        result["rowCount"] = row_count if row_count is not None else len(rows)
    return result


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------


class TestEnsureSchema:
    @pytest.mark.asyncio
    async def test_creates_table_and_index(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], command="CREATE"))
        )
        store = _store(vault)
        await store.ensure_schema()
        # 1 CREATE TABLE + 1 CREATE INDEX
        assert vault._client.post.call_count == 2


# ---------------------------------------------------------------------------
# store_acl
# ---------------------------------------------------------------------------


class TestStoreAcl:
    @pytest.mark.asyncio
    async def test_insert_returns_id(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"id": "acl-uuid-1"}], command="INSERT",
            ))
        )
        store = _store(vault)
        acl_id = await store.store_acl("npub1op", "set_pricing_model", '{"test": true}')
        assert acl_id == "acl-uuid-1"

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"id": "acl-uuid-1"}], command="INSERT",
            ))
        )
        store = _store(vault)
        # First call = insert, second = upsert (same SQL)
        await store.store_acl("npub1op", "set_pricing_model", '{"v": 1}')
        await store.store_acl("npub1op", "set_pricing_model", '{"v": 2}')
        assert vault._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_on_no_rows(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], command="INSERT"))
        )
        store = _store(vault)
        with pytest.raises(RuntimeError, match="no rows"):
            await store.store_acl("npub1op", "test", '{}')


# ---------------------------------------------------------------------------
# fetch_acl
# ---------------------------------------------------------------------------


class TestFetchAcl:
    @pytest.mark.asyncio
    async def test_exact_match(self) -> None:
        vault = _vault()
        event_json = '{"kind": 30078}'
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"signed_event_json": event_json}],
            ))
        )
        store = _store(vault)
        result = await store.fetch_acl("npub1op", "set_pricing_model")
        assert result == event_json

    @pytest.mark.asyncio
    async def test_returns_none_on_miss(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[]))
        )
        store = _store(vault)
        result = await store.fetch_acl("npub1op", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_jsonb_dict_is_serialized(self) -> None:
        """When Neon returns JSONB as a dict, it should be JSON-serialized."""
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"signed_event_json": {"kind": 30078}}],
            ))
        )
        store = _store(vault)
        result = await store.fetch_acl("npub1op", "test")
        assert result is not None
        parsed = json.loads(result)
        assert parsed["kind"] == 30078


# ---------------------------------------------------------------------------
# fetch_all_acls
# ---------------------------------------------------------------------------


class TestFetchAllAcls:
    @pytest.mark.asyncio
    async def test_returns_list(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[
                {"tool_pattern": "set_pricing_model", "signed_event_json": "{}", "updated_at": "2026-03-17"},
                {"tool_pattern": "*", "signed_event_json": "{}", "updated_at": "2026-03-16"},
            ]))
        )
        store = _store(vault)
        acls = await store.fetch_all_acls("npub1op")
        assert len(acls) == 2

    @pytest.mark.asyncio
    async def test_returns_empty(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[]))
        )
        store = _store(vault)
        acls = await store.fetch_all_acls("npub1op")
        assert acls == []


# ---------------------------------------------------------------------------
# delete_acl
# ---------------------------------------------------------------------------


class TestDeleteAcl:
    @pytest.mark.asyncio
    async def test_deletes_existing(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[], row_count=1, command="DELETE",
            ))
        )
        store = _store(vault)
        result = await store.delete_acl("npub1op", "set_pricing_model")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_not_found(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[], row_count=0, command="DELETE",
            ))
        )
        store = _store(vault)
        result = await store.delete_acl("npub1op", "nonexistent")
        assert result is False
