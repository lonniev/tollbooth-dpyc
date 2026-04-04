"""Tests for PricingModelStore — CRUD via Neon HTTP API.

Follows the same mock pattern as test_neon_vault.py: patches
``vault._client.post`` with AsyncMock.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from tollbooth.pricing_model import PricingModel, ToolPrice
from tollbooth.pricing_store import PricingModelStore
from tollbooth.vaults.neon import NeonVault


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_neon_vault.py)
# ---------------------------------------------------------------------------

DATABASE_URL = "postgresql://user:password@ep-test.us-east-2.aws.neon.tech/testdb"
HTTP_ENDPOINT = "https://ep-test.us-east-2.aws.neon.tech/sql"


def _vault() -> NeonVault:
    return NeonVault(database_url=DATABASE_URL)


def _store(vault: NeonVault | None = None) -> PricingModelStore:
    return PricingModelStore(neon_vault=vault or _vault())


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


def _sample_model() -> PricingModel:
    return PricingModel(
        operator="npub1abc",
        name="Test Model",
        tools=[ToolPrice(tool_id="test-uuid-1", tool_name="search", price_sats=5)],
        pipeline=[],
    )


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------


class TestEnsureSchema:
    @pytest.mark.asyncio
    async def test_creates_table_and_indexes(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], command="CREATE"))
        )
        store = _store(vault)
        await store.ensure_schema()
        # 1 CREATE TABLE + 2 CREATE INDEX
        assert vault._client.post.call_count == 3


# ---------------------------------------------------------------------------
# fetch_active_model
# ---------------------------------------------------------------------------


class TestFetchActiveModel:
    @pytest.mark.asyncio
    async def test_returns_model(self) -> None:
        vault = _vault()
        model_json = json.dumps({
            "name": "Active Model",
            "tools": [{"tool_name": "search", "price_sats": 5}],
            "pipeline": [],
        })
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[{
                "id": "uuid-1",
                "operator": "npub1abc",
                "name": "Active Model",
                "model_json": model_json,
                "is_active": True,
            }]))
        )
        store = _store(vault)
        model = await store.fetch_active_model("npub1abc")
        assert model is not None
        assert model.name == "Active Model"
        assert model.is_active is True

    @pytest.mark.asyncio
    async def test_returns_none_when_no_active(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[]))
        )
        store = _store(vault)
        model = await store.fetch_active_model("npub1abc")
        assert model is None


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    @pytest.mark.asyncio
    async def test_returns_list(self) -> None:
        vault = _vault()
        model_json = json.dumps({"name": "M1", "tools": [], "pipeline": []})
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[
                {"id": "u1", "operator": "npub1abc", "name": "M1",
                 "model_json": model_json, "is_active": False},
                {"id": "u2", "operator": "npub1abc", "name": "M2",
                 "model_json": model_json, "is_active": True},
            ]))
        )
        store = _store(vault)
        models = await store.list_models("npub1abc")
        assert len(models) == 2

    @pytest.mark.asyncio
    async def test_returns_empty(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[]))
        )
        store = _store(vault)
        models = await store.list_models("npub1abc")
        assert models == []


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------


class TestCreateModel:
    @pytest.mark.asyncio
    async def test_returns_uuid(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"id": "new-uuid-123"}], command="INSERT",
            ))
        )
        store = _store(vault)
        model_id = await store.create_model(_sample_model())
        assert model_id == "new-uuid-123"

    @pytest.mark.asyncio
    async def test_raises_on_no_rows(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], command="INSERT"))
        )
        store = _store(vault)
        with pytest.raises(RuntimeError, match="no rows"):
            await store.create_model(_sample_model())


# ---------------------------------------------------------------------------
# update_model
# ---------------------------------------------------------------------------


class TestUpdateModel:
    @pytest.mark.asyncio
    async def test_sends_update(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], command="UPDATE"))
        )
        store = _store(vault)
        new_json = json.dumps({"name": "Updated", "tools": [], "pipeline": []})
        await store.update_model("uuid-1", new_json)
        vault._client.post.assert_called_once()


# ---------------------------------------------------------------------------
# activate_model
# ---------------------------------------------------------------------------


class TestActivateModel:
    @pytest.mark.asyncio
    async def test_deactivates_then_activates(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], command="UPDATE"))
        )
        store = _store(vault)
        await store.activate_model("uuid-new", operator="npub1abc")
        # Two UPDATE calls: deactivate all, then activate target
        assert vault._client.post.call_count == 2


# ---------------------------------------------------------------------------
# delete_model
# ---------------------------------------------------------------------------


class TestDeleteModel:
    @pytest.mark.asyncio
    async def test_deletes_inactive(self) -> None:
        vault = _vault()
        call_count = 0

        async def mock_post(url: str, **kwargs: dict) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # SELECT check — inactive
                return _response(200, _sql_result(
                    rows=[{"is_active": False}]
                ))
            else:
                # DELETE succeeds
                return _response(200, _sql_result(
                    rows=[], row_count=1, command="DELETE",
                ))

        vault._client.post = AsyncMock(side_effect=mock_post)
        store = _store(vault)
        result = await store.delete_model("uuid-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_refuses_active(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"is_active": True}]
            ))
        )
        store = _store(vault)
        result = await store.delete_model("uuid-1")
        assert result is False
        # Only the SELECT, no DELETE
        assert vault._client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_false_not_found(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[]))
        )
        store = _store(vault)
        result = await store.delete_model("uuid-nonexistent")
        assert result is False
