"""Tests for pricing tools — get/set round-trip via mock PricingModelStore."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from tollbooth.pricing_store import PricingModelStore
from tollbooth.tools.pricing import get_pricing_model_tool, set_pricing_model_tool
from tollbooth.vaults.neon import NeonVault


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_pricing_store.py)
# ---------------------------------------------------------------------------

DATABASE_URL = "postgresql://user:password@ep-test.us-east-2.aws.neon.tech/testdb"
HTTP_ENDPOINT = "https://ep-test.us-east-2.aws.neon.tech/sql"
OPERATOR = "npub1test"


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


def _sample_model_json() -> str:
    return json.dumps({
        "name": "Test Model",
        "tools": [
            {"tool_name": "search", "price_sats": 1, "category": "read"},
            {"tool_name": "create", "price_sats": 5, "category": "write"},
        ],
        "pipeline": [],
    })


def _sample_row(model_id: str = "uuid-1", is_active: bool = True) -> dict:
    return {
        "id": model_id,
        "operator": OPERATOR,
        "name": "Test Model",
        "model_json": _sample_model_json(),
        "is_active": is_active,
    }


# ---------------------------------------------------------------------------
# get_pricing_model_tool
# ---------------------------------------------------------------------------


class TestGetPricingModelTool:
    @pytest.mark.asyncio
    async def test_returns_model_when_active(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[_sample_row()]))
        )
        store = _store(vault)
        result = await get_pricing_model_tool(store, OPERATOR)

        assert result["status"] == "ok"
        assert result["model_id"] == "uuid-1"
        assert result["name"] == "Test Model"
        assert result["is_active"] is True
        assert len(result["tools"]) == 2
        assert result["tools"][0]["tool_name"] == "search"
        assert result["tools"][0]["price_sats"] == 1

    @pytest.mark.asyncio
    async def test_returns_null_fields_when_no_model(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[]))
        )
        store = _store(vault)
        result = await get_pricing_model_tool(store, OPERATOR)

        assert result["status"] == "ok"
        assert result["model_id"] is None
        assert result["tools"] is None

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            side_effect=RuntimeError("DB connection failed")
        )
        store = _store(vault)
        result = await get_pricing_model_tool(store, OPERATOR)

        assert result["status"] == "error"
        assert "DB connection failed" in result["error"]


# ---------------------------------------------------------------------------
# set_pricing_model_tool
# ---------------------------------------------------------------------------


class TestSetPricingModelTool:
    @pytest.mark.asyncio
    async def test_creates_new_model(self) -> None:
        vault = _vault()
        call_count = 0

        async def mock_post(url: str, **kwargs: dict) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                # ensure_schema: 3 CREATE calls
                return _response(200, _sql_result(rows=[], command="CREATE"))
            elif call_count == 4:
                # create_model: INSERT returns new UUID
                # (model_id is "" from from_json, so skips fetch_active_model)
                return _response(200, _sql_result(
                    rows=[{"id": "new-uuid-abc"}], command="INSERT",
                ))
            else:
                # activate_model: 2 UPDATE calls
                return _response(200, _sql_result(rows=[], command="UPDATE"))

        vault._client.post = AsyncMock(side_effect=mock_post)
        store = _store(vault)
        result = await set_pricing_model_tool(store, OPERATOR, _sample_model_json())

        assert result["status"] == "ok"
        assert result["model_id"] == "new-uuid-abc"
        assert result["tools_count"] == 2
        assert result["action"] == "created"

    @pytest.mark.asyncio
    async def test_updates_existing_model(self) -> None:
        """When model_json includes a model_id matching the active model, update in place."""
        vault = _vault()
        call_count = 0

        async def mock_post(url: str, **kwargs: dict) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                # ensure_schema
                return _response(200, _sql_result(rows=[], command="CREATE"))
            elif call_count == 4:
                # fetch_active_model: existing model with matching ID
                return _response(200, _sql_result(rows=[_sample_row("uuid-existing")]))
            else:
                # update_model + activate_model (3 calls)
                return _response(200, _sql_result(rows=[], command="UPDATE"))

        vault._client.post = AsyncMock(side_effect=mock_post)
        store = _store(vault)

        # Include model_id in the JSON so the tool finds it and takes the update path
        model_with_id = json.dumps({
            "model_id": "uuid-existing",
            "name": "Test Model",
            "tools": [
                {"tool_name": "search", "price_sats": 1, "category": "read"},
                {"tool_name": "create", "price_sats": 5, "category": "write"},
            ],
            "pipeline": [],
        })
        result = await set_pricing_model_tool(store, OPERATOR, model_with_id)

        assert result["status"] == "ok"
        assert result["model_id"] == "uuid-existing"
        assert result["action"] == "updated"

    @pytest.mark.asyncio
    async def test_returns_error_on_invalid_json(self) -> None:
        vault = _vault()
        store = _store(vault)
        result = await set_pricing_model_tool(store, OPERATOR, "not valid json")

        assert result["status"] == "error"
        assert "Invalid model_json" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_error_on_store_failure(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            side_effect=RuntimeError("DB write failed")
        )
        store = _store(vault)
        result = await set_pricing_model_tool(store, OPERATOR, _sample_model_json())

        assert result["status"] == "error"
        assert "DB write failed" in result["error"]


# ---------------------------------------------------------------------------
# Round-trip: create → get → update → get
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_create_get_roundtrip(self) -> None:
        """Verify create via set_pricing_model_tool, then get returns same data."""
        vault = _vault()
        created_model_id = "rt-uuid-001"
        call_count = 0

        async def mock_post(url: str, **kwargs: dict) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                # ensure_schema: 3 CREATE calls
                return _response(200, _sql_result(rows=[], command="CREATE"))
            elif call_count == 4:
                # set: create_model INSERT (model_id="" so skips fetch)
                return _response(200, _sql_result(
                    rows=[{"id": created_model_id}], command="INSERT",
                ))
            elif call_count <= 6:
                # set: activate_model (2 UPDATEs)
                return _response(200, _sql_result(rows=[], command="UPDATE"))
            else:
                # get: fetch_active_model (returns the created model)
                return _response(200, _sql_result(rows=[{
                    "id": created_model_id,
                    "operator": OPERATOR,
                    "name": "Test Model",
                    "model_json": _sample_model_json(),
                    "is_active": True,
                }]))

        vault._client.post = AsyncMock(side_effect=mock_post)
        store = _store(vault)

        # Step 1: Create via set
        set_result = await set_pricing_model_tool(store, OPERATOR, _sample_model_json())
        assert set_result["status"] == "ok"
        assert set_result["model_id"] == created_model_id

        # Step 2: Read back via get
        get_result = await get_pricing_model_tool(store, OPERATOR)
        assert get_result["status"] == "ok"
        assert get_result["model_id"] == created_model_id
        assert get_result["name"] == "Test Model"
        assert len(get_result["tools"]) == 2
        assert get_result["tools"][0]["tool_name"] == "search"
        assert get_result["tools"][0]["price_sats"] == 1
        assert get_result["tools"][1]["tool_name"] == "create"
        assert get_result["tools"][1]["price_sats"] == 5
