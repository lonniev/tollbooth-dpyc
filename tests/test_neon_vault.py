"""Tests for NeonVault — VaultBackend via Neon serverless Postgres HTTP API."""

from unittest.mock import AsyncMock

import httpx
import pytest

from tollbooth.vaults.neon import NeonVault, NeonQueryError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATABASE_URL = "postgresql://user:password@ep-test.us-east-2.aws.neon.tech/testdb"
HTTP_ENDPOINT = "https://ep-test.us-east-2.aws.neon.tech/sql"


def _vault(http_endpoint: str | None = None) -> NeonVault:
    return NeonVault(
        database_url=DATABASE_URL,
        http_endpoint=http_endpoint,
    )


def _response(
    status_code: int = 200, json_data: dict | list | None = None,
) -> httpx.Response:
    """Build a fake httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", HTTP_ENDPOINT),
    )


def _sql_result(
    rows: list[dict] | None = None,
    row_count: int | None = None,
    command: str = "SELECT",
    fields: list[dict] | None = None,
) -> dict:
    """Build a Neon SQL result dict (JSON object mode — rows are dicts)."""
    result: dict = {"command": command}
    if rows is not None:
        result["rows"] = rows
        result["rowCount"] = row_count if row_count is not None else len(rows)
    if fields is not None:
        result["fields"] = fields
    return result


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_derives_endpoint_from_database_url(self) -> None:
        vault = _vault()
        assert vault._endpoint == HTTP_ENDPOINT

    def test_uses_explicit_http_endpoint(self) -> None:
        vault = _vault(http_endpoint="https://custom.endpoint.com/sql/")
        assert vault._endpoint == "https://custom.endpoint.com/sql"

    def test_sets_connection_string_header(self) -> None:
        vault = _vault()
        assert vault._client.headers["neon-connection-string"] == DATABASE_URL


# ---------------------------------------------------------------------------
# _execute
# ---------------------------------------------------------------------------


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_result_dict(self) -> None:
        vault = _vault()
        result = _sql_result(rows=[{"num": 1}], command="SELECT")
        vault._client.post = AsyncMock(return_value=_response(200, result))

        data = await vault._execute("SELECT 1 AS num")
        assert data["rows"] == [{"num": 1}]
        vault._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_query_and_params(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[{"text": "hello"}]))
        )
        await vault._execute("SELECT $1::text", ["hello"])
        call_args = vault._client.post.call_args
        body = call_args.kwargs.get("json", call_args[1] if len(call_args.args) > 1 else None)
        assert body["query"] == "SELECT $1::text"
        assert body["params"] == ["hello"]

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(return_value=_response(500, {}))
        with pytest.raises(httpx.HTTPStatusError):
            await vault._execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_raises_on_neon_query_error(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, {"message": "ERROR: relation does not exist"})
        )
        with pytest.raises(NeonQueryError, match="relation does not exist"):
            await vault._execute("SELECT * FROM nonexistent")

    @pytest.mark.asyncio
    async def test_query_error_carries_sqlstate_from_400_body(self) -> None:
        """Neon 400 bodies include a SQLSTATE ``code`` — it must survive."""
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(
                400,
                {"message": "permission denied for table operator_pricing_models",
                 "code": "42501"},
            )
        )
        with pytest.raises(NeonQueryError, match="permission denied") as exc_info:
            await vault._execute("SELECT 1")
        assert exc_info.value.code == "42501"

    @pytest.mark.asyncio
    async def test_query_error_carries_sqlstate_from_200_body(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(
                200, {"message": "relation does not exist", "code": "42P01"}
            )
        )
        with pytest.raises(NeonQueryError) as exc_info:
            await vault._execute("SELECT 1")
        assert exc_info.value.code == "42P01"

    @pytest.mark.asyncio
    async def test_query_error_code_defaults_empty(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, {"message": "something odd"})
        )
        with pytest.raises(NeonQueryError) as exc_info:
            await vault._execute("SELECT 1")
        assert exc_info.value.code == ""


# ---------------------------------------------------------------------------
# store_ledger
# ---------------------------------------------------------------------------


class TestStoreLedger:
    @pytest.mark.asyncio
    async def test_inserts_new_ledger(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"version": 1}], command="INSERT",
            ))
        )
        result = await vault.store_ledger("npub1abc", '{"v": 4, "tranches": []}')
        assert result == "1"
        assert vault._version_cache["npub1abc"] == 1

    @pytest.mark.asyncio
    async def test_optimistic_update_with_cached_version(self) -> None:
        vault = _vault()
        vault._version_cache["npub1abc"] = 3
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"version": 4}], command="UPDATE",
            ))
        )
        result = await vault.store_ledger("npub1abc", '{"v": 4}')
        assert result == "4"
        assert vault._version_cache["npub1abc"] == 4

        # Verify the UPDATE query included version guard
        call_args = vault._client.post.call_args
        body = call_args.kwargs.get("json", call_args[1] if len(call_args.args) > 1 else None)
        assert "version = $3" in body["query"]
        assert body["params"][2] == 3

    @pytest.mark.asyncio
    async def test_version_conflict_falls_through_to_upsert(self) -> None:
        vault = _vault()
        vault._version_cache["npub1abc"] = 5

        call_count = 0

        async def mock_post(url: str, **kwargs: dict) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Version conflict: UPDATE returns 0 rows
                return _response(200, _sql_result(rows=[], command="UPDATE"))
            else:
                # Upsert succeeds
                return _response(200, _sql_result(
                    rows=[{"version": 6}], command="INSERT",
                ))

        vault._client.post = AsyncMock(side_effect=mock_post)
        result = await vault.store_ledger("npub1abc", '{"v": 4}')
        assert result == "6"
        assert vault._version_cache["npub1abc"] == 6
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_when_upsert_returns_no_rows(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], command="INSERT"))
        )
        with pytest.raises(NeonQueryError, match="UPSERT returned no rows"):
            await vault.store_ledger("npub1abc", '{"v": 4}')


# ---------------------------------------------------------------------------
# fetch_ledger
# ---------------------------------------------------------------------------


class TestFetchLedger:
    @pytest.mark.asyncio
    async def test_returns_ledger_json(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"ledger_json": '{"v": 4}', "version": 3}],
            ))
        )
        result = await vault.fetch_ledger("npub1abc")
        assert result == '{"v": 4}'
        assert vault._version_cache["npub1abc"] == 3

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[]))
        )
        result = await vault.fetch_ledger("npub1unknown")
        assert result is None

    @pytest.mark.asyncio
    async def test_caches_version(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"ledger_json": '{"v": 4}', "version": 7}],
            ))
        )
        await vault.fetch_ledger("npub1abc")
        assert vault._version_cache["npub1abc"] == 7


# ---------------------------------------------------------------------------
# snapshot_ledger
# ---------------------------------------------------------------------------


class TestSnapshotLedger:
    @pytest.mark.asyncio
    async def test_stores_and_records_snapshot(self) -> None:
        vault = _vault()

        call_count = 0

        async def mock_post(url: str, **kwargs: dict) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # store_ledger UPSERT
                return _response(200, _sql_result(
                    rows=[{"version": 2}], command="INSERT",
                ))
            else:
                # snapshot journal INSERT
                return _response(200, _sql_result(
                    rows=[{"id": 42}], command="INSERT",
                ))

        vault._client.post = AsyncMock(side_effect=mock_post)

        ledger = '{"v": 4, "tranches": [{"remaining_sats": 100}]}'
        result = await vault.snapshot_ledger("npub1abc", ledger, "2026-02-23T12:00:00Z")
        assert result == "42"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_returns_none_on_journal_failure(self) -> None:
        vault = _vault()

        call_count = 0

        async def mock_post(url: str, **kwargs: dict) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _response(200, _sql_result(
                    rows=[{"version": 1}], command="INSERT",
                ))
            else:
                return _response(200, {"message": "ERROR: relation does not exist"})

        vault._client.post = AsyncMock(side_effect=mock_post)

        result = await vault.snapshot_ledger("npub1abc", '{"v": 4}', "2026-02-23T12:00:00Z")
        assert result is None  # Journal failed but store succeeded

    @pytest.mark.asyncio
    async def test_extracts_balance_for_journal(self) -> None:
        vault = _vault()

        captured_params: list = []

        async def mock_post(url: str, **kwargs: dict) -> httpx.Response:
            body = kwargs.get("json", {})
            query = body.get("query", "") if isinstance(body, dict) else ""
            if "snapshot" in query:
                captured_params.extend(body.get("params", []))
                return _response(200, _sql_result(
                    rows=[{"id": 99}], command="INSERT",
                ))
            return _response(200, _sql_result(
                rows=[{"version": 1}], command="INSERT",
            ))

        vault._client.post = AsyncMock(side_effect=mock_post)

        ledger = '{"v": 4, "tranches": [{"remaining_sats": 100}, {"remaining_sats": 50}]}'
        await vault.snapshot_ledger("npub1abc", ledger, "2026-02-23T12:00:00Z")

        # The balance_after param should be 150
        assert 150 in captured_params


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------


class TestEnsureSchema:
    @pytest.mark.asyncio
    async def test_creates_tables_and_indexes(self) -> None:
        vault = _vault()

        def mock_post(url, json=None, **kw):
            query = json.get("query", "") if json else ""
            if "SHOW search_path" in query:
                return _response(200, _sql_result(
                    rows=[{"search_path": '"$user", public'}], command="SHOW",
                ))
            return _response(200, _sql_result(rows=[], command="CREATE"))

        vault._client.post = AsyncMock(side_effect=mock_post)
        await vault.ensure_schema()
        # SHOW search_path + CREATE TABLE/INDEX statements
        assert vault._client.post.call_count >= 9  # at minimum SHOW + tables + indexes

    @pytest.mark.asyncio
    async def test_schema_statements_use_if_not_exists(self) -> None:
        vault = _vault()

        def mock_post(url, json=None, **kw):
            query = json.get("query", "") if json else ""
            if "SHOW search_path" in query:
                return _response(200, _sql_result(
                    rows=[{"search_path": '"$user", public'}], command="SHOW",
                ))
            return _response(200, _sql_result(rows=[], command="CREATE"))

        vault._client.post = AsyncMock(side_effect=mock_post)
        await vault.ensure_schema()

        for call in vault._client.post.call_args_list:
            body = call.kwargs.get("json", call[1] if len(call.args) > 1 else None)
            query = body["query"]
            if "SHOW" not in query:
                assert "IF NOT EXISTS" in query


# ---------------------------------------------------------------------------
# _extract_balance
# ---------------------------------------------------------------------------


class TestExtractBalance:
    def test_sums_remaining_sats(self) -> None:
        ledger = '{"tranches": [{"remaining_sats": 100}, {"remaining_sats": 50}]}'
        assert NeonVault._extract_balance(ledger) == 150

    def test_returns_zero_on_empty_tranches(self) -> None:
        assert NeonVault._extract_balance('{"tranches": []}') == 0

    def test_returns_zero_on_invalid_json(self) -> None:
        assert NeonVault._extract_balance("not json") == 0

    def test_returns_zero_on_missing_tranches(self) -> None:
        assert NeonVault._extract_balance('{"v": 4}') == 0


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_closes_http_client(self) -> None:
        vault = _vault()
        vault._client.aclose = AsyncMock()
        await vault.close()
        vault._client.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# VaultBackend protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_implements_vault_backend(self) -> None:
        from tollbooth.vault_backend import VaultBackend

        vault = _vault()
        assert isinstance(vault, VaultBackend)
