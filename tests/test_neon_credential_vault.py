"""Tests for NeonCredentialVault — CredentialVaultBackend via Neon Postgres."""

from unittest.mock import AsyncMock

import httpx
import pytest

from tollbooth.credential_vault_backend import CredentialVaultBackend
from tollbooth.vaults.neon import NeonCredentialVault, NeonVault

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATABASE_URL = "postgresql://user:password@ep-test.us-east-2.aws.neon.tech/testdb"
HTTP_ENDPOINT = "https://ep-test.us-east-2.aws.neon.tech/sql"


def _neon_vault() -> NeonVault:
    return NeonVault(database_url=DATABASE_URL)


def _cred_vault(neon: NeonVault | None = None) -> NeonCredentialVault:
    return NeonCredentialVault(neon_vault=neon or _neon_vault())


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
    async def test_creates_credentials_table(self) -> None:
        neon = _neon_vault()
        neon._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], command="CREATE"))
        )
        vault = _cred_vault(neon)
        await vault.ensure_schema()
        # ensure_schema creates both credentials and session_bindings tables
        assert neon._client.post.call_count == 2
        first_body = neon._client.post.call_args_list[0].kwargs.get("json")
        assert "credentials" in first_body["query"]
        assert "IF NOT EXISTS" in first_body["query"]
        second_body = neon._client.post.call_args_list[1].kwargs.get("json")
        assert "session_bindings" in second_body["query"]

    @pytest.mark.asyncio
    async def test_schema_idempotent(self) -> None:
        """Calling ensure_schema twice doesn't raise."""
        neon = _neon_vault()
        neon._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], command="CREATE"))
        )
        vault = _cred_vault(neon)
        await vault.ensure_schema()
        await vault.ensure_schema()
        # 2 tables per call × 2 calls = 4
        assert neon._client.post.call_count == 4


# ---------------------------------------------------------------------------
# store_credentials
# ---------------------------------------------------------------------------


class TestStoreCredentials:
    @pytest.mark.asyncio
    async def test_stores_blob(self) -> None:
        neon = _neon_vault()
        neon._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], row_count=1, command="INSERT"))
        )
        vault = _cred_vault(neon)
        await vault.store_credentials("thebrain", "npub1abc", "encrypted-data")

        body = neon._client.post.call_args.kwargs.get(
            "json", neon._client.post.call_args[1] if len(neon._client.post.call_args.args) > 1 else None
        )
        assert body["params"] == ["thebrain", "npub1abc", "encrypted-data"]
        assert "ON CONFLICT" in body["query"]


# ---------------------------------------------------------------------------
# fetch_credentials
# ---------------------------------------------------------------------------


class TestFetchCredentials:
    @pytest.mark.asyncio
    async def test_returns_blob(self) -> None:
        neon = _neon_vault()
        neon._client.post = AsyncMock(
            return_value=_response(200, _sql_result(
                rows=[{"encrypted_blob": "my-secret-blob"}],
            ))
        )
        vault = _cred_vault(neon)
        result = await vault.fetch_credentials("thebrain", "npub1abc")
        assert result == "my-secret-blob"

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(self) -> None:
        neon = _neon_vault()
        neon._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[]))
        )
        vault = _cred_vault(neon)
        result = await vault.fetch_credentials("thebrain", "npub1unknown")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite_returns_latest(self) -> None:
        """Store twice, fetch returns the latest blob."""
        neon = _neon_vault()
        neon._client.post = AsyncMock(
            side_effect=[
                _response(200, _sql_result(rows=[], row_count=1, command="INSERT")),
                _response(200, _sql_result(rows=[], row_count=1, command="INSERT")),
                _response(200, _sql_result(rows=[{"encrypted_blob": "blob-v2"}])),
            ]
        )
        vault = _cred_vault(neon)
        await vault.store_credentials("thebrain", "npub1abc", "blob-v1")
        await vault.store_credentials("thebrain", "npub1abc", "blob-v2")
        result = await vault.fetch_credentials("thebrain", "npub1abc")
        assert result == "blob-v2"


# ---------------------------------------------------------------------------
# delete_credentials
# ---------------------------------------------------------------------------


class TestDeleteCredentials:
    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self) -> None:
        neon = _neon_vault()
        neon._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], row_count=1, command="DELETE"))
        )
        vault = _cred_vault(neon)
        result = await vault.delete_credentials("thebrain", "npub1abc")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_missing_returns_false(self) -> None:
        neon = _neon_vault()
        neon._client.post = AsyncMock(
            return_value=_response(200, _sql_result(rows=[], row_count=0, command="DELETE"))
        )
        vault = _cred_vault(neon)
        result = await vault.delete_credentials("thebrain", "npub1unknown")
        assert result is False


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_implements_credential_vault_backend(self) -> None:
        vault = _cred_vault()
        assert isinstance(vault, CredentialVaultBackend)
