"""Tests for TheBrainVault — VaultBackend via TheBrain Cloud API."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tollbooth.vaults.thebrain import TheBrainVault


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BRAIN_ID = "test-brain-id"
HOME_ID = "home-thought-id"
API_KEY = "test-api-key"


def _vault() -> TheBrainVault:
    return TheBrainVault(api_key=API_KEY, brain_id=BRAIN_ID, home_thought_id=HOME_ID)


def _response(status_code: int = 200, json_data: dict | list | None = None) -> httpx.Response:
    """Build a fake httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "https://api.bra.in/test"),
    )
    return resp


# ---------------------------------------------------------------------------
# _get_note
# ---------------------------------------------------------------------------


class TestGetNote:
    @pytest.mark.asyncio
    async def test_returns_markdown(self) -> None:
        vault = _vault()
        vault._client.get = AsyncMock(return_value=_response(200, {"markdown": "hello"}))
        result = await vault._get_note("thought-1")
        assert result == "hello"
        vault._client.get.assert_called_once_with(f"/notes/{BRAIN_ID}/thought-1")

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_markdown(self) -> None:
        vault = _vault()
        vault._client.get = AsyncMock(return_value=_response(200, {"markdown": ""}))
        result = await vault._get_note("thought-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_404(self) -> None:
        vault = _vault()
        vault._client.get = AsyncMock(return_value=_response(404, {}))
        result = await vault._get_note("thought-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self) -> None:
        vault = _vault()
        vault._client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        result = await vault._get_note("thought-1")
        assert result is None


# ---------------------------------------------------------------------------
# _set_note
# ---------------------------------------------------------------------------


class TestSetNote:
    @pytest.mark.asyncio
    async def test_posts_to_update_endpoint(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(return_value=_response(200, {}))
        await vault._set_note("thought-1", "new content")
        vault._client.post.assert_called_once_with(
            f"/notes/{BRAIN_ID}/thought-1/update",
            json={"markdown": "new content"},
        )

    @pytest.mark.asyncio
    async def test_raises_on_failure(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(return_value=_response(500, {}))
        with pytest.raises(httpx.HTTPStatusError):
            await vault._set_note("thought-1", "content")


# ---------------------------------------------------------------------------
# _create_thought — 500-tolerant behavior
# ---------------------------------------------------------------------------


class TestCreateThought:
    @pytest.mark.asyncio
    async def test_returns_id_on_200(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, {"id": "new-id"})
        )
        result = await vault._create_thought("test", "parent-1")
        assert result == {"id": "new-id"}

    @pytest.mark.asyncio
    async def test_returns_id_on_500_with_body(self) -> None:
        """TheBrain sometimes returns HTTP 500 but with a valid ID body."""
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(500, {"id": "created-id"})
        )
        result = await vault._create_thought("test", "parent-1")
        assert result == {"id": "created-id"}

    @pytest.mark.asyncio
    async def test_raises_on_500_without_id(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(500, {"error": "internal"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            await vault._create_thought("test", "parent-1")

    @pytest.mark.asyncio
    async def test_sends_correct_payload(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(
            return_value=_response(200, {"id": "new-id"})
        )
        await vault._create_thought("my thought", "parent-1")
        vault._client.post.assert_called_once_with(
            f"/thoughts/{BRAIN_ID}",
            json={
                "name": "my thought",
                "kind": 1,
                "acType": 1,
                "sourceThoughtId": "parent-1",
                "relation": 1,
            },
        )


# ---------------------------------------------------------------------------
# _get_children
# ---------------------------------------------------------------------------


class TestGetChildren:
    @pytest.mark.asyncio
    async def test_returns_children_list(self) -> None:
        vault = _vault()
        children = [{"id": "c1", "name": "2026-02-20"}, {"id": "c2", "name": "2026-02-21"}]
        vault._client.get = AsyncMock(
            return_value=_response(200, {"children": children})
        )
        result = await vault._get_children("parent-1")
        assert len(result) == 2
        assert result[0]["name"] == "2026-02-20"

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self) -> None:
        vault = _vault()
        vault._client.get = AsyncMock(side_effect=httpx.ConnectError("down"))
        result = await vault._get_children("parent-1")
        assert result == []


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


class TestIndex:
    @pytest.mark.asyncio
    async def test_read_index_caches_result(self) -> None:
        vault = _vault()
        index = {"user1/ledger": "ledger-id"}
        vault._get_note = AsyncMock(return_value=json.dumps(index))
        result1 = await vault._read_index()
        result2 = await vault._read_index()
        assert result1 == index
        # Second call should use cache, not call _get_note again
        vault._get_note.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_index_returns_empty_on_no_note(self) -> None:
        vault = _vault()
        vault._get_note = AsyncMock(return_value=None)
        result = await vault._read_index()
        assert result == {}

    @pytest.mark.asyncio
    async def test_read_index_returns_empty_on_bad_json(self) -> None:
        vault = _vault()
        vault._get_note = AsyncMock(return_value="not valid json")
        result = await vault._read_index()
        assert result == {}

    @pytest.mark.asyncio
    async def test_write_index_updates_cache(self) -> None:
        vault = _vault()
        vault._set_note = AsyncMock()
        index = {"user1": "thought-1"}
        await vault._write_index(index)
        assert vault._index_cache == index
        vault._set_note.assert_called_once_with(HOME_ID, json.dumps(index))


# ---------------------------------------------------------------------------
# store_ledger
# ---------------------------------------------------------------------------


class TestStoreLedger:
    @pytest.mark.asyncio
    async def test_creates_ledger_parent_and_daily_child(self) -> None:
        vault = _vault()
        vault._read_index = AsyncMock(return_value={})
        vault._write_index = AsyncMock()
        vault._create_thought = AsyncMock(
            side_effect=[
                {"id": "ledger-parent-id"},   # ledger parent
                {"id": "daily-child-id"},      # daily child
            ]
        )
        vault._get_children = AsyncMock(return_value=[])
        vault._set_note = AsyncMock()

        result = await vault.store_ledger("user1", '{"balance": 100}')
        assert result == "daily-child-id"

        # Should have created ledger parent under home thought
        vault._create_thought.assert_any_call("user1/ledger", HOME_ID)
        # Should have written index
        vault._write_index.assert_called_once()
        # Should have set note on daily child
        vault._set_note.assert_called_once_with("daily-child-id", '{"balance": 100}')

    @pytest.mark.asyncio
    async def test_reuses_existing_daily_child(self) -> None:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        vault = _vault()
        vault._read_index = AsyncMock(return_value={"user1/ledger": "ledger-parent-id"})
        vault._get_children = AsyncMock(
            return_value=[{"id": "existing-child", "name": today}]
        )
        vault._set_note = AsyncMock()

        result = await vault.store_ledger("user1", '{"balance": 200}')
        assert result == "existing-child"
        vault._set_note.assert_called_once_with("existing-child", '{"balance": 200}')

    @pytest.mark.asyncio
    async def test_fast_path_uses_daily_child_cache(self) -> None:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        vault = _vault()
        vault._daily_child_cache[f"user1/{today}"] = "cached-child-id"
        vault._set_note = AsyncMock()

        result = await vault.store_ledger("user1", '{"balance": 300}')
        assert result == "cached-child-id"
        vault._set_note.assert_called_once_with("cached-child-id", '{"balance": 300}')

    @pytest.mark.asyncio
    async def test_stale_cache_falls_through(self) -> None:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        vault = _vault()
        vault._daily_child_cache[f"user1/{today}"] = "stale-id"

        # First _set_note call (cached path) fails, second (fresh path) succeeds
        call_count = 0
        async def _set_note_side_effect(thought_id: str, markdown: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError(
                    "gone", request=httpx.Request("POST", "https://api.bra.in"), response=_response(404)
                )

        vault._set_note = AsyncMock(side_effect=_set_note_side_effect)
        vault._read_index = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
        vault._get_children = AsyncMock(
            return_value=[{"id": "fresh-child", "name": today}]
        )

        result = await vault.store_ledger("user1", '{"balance": 400}')
        assert result == "fresh-child"
        # Cache should be updated
        assert vault._daily_child_cache[f"user1/{today}"] == "fresh-child"


# ---------------------------------------------------------------------------
# fetch_ledger
# ---------------------------------------------------------------------------


class TestFetchLedger:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_ledger(self) -> None:
        vault = _vault()
        vault._read_index = AsyncMock(return_value={})
        result = await vault.fetch_ledger("user1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_most_recent_daily_child(self) -> None:
        vault = _vault()
        vault._read_index = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
        vault._get_children = AsyncMock(return_value=[
            {"id": "c1", "name": "2026-02-19"},
            {"id": "c2", "name": "2026-02-21"},
            {"id": "c3", "name": "2026-02-20"},
        ])
        vault._get_note = AsyncMock(return_value='{"balance": 999}')

        result = await vault.fetch_ledger("user1")
        assert result == '{"balance": 999}'
        # Should read note from most recent (2026-02-21)
        vault._get_note.assert_called_once_with("c2")

    @pytest.mark.asyncio
    async def test_falls_back_to_parent_note(self) -> None:
        vault = _vault()
        vault._read_index = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
        vault._get_children = AsyncMock(return_value=[])
        vault._get_note = AsyncMock(return_value='{"balance": 42}')

        result = await vault.fetch_ledger("user1")
        assert result == '{"balance": 42}'
        vault._get_note.assert_called_once_with("ledger-parent")

    @pytest.mark.asyncio
    async def test_falls_back_when_child_note_empty(self) -> None:
        vault = _vault()
        vault._read_index = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
        vault._get_children = AsyncMock(
            return_value=[{"id": "c1", "name": "2026-02-21"}]
        )

        # First call (child note) returns None, second (parent) returns data
        call_count = 0
        async def _get_note_side_effect(thought_id: str) -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None
            return '{"balance": 42}'

        vault._get_note = AsyncMock(side_effect=_get_note_side_effect)

        result = await vault.fetch_ledger("user1")
        assert result == '{"balance": 42}'


# ---------------------------------------------------------------------------
# snapshot_ledger
# ---------------------------------------------------------------------------


class TestSnapshotLedger:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_ledger(self) -> None:
        vault = _vault()
        vault._read_index = AsyncMock(return_value={})
        result = await vault.snapshot_ledger("user1", '{"balance": 100}', "2026-02-21T12:00:00Z")
        assert result is None

    @pytest.mark.asyncio
    async def test_creates_snapshot_child(self) -> None:
        vault = _vault()
        vault._read_index = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
        vault._create_thought = AsyncMock(return_value={"id": "snapshot-id"})
        vault._set_note = AsyncMock()

        result = await vault.snapshot_ledger("user1", '{"balance": 100}', "2026-02-21T12:00:00Z")
        assert result == "snapshot-id"
        vault._create_thought.assert_called_once_with("2026-02-21T12:00:00Z", "ledger-parent")
        vault._set_note.assert_called_once_with("snapshot-id", '{"balance": 100}')


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
