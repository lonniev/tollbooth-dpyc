"""Tests for TheBrainVault — VaultBackend via TheBrain Cloud API."""

from unittest.mock import AsyncMock

import httpx
import pytest

from tollbooth.vaults.thebrain import TheBrainVault


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BRAIN_ID = "test-brain-id"
HOME_ID = "home-thought-id"
API_KEY = "test-api-key"


TRASH_ID = "trash-thought-id"


def _vault(trash: bool = False) -> TheBrainVault:
    return TheBrainVault(
        api_key=API_KEY,
        brain_id=BRAIN_ID,
        home_thought_id=HOME_ID,
        trash_thought_id=TRASH_ID if trash else None,
    )


def _response(status_code: int = 200, json_data: dict | list | None = None) -> httpx.Response:
    """Build a fake httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "https://api.bra.in/test"),
    )
    return resp


def _graph_with_children(
    members: dict[str, str],
    extra_links: list[dict] | None = None,
) -> dict:
    """Build a mock graph response with children.

    ``members``: {thought_name: thought_id} — creates children.
    ``extra_links``: additional links (for _register_member tests).
    """
    children = [{"id": tid, "name": name} for name, tid in members.items()]
    links = [
        {
            "id": f"link-{tid}",
            "thoughtIdA": HOME_ID,
            "thoughtIdB": tid,
            "name": None,
            "relation": 1,
        }
        for tid in members.values()
    ]
    if extra_links:
        links.extend(extra_links)
    return {"children": children, "links": links}


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
# _get_graph
# ---------------------------------------------------------------------------


class TestGetGraph:
    @pytest.mark.asyncio
    async def test_returns_full_graph(self) -> None:
        vault = _vault()
        graph = {
            "children": [{"id": "c1", "name": "child1"}],
            "links": [{"id": "link1", "relation": 1}],
            "parents": [],
        }
        vault._client.get = AsyncMock(return_value=_response(200, graph))
        result = await vault._get_graph("thought-1")
        assert result["children"] == [{"id": "c1", "name": "child1"}]
        assert result["links"] == [{"id": "link1", "relation": 1}]

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self) -> None:
        vault = _vault()
        vault._client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        result = await vault._get_graph("thought-1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_on_404(self) -> None:
        vault = _vault()
        vault._client.get = AsyncMock(return_value=_response(404, {}))
        result = await vault._get_graph("thought-1")
        assert result == {}


# ---------------------------------------------------------------------------
# _update_link
# ---------------------------------------------------------------------------


class TestUpdateLink:
    @pytest.mark.asyncio
    async def test_sends_json_patch(self) -> None:
        vault = _vault()
        vault._client.patch = AsyncMock(return_value=_response(200, {}))
        await vault._update_link("link-1", {"name": "hasMember"})
        vault._client.patch.assert_called_once_with(
            f"/links/{BRAIN_ID}/link-1",
            json=[{"op": "replace", "path": "/name", "value": "hasMember"}],
            headers={"Content-Type": "application/json-patch+json"},
        )

    @pytest.mark.asyncio
    async def test_sends_multiple_fields(self) -> None:
        vault = _vault()
        vault._client.patch = AsyncMock(return_value=_response(200, {}))
        await vault._update_link("link-1", {"name": "test", "color": "#ff0000"})
        call_args = vault._client.patch.call_args
        patch_body = call_args.kwargs["json"]
        assert len(patch_body) == 2

    @pytest.mark.asyncio
    async def test_raises_on_failure(self) -> None:
        vault = _vault()
        vault._client.patch = AsyncMock(return_value=_response(500, {}))
        with pytest.raises(httpx.HTTPStatusError):
            await vault._update_link("link-1", {"name": "hasMember"})


# ---------------------------------------------------------------------------
# _update_thought
# ---------------------------------------------------------------------------


class TestUpdateThought:
    @pytest.mark.asyncio
    async def test_sends_json_patch(self) -> None:
        vault = _vault()
        vault._client.patch = AsyncMock(return_value=_response(200, {}))
        await vault._update_thought("thought-1", {"name": "new name"})
        vault._client.patch.assert_called_once_with(
            f"/thoughts/{BRAIN_ID}/thought-1",
            json=[{"op": "replace", "path": "/name", "value": "new name"}],
            headers={"Content-Type": "application/json-patch+json"},
        )

    @pytest.mark.asyncio
    async def test_raises_on_failure(self) -> None:
        vault = _vault()
        vault._client.patch = AsyncMock(return_value=_response(500, {}))
        with pytest.raises(httpx.HTTPStatusError):
            await vault._update_thought("thought-1", {"name": "x"})


# ---------------------------------------------------------------------------
# _delete_link
# ---------------------------------------------------------------------------


class TestDeleteLink:
    @pytest.mark.asyncio
    async def test_calls_delete_endpoint(self) -> None:
        vault = _vault()
        vault._client.delete = AsyncMock(return_value=_response(200, {}))
        await vault._delete_link("link-1")
        vault._client.delete.assert_called_once_with(f"/links/{BRAIN_ID}/link-1")

    @pytest.mark.asyncio
    async def test_raises_on_failure(self) -> None:
        vault = _vault()
        vault._client.delete = AsyncMock(return_value=_response(500, {}))
        with pytest.raises(httpx.HTTPStatusError):
            await vault._delete_link("link-1")


# ---------------------------------------------------------------------------
# _create_link
# ---------------------------------------------------------------------------


class TestCreateLink:
    @pytest.mark.asyncio
    async def test_sends_correct_body(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(return_value=_response(200, {"id": "new-link"}))
        result = await vault._create_link("thought-a", "thought-b", relation=1)
        assert result == {"id": "new-link"}
        vault._client.post.assert_called_once_with(
            f"/links/{BRAIN_ID}",
            json={
                "thoughtIdA": "thought-a",
                "thoughtIdB": "thought-b",
                "relation": 1,
            },
        )

    @pytest.mark.asyncio
    async def test_includes_name_when_provided(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(return_value=_response(200, {"id": "new-link"}))
        await vault._create_link("a", "b", relation=3, name="soft-deleted")
        call_args = vault._client.post.call_args
        assert call_args.kwargs["json"]["name"] == "soft-deleted"

    @pytest.mark.asyncio
    async def test_raises_on_failure_without_id(self) -> None:
        vault = _vault()
        vault._client.post = AsyncMock(return_value=_response(500, {"error": "bad"}))
        with pytest.raises(httpx.HTTPStatusError):
            await vault._create_link("a", "b")


# ---------------------------------------------------------------------------
# _get_link
# ---------------------------------------------------------------------------


class TestGetLink:
    @pytest.mark.asyncio
    async def test_returns_link_data(self) -> None:
        vault = _vault()
        link_data = {"id": "link-1", "name": "hasMember", "relation": 1}
        vault._client.get = AsyncMock(return_value=_response(200, link_data))
        result = await vault._get_link("link-1")
        assert result == link_data
        vault._client.get.assert_called_once_with(f"/links/{BRAIN_ID}/link-1")

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self) -> None:
        vault = _vault()
        vault._client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        result = await vault._get_link("link-1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_on_404(self) -> None:
        vault = _vault()
        vault._client.get = AsyncMock(return_value=_response(404, {}))
        result = await vault._get_link("link-1")
        assert result == {}


# ---------------------------------------------------------------------------
# _discover_members
# ---------------------------------------------------------------------------


class TestDiscoverMembers:
    @pytest.mark.asyncio
    async def test_returns_all_children_by_name(self) -> None:
        vault = _vault()
        graph = _graph_with_children({"user1/ledger": "member-1", "user2/ledger": "member-2"})
        vault._get_graph = AsyncMock(return_value=graph)

        result = await vault._discover_members()
        assert result == {"user1/ledger": "member-1", "user2/ledger": "member-2"}

    @pytest.mark.asyncio
    async def test_includes_children_regardless_of_link_labels(self) -> None:
        """Children are discovered by name, not by link labels."""
        vault = _vault()
        graph = {
            "children": [
                {"id": "member-1", "name": "user1"},
                {"id": "member-2", "name": "user2"},
            ],
            "links": [
                {
                    "id": "link-1",
                    "thoughtIdA": HOME_ID,
                    "thoughtIdB": "member-1",
                    "name": "hasMember",
                    "relation": 1,
                },
                {
                    "id": "link-2",
                    "thoughtIdA": HOME_ID,
                    "thoughtIdB": "member-2",
                    "name": None,  # No label — still discovered
                    "relation": 1,
                },
            ],
        }
        vault._get_graph = AsyncMock(return_value=graph)

        result = await vault._discover_members()
        assert result == {"user1": "member-1", "user2": "member-2"}

    @pytest.mark.asyncio
    async def test_skips_children_without_names(self) -> None:
        vault = _vault()
        graph = {
            "children": [
                {"id": "member-1", "name": "user1"},
                {"id": "member-2", "name": ""},
                {"id": "member-3"},  # no name key at all
            ],
            "links": [],
        }
        vault._get_graph = AsyncMock(return_value=graph)

        result = await vault._discover_members()
        assert result == {"user1": "member-1"}

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty(self) -> None:
        vault = _vault()
        vault._get_graph = AsyncMock(return_value={})

        result = await vault._discover_members()
        assert result == {}

    @pytest.mark.asyncio
    async def test_caches_result(self) -> None:
        vault = _vault()
        graph = _graph_with_children({"user1/ledger": "member-1"})
        vault._get_graph = AsyncMock(return_value=graph)

        result1 = await vault._discover_members()
        result2 = await vault._discover_members()
        assert result1 == result2
        vault._get_graph.assert_called_once()  # Second call uses cache

    @pytest.mark.asyncio
    async def test_no_children_key_returns_empty(self) -> None:
        vault = _vault()
        vault._get_graph = AsyncMock(return_value={"links": []})

        result = await vault._discover_members()
        assert result == {}


# ---------------------------------------------------------------------------
# _register_member
# ---------------------------------------------------------------------------


class TestRegisterMember:
    @pytest.mark.asyncio
    async def test_labels_existing_child_link(self) -> None:
        vault = _vault()
        graph = {
            "children": [{"id": "member-1", "name": "user1"}],
            "links": [{
                "id": "link-1",
                "thoughtIdA": HOME_ID,
                "thoughtIdB": "member-1",
                "name": None,
                "relation": 1,
            }],
        }
        vault._update_link = AsyncMock()

        await vault._register_member("member-1", graph=graph)
        vault._update_link.assert_called_once_with("link-1", {"name": "hasMember"})

    @pytest.mark.asyncio
    async def test_invalidates_cache(self) -> None:
        vault = _vault()
        vault._index_cache = {"old": "data"}
        graph = {
            "children": [{"id": "member-1", "name": "user1"}],
            "links": [{
                "id": "link-1",
                "thoughtIdA": HOME_ID,
                "thoughtIdB": "member-1",
                "name": None,
                "relation": 1,
            }],
        }
        vault._update_link = AsyncMock()

        await vault._register_member("member-1", graph=graph)
        assert vault._index_cache is None

    @pytest.mark.asyncio
    async def test_fetches_graph_if_not_provided(self) -> None:
        vault = _vault()
        graph = {
            "children": [{"id": "member-1", "name": "user1"}],
            "links": [{
                "id": "link-1",
                "thoughtIdA": HOME_ID,
                "thoughtIdB": "member-1",
                "name": None,
                "relation": 1,
            }],
        }
        vault._get_graph = AsyncMock(return_value=graph)
        vault._update_link = AsyncMock()

        await vault._register_member("member-1")
        vault._get_graph.assert_called_once_with(HOME_ID)

    @pytest.mark.asyncio
    async def test_no_error_when_link_not_found(self) -> None:
        vault = _vault()
        graph = {"children": [], "links": []}
        vault._update_link = AsyncMock()

        # Should not raise, just warn
        await vault._register_member("nonexistent", graph=graph)
        vault._update_link.assert_not_called()


# ---------------------------------------------------------------------------
# store_member_note
# ---------------------------------------------------------------------------


class TestStoreMemberNote:
    @pytest.mark.asyncio
    async def test_existing_member_updates_note(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={"user1": "thought-1"})
        vault._set_note = AsyncMock()

        result = await vault.store_member_note("user1", "encrypted-blob")
        assert result == "thought-1"
        vault._set_note.assert_called_once_with("thought-1", "encrypted-blob")

    @pytest.mark.asyncio
    async def test_new_member_creates_and_labels(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={})
        vault._create_thought = AsyncMock(return_value={"id": "new-thought"})
        vault._set_note = AsyncMock()
        vault._get_graph = AsyncMock(return_value={"children": [], "links": []})
        vault._register_member = AsyncMock()

        result = await vault.store_member_note("user1", "encrypted-blob")
        assert result == "new-thought"
        vault._create_thought.assert_called_once_with("user1", HOME_ID)
        vault._set_note.assert_called_once_with("new-thought", "encrypted-blob")
        vault._register_member.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_member_note
# ---------------------------------------------------------------------------


class TestFetchMemberNote:
    @pytest.mark.asyncio
    async def test_found_returns_note(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={"user1": "thought-1"})
        vault._get_note = AsyncMock(return_value="encrypted-blob")

        result = await vault.fetch_member_note("user1")
        assert result == "encrypted-blob"

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={})

        result = await vault.fetch_member_note("unknown")
        assert result is None


# ---------------------------------------------------------------------------
# soft_delete_member
# ---------------------------------------------------------------------------


class TestSoftDeleteMember:
    @pytest.mark.asyncio
    async def test_not_found_returns_none(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={})
        result = await vault.soft_delete_member("unknown", "cleanup")
        assert result is None

    @pytest.mark.asyncio
    async def test_unlinks_renames_annotates(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={"user1": "thought-1"})
        vault._get_graph = AsyncMock(return_value={
            "links": [
                {"id": "link-a"},
                {"id": "link-b"},
            ],
        })
        vault._delete_link = AsyncMock()
        vault._update_thought = AsyncMock()
        vault._get_note = AsyncMock(return_value="original content")
        vault._set_note = AsyncMock()

        result = await vault.soft_delete_member("user1", "orphan cleanup")
        assert result == "thought-1"

        # All links deleted
        assert vault._delete_link.call_count == 2
        vault._delete_link.assert_any_call("link-a")
        vault._delete_link.assert_any_call("link-b")

        # Renamed
        vault._update_thought.assert_called_once_with(
            "thought-1", {"name": "DELETED thought-1"},
        )

        # Note prepended
        vault._set_note.assert_called_once()
        note_content = vault._set_note.call_args[0][1]
        assert note_content.startswith("DELETED because orphan cleanup")
        assert "original content" in note_content

    @pytest.mark.asyncio
    async def test_moves_to_trash_when_configured(self) -> None:
        vault = _vault(trash=True)
        vault._discover_members = AsyncMock(return_value={"user1": "thought-1"})
        vault._get_graph = AsyncMock(return_value={"links": []})
        vault._delete_link = AsyncMock()
        vault._update_thought = AsyncMock()
        vault._get_note = AsyncMock(return_value=None)
        vault._set_note = AsyncMock()
        vault._create_link = AsyncMock(return_value={"id": "trash-link"})

        await vault.soft_delete_member("user1", "retired")

        vault._create_link.assert_called_once_with(
            TRASH_ID, "thought-1", relation=1, name="soft-deleted",
        )

    @pytest.mark.asyncio
    async def test_no_trash_link_without_config(self) -> None:
        vault = _vault(trash=False)
        vault._discover_members = AsyncMock(return_value={"user1": "thought-1"})
        vault._get_graph = AsyncMock(return_value={"links": []})
        vault._delete_link = AsyncMock()
        vault._update_thought = AsyncMock()
        vault._get_note = AsyncMock(return_value=None)
        vault._set_note = AsyncMock()
        vault._create_link = AsyncMock()

        await vault.soft_delete_member("user1", "retired")

        vault._create_link.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidates_cache(self) -> None:
        vault = _vault()
        vault._index_cache = {"user1": "thought-1"}
        vault._discover_members = AsyncMock(return_value={"user1": "thought-1"})
        vault._get_graph = AsyncMock(return_value={"links": []})
        vault._delete_link = AsyncMock()
        vault._update_thought = AsyncMock()
        vault._get_note = AsyncMock(return_value=None)
        vault._set_note = AsyncMock()

        await vault.soft_delete_member("user1", "cleanup")
        assert vault._index_cache is None

    @pytest.mark.asyncio
    async def test_continues_on_link_delete_failure(self) -> None:
        """A failing link delete should not abort the soft-delete."""
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={"user1": "thought-1"})
        vault._get_graph = AsyncMock(return_value={
            "links": [{"id": "bad-link"}, {"id": "good-link"}],
        })
        vault._delete_link = AsyncMock(side_effect=[
            httpx.HTTPStatusError(
                "400", request=httpx.Request("DELETE", "https://api.bra.in"),
                response=_response(400),
            ),
            None,  # second link succeeds
        ])
        vault._update_thought = AsyncMock()
        vault._get_note = AsyncMock(return_value=None)
        vault._set_note = AsyncMock()

        result = await vault.soft_delete_member("user1", "cleanup")
        assert result == "thought-1"
        # Rename and note update still happened
        vault._update_thought.assert_called_once()
        vault._set_note.assert_called_once()


# ---------------------------------------------------------------------------
# store_ledger
# ---------------------------------------------------------------------------


class TestStoreLedger:
    @pytest.mark.asyncio
    async def test_creates_ledger_parent_and_daily_child(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={})
        vault._create_thought = AsyncMock(
            side_effect=[
                {"id": "ledger-parent-id"},   # ledger parent
                {"id": "daily-child-id"},      # daily child
            ]
        )
        vault._get_graph = AsyncMock(return_value={"children": [], "links": []})
        vault._register_member = AsyncMock()
        vault._get_children = AsyncMock(return_value=[])
        vault._set_note = AsyncMock()

        result = await vault.store_ledger("user1", '{"balance": 100}')
        assert result == "daily-child-id"

        # Should have created ledger parent under home thought
        vault._create_thought.assert_any_call("user1/ledger", HOME_ID)
        # Should have registered the new member
        vault._register_member.assert_called_once()
        # Should have set note on daily child
        vault._set_note.assert_called_once_with("daily-child-id", '{"balance": 100}')

    @pytest.mark.asyncio
    async def test_reuses_existing_daily_child(self) -> None:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        vault = _vault()
        vault._discover_members = AsyncMock(return_value={"user1/ledger": "ledger-parent-id"})
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
        vault._discover_members = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
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
        vault._discover_members = AsyncMock(return_value={})
        result = await vault.fetch_ledger("user1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_most_recent_daily_child(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
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
        vault._discover_members = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
        vault._get_children = AsyncMock(return_value=[])
        vault._get_note = AsyncMock(return_value='{"balance": 42}')

        result = await vault.fetch_ledger("user1")
        assert result == '{"balance": 42}'
        vault._get_note.assert_called_once_with("ledger-parent")

    @pytest.mark.asyncio
    async def test_falls_back_when_child_note_empty(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
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
        vault._discover_members = AsyncMock(return_value={})
        result = await vault.snapshot_ledger("user1", '{"balance": 100}', "2026-02-21T12:00:00Z")
        assert result is None

    @pytest.mark.asyncio
    async def test_creates_snapshot_child(self) -> None:
        vault = _vault()
        vault._discover_members = AsyncMock(return_value={"user1/ledger": "ledger-parent"})
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
