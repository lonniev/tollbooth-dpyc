"""TheBrainVault — VaultBackend implementation using TheBrain Cloud API.

Self-contained: uses raw httpx, no thebrain-mcp dependency. Persists
commerce ledgers via TheBrain's thought/note API in a daily-child pattern.
Also provides generic member discovery and storage for credential vaults.

Member discovery enumerates all children of the vault home thought.
Each child's ``name`` serves as the member key. Link labels (``hasMember``)
are set as best-effort metadata but are NOT used for discovery — TheBrain
Cloud's Azure App Service caching makes link metadata unreliable.

Correct API endpoints (TheBrain Cloud API):
- Base URL: https://api.bra.in
- Read note: GET /notes/{brain_id}/{thought_id} -> JSON with ``markdown`` field
- Write note: POST /notes/{brain_id}/{thought_id}/update -> JSON body ``{"markdown": "..."}``
- Create thought: POST /thoughts/{brain_id} -> tolerate HTTP 500 with valid ``{"id": "..."}`` body
- Get graph: GET /thoughts/{brain_id}/{thought_id}/graph -> JSON with ``children``, ``links`` arrays
- Update link: PATCH /links/{brain_id}/{link_id} -> JSON Patch body
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.bra.in"


class TheBrainVault:
    """Vault persistence via TheBrain Cloud API using child-based member discovery.

    Implements the tollbooth ``VaultBackend`` protocol:

    - ``store_ledger(user_id, ledger_json) -> str``
    - ``fetch_ledger(user_id) -> str | None``
    - ``snapshot_ledger(user_id, ledger_json, timestamp) -> str | None``

    Also provides generic member operations for credential storage:

    - ``store_member_note(user_id, content) -> str``
    - ``fetch_member_note(user_id) -> str | None``
    - ``soft_delete_member(user_id, reason) -> str | None``

    Member discovery: all children of the vault home thought are treated
    as members. Each child's ``name`` is the member key. Link labels
    (``hasMember``) are set as best-effort metadata but are NOT used for
    discovery — TheBrain Cloud's Azure caching makes link names unreliable.

    **Vault home hygiene**: only member thoughts should be direct children.
    Non-member thoughts (sub-vault homes, etc.) should be linked as jumps
    or siblings, not children.

    Caching strategy:

    - ``_index_cache``: Members discovered via ``_discover_members()``,
      invalidated when a new member is registered.
    - ``_daily_child_cache``: Maps ``"{user_id}/{YYYY-MM-DD}"`` to thought ID.
      On cache hit, ``store_ledger`` is a single ``_set_note`` call (1 API call
      instead of 3-4). On stale cache (set_note fails), evicts and falls through.
    """

    def __init__(
        self,
        api_key: str,
        brain_id: str,
        home_thought_id: str,
        trash_thought_id: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._brain_id = brain_id
        self._home_thought_id = home_thought_id
        self._trash_thought_id = trash_thought_id
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        self._index_cache: dict[str, str] | None = None
        self._daily_child_cache: dict[str, str] = {}

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # -- TheBrain API helpers ------------------------------------------------

    async def _get_note(self, thought_id: str) -> str | None:
        """Fetch a thought's note as markdown. Returns None on failure."""
        try:
            resp = await self._client.get(
                f"/notes/{self._brain_id}/{thought_id}"
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("markdown") or None
        except httpx.HTTPError:
            logger.warning("Failed to read note for thought %s", thought_id)
        return None

    async def _set_note(self, thought_id: str, markdown: str) -> None:
        """Create or update a thought's note."""
        resp = await self._client.post(
            f"/notes/{self._brain_id}/{thought_id}/update",
            json={"markdown": markdown},
        )
        resp.raise_for_status()

    async def _create_thought(
        self, name: str, parent_id: str
    ) -> dict[str, Any]:
        """Create a child thought. Returns ``{"id": "..."}``.

        TheBrain API may return HTTP 500 on successful creates, so we
        check for an ``id`` field in the response body before raising.
        """
        resp = await self._client.post(
            f"/thoughts/{self._brain_id}",
            json={
                "name": name,
                "kind": 1,
                "acType": 1,  # Private
                "sourceThoughtId": parent_id,
                "relation": 1,  # Child
            },
        )
        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
            return {}
        if "id" in data:
            return data
        resp.raise_for_status()
        return data

    async def _get_children(self, thought_id: str) -> list[dict[str, Any]]:
        """Get a thought's children via the graph endpoint."""
        try:
            resp = await self._client.get(
                f"/thoughts/{self._brain_id}/{thought_id}/graph"
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("children", [])
        except httpx.HTTPError:
            logger.warning("Failed to read graph for thought %s", thought_id)
        return []

    async def _get_graph(self, thought_id: str) -> dict[str, Any]:
        """GET /thoughts/{brainId}/{thoughtId}/graph -> full graph dict.

        Returns the raw graph response including ``children``, ``links``,
        ``parents``, ``jumps``, etc. Returns empty dict on failure.
        """
        try:
            resp = await self._client.get(
                f"/thoughts/{self._brain_id}/{thought_id}/graph"
            )
            if resp.status_code == 200:
                return resp.json()
        except httpx.HTTPError:
            logger.warning("Failed to read graph for thought %s", thought_id)
        return {}

    async def _get_link(self, link_id: str) -> dict[str, Any]:
        """GET /links/{brainId}/{linkId} -> full link dict.

        Returns the individual link object including ``name``, ``relation``,
        ``thoughtIdA``, ``thoughtIdB``, etc. Returns empty dict on failure.

        Note: The graph endpoint may omit link ``name`` fields due to
        TheBrain Cloud API caching inconsistencies. This method fetches the
        link individually and always returns the ``name`` field reliably.
        """
        try:
            resp = await self._client.get(
                f"/links/{self._brain_id}/{link_id}"
            )
            if resp.status_code == 200:
                return resp.json()
        except httpx.HTTPError:
            logger.warning("Failed to read link %s", link_id)
        return {}

    async def _update_link(self, link_id: str, updates: dict[str, Any]) -> None:
        """PATCH /links/{brainId}/{linkId} with JSON Patch format.

        ``updates`` is a dict of field -> value, e.g. ``{"name": "hasMember"}``.
        Converted to JSON Patch format internally.
        """
        patch = [
            {"op": "replace", "path": f"/{field}", "value": value}
            for field, value in updates.items()
        ]
        resp = await self._client.patch(
            f"/links/{self._brain_id}/{link_id}",
            json=patch,
            headers={"Content-Type": "application/json-patch+json"},
        )
        resp.raise_for_status()

    async def _update_thought(
        self, thought_id: str, updates: dict[str, Any],
    ) -> None:
        """PATCH /thoughts/{brainId}/{thoughtId} with JSON Patch format.

        ``updates`` is a dict of field -> value, e.g. ``{"name": "new name"}``.
        Converted to JSON Patch format internally.
        """
        patch = [
            {"op": "replace", "path": f"/{field}", "value": value}
            for field, value in updates.items()
        ]
        resp = await self._client.patch(
            f"/thoughts/{self._brain_id}/{thought_id}",
            json=patch,
            headers={"Content-Type": "application/json-patch+json"},
        )
        resp.raise_for_status()

    async def _delete_link(self, link_id: str) -> None:
        """DELETE /links/{brainId}/{linkId}."""
        resp = await self._client.delete(
            f"/links/{self._brain_id}/{link_id}"
        )
        resp.raise_for_status()

    async def _create_link(
        self,
        thought_id_a: str,
        thought_id_b: str,
        relation: int = 1,
        name: str | None = None,
    ) -> dict[str, Any]:
        """POST /links/{brainId} to create a relationship.

        ``relation``: 1=Child, 2=Parent, 3=Jump, 4=Sibling.
        Returns ``{"id": "..."}`` on success.
        """
        body: dict[str, Any] = {
            "thoughtIdA": thought_id_a,
            "thoughtIdB": thought_id_b,
            "relation": relation,
        }
        if name:
            body["name"] = name
        resp = await self._client.post(
            f"/links/{self._brain_id}",
            json=body,
        )
        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
            return {}
        if "id" in data:
            return data
        resp.raise_for_status()
        return data

    # -- Child-based member discovery ------------------------------------------

    async def _discover_members(self) -> dict[str, str]:
        """Discover vault members from the home thought's children.

        Returns ``{child_name: child_thought_id}`` for all children of the
        vault home thought. Callers locate their member by name.

        This approach is intentionally simple: the graph endpoint reliably
        returns children with their names and IDs. Earlier versions filtered
        by hasMember-labeled links, but TheBrain Cloud's Azure App Service
        caching makes link metadata (names, labels) unreliable — the graph
        endpoint may omit link names for hours after they're set.

        **Vault home hygiene**: the vault home should only contain member
        thoughts as children. Non-member children (sub-vault homes, etc.)
        should be linked as jumps or siblings, not children.

        Results are cached in ``_index_cache``; invalidated when a new
        member is registered.
        """
        if self._index_cache is not None:
            return self._index_cache

        graph = await self._get_graph(self._home_thought_id)
        if not graph:
            return {}

        children = graph.get("children", [])
        members = {c.get("name", ""): c["id"] for c in children if c.get("name")}

        self._index_cache = members
        return members

    async def _register_member(
        self, thought_id: str, graph: dict[str, Any] | None = None,
    ) -> None:
        """Label the child link from home -> thought_id as 'hasMember'.

        Scans the graph for the child link connecting the home thought to
        the given thought_id, then PATCHes the link name.
        """
        if graph is None:
            graph = await self._get_graph(self._home_thought_id)

        links = graph.get("links", [])
        home = self._home_thought_id
        for link in links:
            if link.get("relation") != 1:
                continue
            a = link.get("thoughtIdA", "")
            b = link.get("thoughtIdB", "")
            if (a == home and b == thought_id) or (b == home and a == thought_id):
                await self._update_link(link["id"], {"name": "hasMember"})
                self._index_cache = None  # Invalidate cache
                return

        logger.warning(
            "No child link found from %s to %s — cannot register member.",
            home, thought_id,
        )

    # -- Generic member operations -------------------------------------------

    async def store_member_note(self, user_id: str, content: str) -> str:
        """Find or create a member thought and write content to its note.

        If a member thought already exists (discovered via hasMember links),
        updates its note. Otherwise creates a new child thought under the
        home thought and labels the link as hasMember.

        Returns the member thought ID.
        """
        members = await self._discover_members()
        thought_id = members.get(user_id)

        if thought_id:
            await self._set_note(thought_id, content)
            return thought_id

        # Create new member thought
        result = await self._create_thought(user_id, self._home_thought_id)
        thought_id = result["id"]
        await self._set_note(thought_id, content)

        # Label the new child link as hasMember
        graph = await self._get_graph(self._home_thought_id)
        await self._register_member(thought_id, graph)

        return thought_id

    async def fetch_member_note(self, user_id: str) -> str | None:
        """Find a member by user_id and return its note content.

        Returns None if the member is not found.
        """
        members = await self._discover_members()
        thought_id = members.get(user_id)
        if not thought_id:
            return None
        return await self._get_note(thought_id)

    async def soft_delete_member(self, user_id: str, reason: str) -> str | None:
        """Soft-delete a vault member: unlink, rename, annotate, and move to trash.

        TheBrain's ``DELETE /thoughts`` is slow and leaves ghost entries in the
        Azure-cached graph endpoint for hours. This method avoids hard deletes:

        1. Unlink the thought from all linked peers.
        2. Rename to ``DELETED <thought_id>``.
        3. Prepend the note with ``DELETED because <reason>``.
        4. Child-link to the Trash Can thought (if ``trash_thought_id`` was set).

        Returns the thought ID if found and soft-deleted, ``None`` if the
        member was not found.
        """
        members = await self._discover_members()
        thought_id = members.get(user_id)
        if not thought_id:
            return None

        # 1. Unlink from all peers
        graph = await self._get_graph(thought_id)
        for link in graph.get("links", []):
            try:
                await self._delete_link(link["id"])
            except httpx.HTTPError:
                logger.warning(
                    "Failed to delete link %s during soft-delete of %s",
                    link.get("id"), thought_id,
                )

        # 2. Rename
        await self._update_thought(thought_id, {"name": f"DELETED {thought_id}"})

        # 3. Prepend note with deletion reason
        existing_note = await self._get_note(thought_id) or ""
        deletion_header = f"DELETED because {reason}\n\n---\n\n"
        await self._set_note(thought_id, deletion_header + existing_note)

        # 4. Move to trash
        if self._trash_thought_id:
            await self._create_link(
                self._trash_thought_id, thought_id, relation=1, name="soft-deleted",
            )

        # 5. Invalidate cache
        self._index_cache = None

        return thought_id

    # -- VaultBackend protocol -----------------------------------------------

    async def store_ledger(self, user_id: str, ledger_json: str) -> str:
        """Store ledger JSON in a daily child thought under the ledger parent.

        Creates one child per day (named ``YYYY-MM-DD`` in UTC). Subsequent
        flushes on the same day update the existing child's note. Previous
        days are preserved as immutable history.

        The ledger parent is discovered via hasMember links using the key
        ``"{user_id}/ledger"``.
        Returns the daily child thought ID.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache_key = f"{user_id}/{today}"

        # Fast path: cached daily child ID -> single set_note call
        cached_id = self._daily_child_cache.get(cache_key)
        if cached_id:
            try:
                await self._set_note(cached_id, ledger_json)
                return cached_id
            except httpx.HTTPError:
                logger.warning(
                    "Stale daily child cache for %s, falling through to full lookup.",
                    cache_key,
                )
                del self._daily_child_cache[cache_key]

        # Slow path: discover members via links + graph traversal
        members = await self._discover_members()
        ledger_key = f"{user_id}/ledger"
        ledger_parent_id = members.get(ledger_key)

        # Create ledger parent if needed
        if not ledger_parent_id:
            result = await self._create_thought(ledger_key, self._home_thought_id)
            ledger_parent_id = result["id"]
            # Label the new child link as hasMember
            graph = await self._get_graph(self._home_thought_id)
            await self._register_member(ledger_parent_id, graph)

        # Find or create today's daily child
        children = await self._get_children(ledger_parent_id)
        daily_child_id: str | None = None
        for child in children:
            if child.get("name") == today:
                daily_child_id = child.get("id")
                break

        if daily_child_id:
            await self._set_note(daily_child_id, ledger_json)
        else:
            result = await self._create_thought(today, ledger_parent_id)
            daily_child_id = result["id"]
            await self._set_note(daily_child_id, ledger_json)

        # Populate cache for subsequent flushes
        self._daily_child_cache[cache_key] = daily_child_id
        return daily_child_id

    async def fetch_ledger(self, user_id: str) -> str | None:
        """Fetch the most recent ledger JSON for a user.

        Reads the most recent daily child (sorted by ``YYYY-MM-DD`` name
        descending). Falls back to the parent thought's note for pre-migration
        ledgers that haven't been flushed since the upgrade.
        Returns None if no ledger exists.
        """
        members = await self._discover_members()
        ledger_key = f"{user_id}/ledger"
        ledger_parent_id = members.get(ledger_key)
        if not ledger_parent_id:
            return None

        children = await self._get_children(ledger_parent_id)
        if children:
            # Sort by name descending -- ISO dates sort lexicographically
            children_sorted = sorted(
                children, key=lambda t: t.get("name", ""), reverse=True
            )
            most_recent = children_sorted[0]
            note = await self._get_note(most_recent["id"])
            if note:
                return note

        # Fallback: read parent note (pre-migration state)
        return await self._get_note(ledger_parent_id)

    async def snapshot_ledger(
        self, user_id: str, ledger_json: str, timestamp: str
    ) -> str | None:
        """Create a timestamped snapshot under the ledger parent.

        Returns the snapshot thought ID, or None if no ledger thought exists.
        """
        members = await self._discover_members()
        ledger_key = f"{user_id}/ledger"
        ledger_parent_id = members.get(ledger_key)
        if not ledger_parent_id:
            return None

        result = await self._create_thought(timestamp, ledger_parent_id)
        snapshot_id = result["id"]
        await self._set_note(snapshot_id, ledger_json)
        return snapshot_id
