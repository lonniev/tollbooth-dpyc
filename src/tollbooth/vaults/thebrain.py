"""TheBrainVault — VaultBackend implementation using TheBrain Cloud API.

Self-contained: uses raw httpx, no thebrain-mcp dependency. Persists
commerce ledgers via TheBrain's thought/note API in a daily-child pattern.

Correct API endpoints (TheBrain Cloud API):
- Base URL: https://api.bra.in
- Read note: GET /notes/{brain_id}/{thought_id} -> JSON with ``markdown`` field
- Write note: POST /notes/{brain_id}/{thought_id}/update -> JSON body ``{"markdown": "..."}``
- Create thought: POST /thoughts/{brain_id} -> tolerate HTTP 500 with valid ``{"id": "..."}`` body
- Get graph: GET /thoughts/{brain_id}/{thought_id}/graph -> JSON with ``children`` array
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.bra.in"


class TheBrainVault:
    """Commerce ledger persistence via TheBrain Cloud API.

    Implements the tollbooth ``VaultBackend`` protocol:

    - ``store_ledger(user_id, ledger_json) -> str``
    - ``fetch_ledger(user_id) -> str | None``
    - ``snapshot_ledger(user_id, ledger_json, timestamp) -> str | None``

    Index pattern: home thought note stores JSON ``{user_id: thought_id}``.

    Caching strategy:

    - ``_index_cache``: Read once per process, invalidated on ledger parent creation.
    - ``_daily_child_cache``: Maps ``"{user_id}/{YYYY-MM-DD}"`` to thought ID.
      On cache hit, ``store_ledger`` is a single ``_set_note`` call (1 API call
      instead of 3-4). On stale cache (set_note fails), evicts and falls through.
    """

    def __init__(
        self,
        api_key: str,
        brain_id: str,
        home_thought_id: str,
    ) -> None:
        self._api_key = api_key
        self._brain_id = brain_id
        self._home_thought_id = home_thought_id
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

    # -- Index management ----------------------------------------------------

    async def _read_index(self) -> dict[str, str]:
        """Read the user_id -> thought_id index from the home thought."""
        if self._index_cache is not None:
            return self._index_cache
        text = await self._get_note(self._home_thought_id)
        if text:
            try:
                self._index_cache = json.loads(text)
                return self._index_cache
            except json.JSONDecodeError:
                logger.warning("Vault index is corrupted (invalid JSON).")
        return {}

    async def _write_index(self, index: dict[str, str]) -> None:
        """Write the user_id -> thought_id index to the home thought."""
        await self._set_note(self._home_thought_id, json.dumps(index))
        self._index_cache = dict(index)

    # -- VaultBackend protocol -----------------------------------------------

    async def store_ledger(self, user_id: str, ledger_json: str) -> str:
        """Store ledger JSON in a daily child thought under the ledger parent.

        Creates one child per day (named ``YYYY-MM-DD`` in UTC). Subsequent
        flushes on the same day update the existing child's note. Previous
        days are preserved as immutable history.

        Uses index key ``"{user_id}/ledger"`` to track the ledger parent thought ID.
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

        # Slow path: full index read + graph traversal
        index = await self._read_index()
        ledger_key = f"{user_id}/ledger"
        ledger_parent_id = index.get(ledger_key)

        # Create ledger parent if needed
        if not ledger_parent_id:
            parent_id = index.get(user_id, self._home_thought_id)
            result = await self._create_thought(f"{user_id}/ledger", parent_id)
            ledger_parent_id = result["id"]
            index[ledger_key] = ledger_parent_id
            self._index_cache = None  # Invalidate -- new key added
            await self._write_index(index)

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
        index = await self._read_index()
        ledger_key = f"{user_id}/ledger"
        ledger_parent_id = index.get(ledger_key)
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
        index = await self._read_index()
        ledger_key = f"{user_id}/ledger"
        ledger_parent_id = index.get(ledger_key)
        if not ledger_parent_id:
            return None

        result = await self._create_thought(timestamp, ledger_parent_id)
        snapshot_id = result["id"]
        await self._set_note(snapshot_id, ledger_json)
        return snapshot_id
