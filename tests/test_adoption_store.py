"""Tests for the durable operator-adoption store (deferred courtship)."""

from __future__ import annotations

from typing import Any

from tollbooth.authority import adoption_store


class FakeVault:
    """Records SQL + params; returns canned results. Mirrors NeonVault's
    ``_t`` schema-prefix and ``_execute`` HTTP-SQL helper."""

    def __init__(self, rows: list[dict[str, Any]] | None = None, row_count: int = 1):
        self.schema_prefix = "authority."
        self.calls: list[tuple[str, list[Any] | None]] = []
        self._rows = rows if rows is not None else []
        self._row_count = row_count

    def _t(self, table: str) -> str:
        return f"{self.schema_prefix}{table}"

    async def _execute(self, query: str, params: list[Any] | None = None) -> dict[str, Any]:
        self.calls.append((query, params))
        if query.strip().upper().startswith("SELECT"):
            return {"rows": self._rows}
        return {"rowCount": self._row_count, "rows": []}


async def test_ensure_schema_is_schema_qualified():
    v = FakeVault()
    await adoption_store.ensure_schema(v)
    sql, _ = v.calls[0]
    assert "CREATE TABLE IF NOT EXISTS authority.operator_adoption_requests" in sql
    assert "operator_npub TEXT PRIMARY KEY" in sql


async def test_upsert_pending_binds_params_and_upserts():
    v = FakeVault()
    await adoption_store.upsert_pending(v, "npub1op", "https://svc", note="hi", dpop_token_hash="h")
    sql, params = v.calls[0]
    assert "INSERT INTO authority.operator_adoption_requests" in sql
    assert "ON CONFLICT (operator_npub) DO UPDATE" in sql
    assert params == ["npub1op", "https://svc", "h", "hi"]


async def test_get_returns_row_or_none():
    row = {"operator_npub": "npub1op", "service_url": "u", "status": "pending"}
    assert (await adoption_store.get(FakeVault(rows=[row]), "npub1op"))["status"] == "pending"
    assert await adoption_store.get(FakeVault(rows=[]), "npub1op") is None


async def test_list_pending_filters_and_returns():
    rows = [{"operator_npub": "a"}, {"operator_npub": "b"}]
    v = FakeVault(rows=rows)
    out = await adoption_store.list_pending(v)
    assert len(out) == 2
    # the query restricts to pending + non-expired
    assert "status = 'pending'" in v.calls[0][0] and "expires_at > now()" in v.calls[0][0]


async def test_mark_uses_rowcount_camelcase():
    assert await adoption_store.mark(FakeVault(row_count=1), "npub1op", "provisioned") is True
    assert await adoption_store.mark(FakeVault(row_count=0), "npub1op", "rejected") is False
    v = FakeVault(row_count=1)
    await adoption_store.mark(v, "npub1op", "provisioned")
    sql, params = v.calls[0]
    assert "UPDATE authority.operator_adoption_requests" in sql and "decided_at = now()" in sql
    assert params == ["npub1op", "provisioned"]


async def test_prune_expired_returns_count():
    assert await adoption_store.prune_expired(FakeVault(row_count=3)) == 3
    v = FakeVault(row_count=0)
    assert await adoption_store.prune_expired(v) == 0
    assert "DELETE FROM authority.operator_adoption_requests" in v.calls[0][0]
