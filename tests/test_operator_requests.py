"""Tests for the durable store of operator requests: adoptions and registry updates."""

from __future__ import annotations

from typing import Any

import pytest

from tollbooth.authority import operator_requests as req


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


async def test_ensure_schema_creates_both_tables_and_migrates_adoptions():
    v = FakeVault()
    await req.ensure_schema(v)
    sqls = [sql for sql, _ in v.calls]
    assert any("CREATE TABLE IF NOT EXISTS authority.operator_adoption_requests" in s for s in sqls)
    assert any("CREATE TABLE IF NOT EXISTS authority.operator_update_requests" in s for s in sqls)
    assert all("operator_npub TEXT PRIMARY KEY" in s for s in sqls if "CREATE" in s)
    alter = next(s for s in sqls if s.startswith("ALTER"))
    assert "ADD COLUMN IF NOT EXISTS display_name" in alter
    assert "DROP COLUMN IF EXISTS dpop_token_hash" in alter


@pytest.mark.parametrize("kind,table", [
    (req.ADOPTION, "authority.operator_adoption_requests"),
    (req.UPDATE, "authority.operator_update_requests"),
])
async def test_upsert_pending_binds_params_into_its_kinds_table(kind, table):
    v = FakeVault()
    await req.upsert_pending(v, kind, "npub1op", service_url="https://svc", display_name="Svc", note="hi")
    sql, params = v.calls[0]
    assert f"INSERT INTO {table}" in sql
    assert "ON CONFLICT (operator_npub) DO UPDATE" in sql
    assert params == ["npub1op", "https://svc", "Svc", "hi"]


async def test_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        await req.get(FakeVault(), "merger", "npub1op")


async def test_get_returns_row_or_none():
    row = {"operator_npub": "npub1op", "service_url": "u", "status": "pending"}
    assert (await req.get(FakeVault(rows=[row]), req.ADOPTION, "npub1op"))["status"] == "pending"
    assert await req.get(FakeVault(rows=[]), req.UPDATE, "npub1op") is None


async def test_list_pending_names_each_rows_kind():
    v = FakeVault(rows=[{"operator_npub": "a", "requested_at": "2026-10-17"}])
    out = await req.list_pending(v)
    assert sorted(r["kind"] for r in out) == [req.ADOPTION, req.UPDATE]
    assert all("status = 'pending'" in sql and "expires_at > now()" in sql for sql, _ in v.calls)


async def test_mark_uses_rowcount_camelcase():
    assert await req.mark(FakeVault(row_count=1), req.ADOPTION, "npub1op", req.PROVISIONED) is True
    assert await req.mark(FakeVault(row_count=0), req.UPDATE, "npub1op", req.REJECTED) is False
    v = FakeVault(row_count=1)
    await req.mark(v, req.UPDATE, "npub1op", req.APPLIED)
    sql, params = v.calls[0]
    assert "UPDATE authority.operator_update_requests" in sql and "decided_at = now()" in sql
    assert params == ["npub1op", req.APPLIED]


async def test_prune_expired_sums_both_tables():
    assert await req.prune_expired(FakeVault(row_count=3)) == 6
    v = FakeVault(row_count=0)
    assert await req.prune_expired(v) == 0
    assert {sql.split()[2] for sql, _ in v.calls} == {
        "authority.operator_adoption_requests", "authority.operator_update_requests",
    }
