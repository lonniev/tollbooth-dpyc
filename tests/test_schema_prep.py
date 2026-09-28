"""Schema preparation: one breadcrumb, one read.

The usual cold start must cost exactly one request to Neon. These tests drive
a real ``NeonVault`` against a fake Neon HTTP endpoint that keeps honest state
— which tables exist, what the crumb says — and count requests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from tollbooth.vaults.neon import NeonVault
from tollbooth.vaults.schema import (
    SCHEMA_DIGEST,
    SCHEMA_VERSION,
    SchemaPrepError,
    concerns,
    digest,
    prepare_schema,
)

PREFIXED = "postgresql://u:p@ep-x-pooler.us-east-2.aws.neon.tech/db?options=-c%20search_path%3Dop_test,public"
PLAIN = "postgresql://u:p@ep-x-pooler.us-east-2.aws.neon.tech/db"


class FakeNeon:
    """Just enough of Neon's /sql endpoint: single queries and batches."""

    def __init__(self, *, crumb: int | None = None, schema_exists: bool = True, fail_on: str = "") -> None:
        self.crumb = crumb
        self.schema_exists = schema_exists
        self.fail_on = fail_on
        self.requests: list[dict] = []

    @staticmethod
    def _resp(status: int, body: dict) -> httpx.Response:
        return httpx.Response(status, json=body, request=httpx.Request("POST", "https://x/sql"))

    async def post(self, url: str, json: dict | None = None, **_: object) -> httpx.Response:
        body = json or {}
        self.requests.append(body)
        if "queries" in body:
            for q in body["queries"]:
                if self.fail_on and self.fail_on in q["query"]:
                    return self._resp(400, {"message": "permission denied for table", "code": "42501"})
                if "INSERT INTO" in q["query"] and "tollbooth_schema" in q["query"]:
                    self.crumb = int(q["query"].split("VALUES (true, ")[1].split(",")[0])
            return self._resp(200, {"results": [{"rows": []} for _ in body["queries"]]})
        query = body["query"]
        if query.startswith("SELECT version FROM"):
            if self.crumb is None:
                return self._resp(400, {"message": 'relation "op_test.tollbooth_schema" does not exist', "code": "42P01"})
            return self._resp(200, {"rows": [{"version": self.crumb}], "rowCount": 1})
        if "pg_namespace" in query:
            return self._resp(200, {"rows": [{"?column?": 1}] if self.schema_exists else []})
        if query.startswith("CREATE SCHEMA"):
            self.schema_exists = True
        return self._resp(200, {"rows": []})


def _vault(url: str, neon: FakeNeon) -> NeonVault:
    v = NeonVault(database_url=url, encryption_nsec_hex="a" * 64)
    v._client.post = AsyncMock(side_effect=neon.post)  # type: ignore[method-assign]
    return v


@pytest.mark.asyncio
async def test_a_prepared_database_costs_one_request():
    neon = FakeNeon(crumb=SCHEMA_VERSION)
    steps = await _vault(PREFIXED, neon).ensure_schema()
    assert steps == []
    assert len(neon.requests) == 1, "the breadcrumb read, and nothing else"


@pytest.mark.asyncio
async def test_a_newer_crumb_is_also_prepared():
    neon = FakeNeon(crumb=SCHEMA_VERSION + 3)
    await _vault(PREFIXED, neon).ensure_schema()
    assert len(neon.requests) == 1


@pytest.mark.asyncio
async def test_a_fresh_database_is_prepared_once_per_concern_then_crumbed():
    neon = FakeNeon(crumb=None)
    vault = _vault(PREFIXED, neon)
    steps = await vault.ensure_schema()
    names = [s["step"] for s in steps]
    assert names == [n for n, _ in concerns()] + ["crumb"]
    assert all(s["ok"] for s in steps)
    batches = [r for r in neon.requests if "queries" in r]
    assert len(batches) == len(concerns()) + 1, "one compound request per concern, plus the crumb"
    assert neon.crumb == SCHEMA_VERSION
    # Its peers now pay one read.
    neon.requests.clear()
    await _vault(PREFIXED, neon).ensure_schema()
    assert len(neon.requests) == 1


@pytest.mark.asyncio
async def test_an_older_crumb_prepares_again():
    neon = FakeNeon(crumb=SCHEMA_VERSION - 1)
    steps = await _vault(PREFIXED, neon).ensure_schema()
    assert steps and neon.crumb == SCHEMA_VERSION


@pytest.mark.asyncio
async def test_a_missing_operator_schema_is_created_first():
    neon = FakeNeon(crumb=None, schema_exists=False)
    steps = await _vault(PREFIXED, neon).ensure_schema()
    assert steps[0]["step"] == "schema"
    assert any(r.get("query", "").startswith("CREATE SCHEMA IF NOT EXISTS op_test") for r in neon.requests)


@pytest.mark.asyncio
async def test_an_unprefixed_database_skips_the_schema_probe():
    neon = FakeNeon(crumb=None)
    await _vault(PLAIN, neon).ensure_schema()
    assert not any("pg_namespace" in r.get("query", "") for r in neon.requests)


@pytest.mark.asyncio
async def test_a_failed_read_is_never_mistaken_for_an_unprepared_database():
    neon = FakeNeon(crumb=SCHEMA_VERSION)

    async def broken(url, json=None, **_):
        return FakeNeon._resp(400, {"message": "password authentication failed", "code": "28P01"})

    vault = _vault(PREFIXED, neon)
    vault._client.post = AsyncMock(side_effect=broken)  # type: ignore[method-assign]
    with pytest.raises(Exception, match="password authentication failed"):
        await vault.ensure_schema()
    assert vault._client.post.await_count == 1, "no DDL attempted on an unreadable database"


@pytest.mark.asyncio
async def test_a_failed_step_writes_no_crumb_and_reports_where():
    neon = FakeNeon(crumb=None, fail_on="coupons")
    with pytest.raises(SchemaPrepError) as caught:
        await _vault(PREFIXED, neon).ensure_schema()
    steps = caught.value.steps
    assert steps[-1]["step"] == "coupons" and steps[-1]["ok"] is False
    assert "permission denied" in steps[-1]["error"]
    assert neon.crumb is None, "no crumb unless everything before it landed"


@pytest.mark.asyncio
async def test_force_ignores_the_crumb():
    neon = FakeNeon(crumb=SCHEMA_VERSION)
    steps = await prepare_schema(_vault(PREFIXED, neon), force=True)
    assert [s["step"] for s in steps][-1] == "crumb"
    assert not any(r.get("query", "").startswith("SELECT version") for r in neon.requests)


@pytest.mark.asyncio
async def test_a_batch_is_one_request_with_every_statement():
    neon = FakeNeon()
    vault = _vault(PLAIN, neon)
    await vault._execute_batch(["CREATE TABLE a (x int)", "CREATE TABLE b (y int)"])
    assert neon.requests == [{"queries": [
        {"query": "CREATE TABLE a (x int)", "params": []},
        {"query": "CREATE TABLE b (y int)", "params": []},
    ]}]


def test_every_statement_is_idempotent():
    t = lambda n: f"op_test.{n}"
    for name, statements in concerns():
        for sql in statements(t, "op_test_"):
            assert "IF NOT EXISTS" in sql or "IF EXISTS" in sql, f"{name}: {sql[:60]}"


def test_duplicate_pricing_indexes_are_dropped_only_on_a_prefixed_schema():
    from tollbooth.pricing_store import schema_statements

    prefixed = schema_statements(lambda n: f"op_test.{n}", "op_test_")
    plain = schema_statements(lambda n: n, "")
    assert "DROP INDEX IF EXISTS op_test.one_active_per_operator" in prefixed
    assert not any(s.startswith("DROP INDEX") for s in plain), "on public, the names are the same index"


def test_a_ddl_change_cannot_ship_behind_a_stale_crumb():
    got = digest(lambda n: f"op_test.{n}", "op_test_")
    assert got == SCHEMA_DIGEST, (
        "The schema DDL changed. Bump SCHEMA_VERSION in tollbooth/vaults/schema.py "
        f"and set SCHEMA_DIGEST = {got!r} — otherwise every prepared database keeps "
        "a crumb that says the new tables are already there."
    )
