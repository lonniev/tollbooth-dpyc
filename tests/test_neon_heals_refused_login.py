"""A vault whose database password was rotated takes the Authority's newer connection string, once."""

import asyncio

import httpx
import pytest

from tollbooth.persistence_errors import is_refused_login
from tollbooth.vaults.neon import NeonQueryError, NeonVault

OLD = "postgres://op:old@ep-x-pooler.us-east-2.aws.neon.tech/neondb"
NEW = "postgres://op:new@ep-x-pooler.us-east-2.aws.neon.tech/neondb"
ELSEWHERE = "postgres://op:new@ep-other-pooler.us-east-2.aws.neon.tech/neondb"


def vault_with(refresh, *, password_ok="new", seen=None):
    seen = seen if seen is not None else []

    def handle(request: httpx.Request) -> httpx.Response:
        url = request.headers["Neon-Connection-String"]
        seen.append(url)
        if f":{password_ok}@" not in url:
            return httpx.Response(400, json={"message": "password authentication failed for user \"op\"",
                                             "code": "28P01"})
        body = request.read().decode()
        if '"queries"' in body:
            return httpx.Response(200, json={"results": [{"rows": [{"ok": 1}]}]})
        return httpx.Response(200, json={"rows": [{"ok": 1}], "rowCount": 1})

    v = NeonVault(OLD, encryption_nsec_hex="11" * 32, refresh_database_url=refresh)
    v._client = httpx.AsyncClient(transport=httpx.MockTransport(handle),
                                  headers={"Neon-Connection-String": OLD, "Content-Type": "application/json"})
    return v, seen


def counting(url):
    calls = []

    async def refresh():
        calls.append(1)
        return url

    return refresh, calls


async def test_a_refused_login_takes_the_newer_url_and_retries():
    refresh, calls = counting(NEW)
    v, seen = vault_with(refresh)
    assert (await v._execute("SELECT 1"))["rows"] == [{"ok": 1}]
    assert calls == [1] and seen == [OLD, NEW]
    assert (await v._execute("SELECT 1"))["rows"] == [{"ok": 1}]  # stays on the new one
    assert seen[-1] == NEW and calls == [1]


async def test_the_batch_path_heals_too():
    refresh, calls = counting(NEW)
    v, _ = vault_with(refresh)
    assert await v._execute_batch(["SELECT 1"]) == [{"rows": [{"ok": 1}]}]
    assert calls == [1]


async def test_without_a_refresher_the_refusal_stands():
    v, _ = vault_with(None)
    with pytest.raises(NeonQueryError):
        await v._execute("SELECT 1")


async def test_no_newer_url_raises_and_is_not_asked_again_within_a_minute():
    refresh, calls = counting(OLD)
    v, _ = vault_with(refresh)
    for _ in range(3):
        with pytest.raises(NeonQueryError):
            await v._execute("SELECT 1")
    assert calls == [1]


async def test_a_url_for_another_database_host_is_not_taken():
    refresh, _ = counting(ELSEWHERE)
    v, seen = vault_with(refresh)
    with pytest.raises(NeonQueryError):
        await v._execute("SELECT 1")
    assert ELSEWHERE not in seen


async def test_other_sql_errors_never_ask():
    calls = []

    async def refresh():
        calls.append(1)
        return NEW

    v = NeonVault(OLD, encryption_nsec_hex="11" * 32, refresh_database_url=refresh)
    v._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json={"message": "relation \"x\" does not exist", "code": "42P01"})))
    with pytest.raises(NeonQueryError):
        await v._execute("SELECT * FROM x")
    assert calls == []


async def test_many_refused_calls_at_once_read_the_relays_once():
    calls = []

    async def refresh():
        calls.append(1)
        await asyncio.sleep(0.01)
        return NEW

    v, _ = vault_with(refresh)
    results = await asyncio.gather(*(v._execute("SELECT 1") for _ in range(5)))
    assert all(r["rows"] == [{"ok": 1}] for r in results)
    assert calls == [1]


def test_a_refused_login_is_named():
    assert is_refused_login(NeonQueryError("x", code="28P01"))
    assert is_refused_login(NeonQueryError("Neon HTTP 400: password authentication failed for user"))
    assert not is_refused_login(NeonQueryError("x", code="42P01"))


async def test_the_runtime_reads_the_newest_bootstrap(monkeypatch):
    from tollbooth import bootstrap as bs
    from tollbooth import runtime

    bs._cached_result = bs.BootstrapResult(success=True, neon_database_url=OLD)

    async def fresh(*a, **k):
        assert bs._cached_result is None, "the stale config must be forgotten first"
        return bs.BootstrapResult(success=True, neon_database_url=NEW)

    monkeypatch.setattr(bs, "ensure_bootstrapped", fresh)
    assert await runtime._newest_neon_url() == NEW
    bs._cached_result = None
