"""Tests for PatronSessionCache."""

import pytest
from dataclasses import dataclass

from tollbooth.patron_session import PatronSessionCache


@dataclass
class FakeSession:
    api_key: str


class FakeRuntime:
    """Minimal runtime mock for PatronSessionCache tests."""

    def __init__(self):
        self._vault: dict[tuple[str, str], dict] = {}

    async def load_patron_session(self, npub: str, service: str = "") -> dict | None:
        return self._vault.get((service, npub))

    async def store_patron_session(self, npub: str, data: dict, service: str = "") -> None:
        self._vault[(service, npub)] = data


NPUB = "npub1" + "a" * 58


async def _restore(creds: dict) -> FakeSession | None:
    if "api_key" not in creds:
        return None
    return FakeSession(api_key=creds["api_key"])


def _make_cache(rt=None) -> PatronSessionCache[FakeSession]:
    return PatronSessionCache(
        runtime=rt or FakeRuntime(),
        service="test-svc",
        restore=_restore,
        ttl_seconds=3600,
    )


class TestPatronSessionCache:

    @pytest.mark.asyncio
    async def test_store_and_get(self):
        rt = FakeRuntime()
        cache = _make_cache(rt)
        s = FakeSession(api_key="k1")
        await cache.store("u1", NPUB, s, vault_data={"api_key": "k1"})

        assert cache.get("u1") is s
        assert rt._vault[("test-svc", NPUB)] == {"api_key": "k1"}

    @pytest.mark.asyncio
    async def test_get_or_restore_from_memory(self):
        cache = _make_cache()
        s = FakeSession(api_key="k1")
        cache.set_local("u1", s, npub=NPUB)

        result = await cache.get_or_restore("u1", NPUB)
        assert result is s

    @pytest.mark.asyncio
    async def test_get_or_restore_from_vault(self):
        rt = FakeRuntime()
        rt._vault[("test-svc", NPUB)] = {"api_key": "restored-key"}
        cache = _make_cache(rt)

        result = await cache.get_or_restore("u1", NPUB)
        assert result is not None
        assert result.api_key == "restored-key"
        # Should now be in memory cache
        assert cache.get("u1") is result

    @pytest.mark.asyncio
    async def test_get_or_restore_no_vault_data(self):
        cache = _make_cache()
        result = await cache.get_or_restore("u1", NPUB)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_restore_no_npub(self):
        cache = _make_cache()
        result = await cache.get_or_restore("u1", "")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_restore_insufficient_creds(self):
        rt = FakeRuntime()
        rt._vault[("test-svc", NPUB)] = {"not_api_key": "useless"}
        cache = _make_cache(rt)

        result = await cache.get_or_restore("u1", NPUB)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate(self):
        cache = _make_cache()
        cache.set_local("u1", FakeSession("k"), npub=NPUB)
        assert cache.get("u1") is not None

        cache.invalidate("u1")
        assert cache.get("u1") is None

    @pytest.mark.asyncio
    async def test_clear_local(self):
        cache = _make_cache()
        cache.set_local("u1", FakeSession("k"), npub=NPUB)
        cache.clear_local("u1")
        assert cache.get("u1") is None

    @pytest.mark.asyncio
    async def test_npub_mapping_persists_across_restore(self):
        rt = FakeRuntime()
        rt._vault[("test-svc", NPUB)] = {"api_key": "k1"}
        cache = _make_cache(rt)

        # First call with explicit npub
        await cache.get_or_restore("u1", NPUB)
        # Second call without npub — should use stored mapping
        cache._cache.clear("u1")  # clear memory but keep npub mapping
        result = await cache.get_or_restore("u1")
        assert result is not None
        assert result.api_key == "k1"

    @pytest.mark.asyncio
    async def test_cache_property(self):
        cache = _make_cache()
        assert cache.cache.ttl_seconds == 3600
