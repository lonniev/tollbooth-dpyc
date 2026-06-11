"""Tests for the generic SessionCache."""

import time
from dataclasses import dataclass

from tollbooth.session_cache import SessionCache


@dataclass
class FakeSession:
    api_key: str
    user_id: str


class TestSessionCache:

    def test_set_and_get(self):
        cache: SessionCache[FakeSession] = SessionCache(ttl_seconds=60)
        s = FakeSession(api_key="key1", user_id="u1")
        cache.set("u1", s)
        assert cache.get("u1") is s

    def test_get_missing_returns_none(self):
        cache: SessionCache[FakeSession] = SessionCache()
        assert cache.get("nonexistent") is None

    def test_expiry(self):
        cache: SessionCache[FakeSession] = SessionCache(ttl_seconds=1)
        cache.set("u1", FakeSession("k", "u1"))
        assert cache.get("u1") is not None

        # Manually expire by backdating
        session, _ = cache._entries["u1"]
        cache._entries["u1"] = (session, time.time() - 2)
        assert cache.get("u1") is None

    def test_clear(self):
        cache: SessionCache[FakeSession] = SessionCache()
        s = FakeSession("k", "u1")
        cache.set("u1", s)
        removed = cache.clear("u1")
        assert removed is s
        assert cache.get("u1") is None

    def test_clear_missing(self):
        cache: SessionCache[FakeSession] = SessionCache()
        assert cache.clear("nope") is None

    def test_clear_all(self):
        cache: SessionCache[FakeSession] = SessionCache()
        cache.set("a", FakeSession("k1", "a"))
        cache.set("b", FakeSession("k2", "b"))
        count = cache.clear_all()
        assert count == 2
        assert len(cache) == 0

    def test_contains(self):
        cache: SessionCache[FakeSession] = SessionCache()
        cache.set("u1", FakeSession("k", "u1"))
        assert "u1" in cache
        assert "u2" not in cache

    def test_len(self):
        cache: SessionCache[FakeSession] = SessionCache()
        assert len(cache) == 0
        cache.set("a", FakeSession("k", "a"))
        cache.set("b", FakeSession("k", "b"))
        assert len(cache) == 2

    def test_set_returns_session(self):
        cache: SessionCache[FakeSession] = SessionCache()
        s = FakeSession("k", "u1")
        returned = cache.set("u1", s)
        assert returned is s

    def test_overwrite_resets_ttl(self):
        cache: SessionCache[FakeSession] = SessionCache(ttl_seconds=10)
        s1 = FakeSession("k1", "u1")
        cache.set("u1", s1)

        # Backdate to near-expiry
        session, _ = cache._entries["u1"]
        cache._entries["u1"] = (session, time.time() - 9)

        # Overwrite with fresh session
        s2 = FakeSession("k2", "u1")
        cache.set("u1", s2)
        assert cache.get("u1") is s2  # fresh TTL, not expired

    def test_ttl_property(self):
        cache: SessionCache[FakeSession] = SessionCache(ttl_seconds=42)
        assert cache.ttl_seconds == 42

    # ── M3.2: max-size eviction + opportunistic expiry sweep ──────────────

    def test_unbounded_by_default(self):
        cache: SessionCache[FakeSession] = SessionCache()
        for i in range(50):
            cache.set(f"u{i}", FakeSession("k", f"u{i}"))
        assert len(cache) == 50  # no eviction without max_size

    def test_max_size_evicts_oldest_written(self):
        cache: SessionCache[FakeSession] = SessionCache(max_size=2)
        cache.set("a", FakeSession("k", "a"))
        cache.set("b", FakeSession("k", "b"))
        cache.set("c", FakeSession("k", "c"))  # over cap → evict oldest (a)
        assert len(cache) == 2
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_reset_moves_to_most_recent_for_eviction(self):
        cache: SessionCache[FakeSession] = SessionCache(max_size=2)
        cache.set("a", FakeSession("k", "a"))
        cache.set("b", FakeSession("k", "b"))
        cache.set("a", FakeSession("k", "a2"))  # a is now most-recently-written
        cache.set("c", FakeSession("k", "c"))   # evicts b (least recent), not a
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_set_sweeps_expired_entries(self):
        cache: SessionCache[FakeSession] = SessionCache(ttl_seconds=10)
        cache.set("a", FakeSession("k", "a"))
        cache.set("b", FakeSession("k", "b"))
        # Backdate "a" past its TTL
        session, _ = cache._entries["a"]
        cache._entries["a"] = (session, time.time() - 20)
        # A subsequent set proactively sweeps the expired "a"
        cache.set("c", FakeSession("k", "c"))
        assert "a" not in cache._entries  # swept, not lingering
        assert len(cache) == 2  # b + c
