"""In-memory LRU cache for UserLedger with write-behind flush to vault.

The cache is the hot path for all credit operations. The vault is the
durable backing store, updated asynchronously every ``flush_interval_secs``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tollbooth.ledger import UserLedger
from tollbooth.vault_backend import (
    LedgerUnavailableError,
    LedgerVersionConflict,
    LedgerWriteError,
)

if TYPE_CHECKING:
    from typing import Callable, TypeVar

    from tollbooth.vault_backend import VaultBackend

    _T = TypeVar("_T")

logger = logging.getLogger(__name__)

# How many times a mutation re-fetches + re-applies after losing a CAS race
# before giving up. Conflicts only happen when multiple replicas write the
# same ledger in the same instant, so a handful of retries is ample.
_MAX_WRITE_RETRIES = 6


@dataclass
class _CacheEntry:
    """Internal cache entry wrapping a UserLedger with dirty tracking."""

    ledger: UserLedger
    dirty: bool = False
    dirty_count: int = 0
    last_flush_time: float = field(default_factory=time.monotonic)


class LedgerCache:
    """LRU cache for UserLedger objects with write-behind flush.

    - ``get()`` returns a cached ledger or loads from vault on miss.
    - Mutations should be followed by ``mark_dirty(user_id)``.
    - A background task flushes dirty entries to the vault periodically.
    - On LRU eviction, dirty entries are flushed synchronously.
    - Per-user asyncio locks prevent concurrent access races.
    """

    def __init__(
        self,
        vault: VaultBackend,
        maxsize: int = 20,
        flush_interval_secs: int = 60,
        flush_retries: int = 1,
        flush_retry_delay: float = 2.0,
        flush_batch_size: int = 10,
        flush_staleness_secs: float = 120.0,
    ) -> None:
        self._vault = vault
        self._maxsize = maxsize
        self._flush_interval = flush_interval_secs
        self._flush_retries = flush_retries
        self._flush_retry_delay = flush_retry_delay
        self._flush_batch_size = flush_batch_size
        self._flush_staleness_secs = flush_staleness_secs
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._flush_task: asyncio.Task[None] | None = None
        self._last_flush_at: str | None = None
        self._total_flushes: int = 0

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a per-user lock."""
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def _flush_due_entry(self, entry: _CacheEntry) -> bool:
        """Check if this entry needs flushing based on count or staleness."""
        if not entry.dirty:
            return False
        if entry.dirty_count >= self._flush_batch_size:
            return True
        if time.monotonic() - entry.last_flush_time > self._flush_staleness_secs:
            return True
        return False

    def flush_due(self, user_id: str) -> bool:
        """Check if a user's cache entry is due for flushing."""
        entry = self._entries.get(user_id)
        if entry is None:
            return False
        return self._flush_due_entry(entry)

    async def get(self, user_id: str) -> UserLedger:
        """Return the cached ledger, loading from vault on miss."""
        lock = self._get_lock(user_id)
        async with lock:
            if user_id in self._entries:
                entry = self._entries[user_id]
                # Migrate any perpetual tranches still in cache from pre-TTL code
                if self._migrate_perpetual_tranches(entry.ledger):
                    entry.dirty = True
                if self._flush_due_entry(entry):
                    self._fire_and_forget_flush(user_id, entry)
                self._entries.move_to_end(user_id)
                return entry.ledger

            # Cache miss — load from vault
            ledger, from_vault = await self._load_from_vault(user_id)

            if not from_vault:
                # Vault fetch failed (cold start, Neon not ready).
                # Return the empty ledger but do NOT cache it — next call
                # will retry the vault fetch after it has had time to connect.
                ledger._vault_unavailable = True  # type: ignore[attr-defined]
                return ledger

            # Belt-and-suspenders: migrate perpetual tranches even on fresh load
            # (from_dict should handle this, but catch any edge cases)
            dirty = self._migrate_perpetual_tranches(ledger)

            # Evict LRU if at capacity
            while len(self._entries) >= self._maxsize:
                self._fire_and_forget_evict()

            entry = _CacheEntry(ledger=ledger)
            if dirty:
                entry.dirty = True
            self._entries[user_id] = entry
            self._entries.move_to_end(user_id)
            return ledger

    @staticmethod
    def _migrate_perpetual_tranches(ledger: UserLedger) -> bool:
        """Assign 7-day TTL to any cached tranches that still lack expiration.

        Returns True if any tranches were migrated (entry should be flushed).
        """
        from datetime import datetime, timedelta, timezone

        migrated = False
        for t in ledger.tranches:
            if t.expires_at is None and t.remaining_sats > 0:
                t.expires_at = (
                    datetime.now(timezone.utc) + timedelta(days=7)
                ).isoformat()
                migrated = True
        if migrated:
            logger.info(
                "Migrated perpetual tranches for cached ledger "
                "(patron=%s, count=%d)",
                getattr(ledger, "user_id", "?"),
                sum(1 for t in ledger.tranches if t.expires_at is not None),
            )
        return migrated

    def mark_dirty(self, user_id: str) -> None:
        """Mark a cached entry as dirty (needs flush to vault)."""
        entry = self._entries.get(user_id)
        if entry:
            entry.dirty = True
            entry.dirty_count += 1

    async def flush_user(self, user_id: str) -> bool:
        """Immediately flush a single user's entry to vault.

        Use for credit-critical paths (check_payment, purchase_credits)
        where data MUST be durable before returning success.
        Returns True on success, False on failure (logged, not raised).
        """
        entry = self._entries.get(user_id)
        if not entry or not entry.dirty:
            return True  # Nothing to flush
        return await self._flush_entry(user_id, entry)

    async def mutate(
        self,
        user_id: str,
        fn: Callable[[UserLedger], _T],
        *,
        retries: int = _MAX_WRITE_RETRIES,
    ) -> _T:
        """Atomically read-modify-**write-through** a ledger against the vault.

        The definitive store is the source of truth. This fetches the CURRENT
        ledger from the vault (never a stale in-memory copy), applies
        ``fn(ledger)``, and CAS-writes it back. If another replica wrote first
        (``LedgerVersionConflict``) it re-fetches and re-applies ``fn`` on the
        fresh state, up to ``retries`` — so concurrent replicas can't clobber
        each other's balance changes.

        ``fn`` returning ``False`` signals a no-op (e.g. debit found
        insufficient balance): nothing is written and ``False`` is returned.
        Any other return value is written through and returned.

        Raises ``LedgerUnavailableError`` if the vault can't be read (so a cold
        store never makes a mutation apply to an empty fallback ledger and zero
        a real balance), or ``LedgerWriteError`` if retries are exhausted.
        """
        lock = self._get_lock(user_id)
        async with lock:
            for _ in range(retries):
                ledger, from_vault = await self._load_from_vault(user_id)
                if not from_vault:
                    raise LedgerUnavailableError(
                        f"definitive ledger store unreadable for {user_id[:20]}"
                    )
                result = fn(ledger)
                if result is False:
                    return result  # explicit no-op — nothing to persist
                try:
                    await self._vault.store_ledger(user_id, ledger.to_json())
                except LedgerVersionConflict:
                    continue  # lost the race — re-fetch fresh state and re-apply
                # Persisted. Cache the authoritative post-write ledger (clean)
                # so same-process reads see it without another round-trip.
                self._entries[user_id] = _CacheEntry(ledger=ledger)
                self._entries.move_to_end(user_id)
                return result
        raise LedgerWriteError(
            f"ledger write for {user_id[:20]} lost {retries} consecutive CAS races"
        )

    async def debit(self, user_id: str, tool_name: str, cost: int) -> bool:
        """Atomic write-through debit. Returns False on insufficient balance.

        Read-modify-write against the definitive store with conflict retry, so
        two replicas can't both spend the same sats.
        """
        return await self.mutate(user_id, lambda ledger: ledger.debit(tool_name, cost))

    async def credit(
        self,
        user_id: str,
        api_sats: int,
        invoice_id: str,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Atomic write-through credit — adds a tranche against fresh state."""
        await self.mutate(
            user_id,
            lambda ledger: ledger.credit_deposit(api_sats, invoice_id, ttl_seconds=ttl_seconds),
        )

    async def get_fresh(self, user_id: str) -> UserLedger:
        """Force-reload the ledger from the definitive store, refreshing the cache.

        Use immediately before a mutate-then-``flush_user`` sequence so the
        subsequent CAS write matches the CURRENT version — not a stale cached
        one that would conflict forever. On a vault-read failure the returned
        ledger is flagged ``_vault_unavailable`` (and NOT cached), so callers
        can refuse to credit a phantom balance during a cold start.
        """
        lock = self._get_lock(user_id)
        async with lock:
            ledger, from_vault = await self._load_from_vault(user_id)
            if not from_vault:
                ledger._vault_unavailable = True  # type: ignore[attr-defined]
                return ledger
            entry = _CacheEntry(ledger=ledger)
            self._entries[user_id] = entry
            self._entries.move_to_end(user_id)
            return ledger

    async def write_through_credit(self, user_id: str) -> bool:
        """Mark dirty and immediately flush. Use for credit settlements."""
        self.mark_dirty(user_id)
        return await self.flush_user(user_id)

    async def _load_from_vault(self, user_id: str) -> tuple[UserLedger, bool]:
        """Load ledger JSON from vault.

        Returns (ledger, from_vault) — from_vault is True if the data came
        from Neon (or the user genuinely has no ledger), False if the vault
        fetch failed and we're returning an empty fallback that should NOT
        be cached (to avoid masking a cold-start timing issue).
        """
        try:
            ledger_json = await self._vault.fetch_ledger(user_id)
        except Exception as exc:
            logger.warning(
                "Failed to load ledger from vault for %s — returning uncached empty ledger. Underlying error: %s: %s",
                user_id, type(exc).__name__, exc,
            )
            return UserLedger(), False

        if ledger_json is None:
            return UserLedger(), True
        return UserLedger.from_json(ledger_json), True

    def _fire_and_forget_flush(
        self, user_id: str, entry: _CacheEntry,
    ) -> None:
        """Schedule a flush as a background task — never blocks the hot path.

        Idempotent: if the task fails or two tasks race, the worst outcome is
        a duplicate Neon write (full ledger overwrite, not an increment).
        """
        asyncio.create_task(self._safe_flush(user_id, entry))

    def _fire_and_forget_evict(self) -> None:
        """Evict the LRU entry, flushing dirty state in the background."""
        if not self._entries:
            return
        user_id, entry = next(iter(self._entries.items()))
        if entry.dirty:
            asyncio.create_task(self._safe_flush(user_id, entry))
        del self._entries[user_id]
        self._locks.pop(user_id, None)

    async def _safe_flush(self, user_id: str, entry: _CacheEntry) -> None:
        """Flush wrapper that swallows exceptions — safe for create_task."""
        try:
            await self._flush_entry(user_id, entry)
        except Exception:
            logger.warning(
                "Background flush failed for %s (swallowed).", user_id,
            )

    async def _evict_lru(self) -> None:
        """Evict the least-recently-used entry, flushing if dirty.

        Retained for callers that need awaited eviction (e.g. shutdown).
        The hot path uses ``_fire_and_forget_evict()`` instead.
        """
        if not self._entries:
            return
        user_id, entry = next(iter(self._entries.items()))
        if entry.dirty:
            await self._flush_entry(user_id, entry)
        del self._entries[user_id]
        self._locks.pop(user_id, None)

    async def _flush_entry(self, user_id: str, entry: _CacheEntry) -> bool:
        """Flush a single entry to vault with retry. Returns True on success."""
        from datetime import datetime, timezone

        max_attempts = 1 + self._flush_retries
        for attempt in range(max_attempts):
            try:
                ledger_json = entry.ledger.to_json()
                version = await self._vault.store_ledger(user_id, ledger_json)
                logger.info(
                    "Flushed ledger for %s to vault (v%s, %d bytes).",
                    user_id[:20], version, len(ledger_json),
                )
                entry.dirty = False
                entry.dirty_count = 0
                entry.last_flush_time = time.monotonic()
                self._last_flush_at = datetime.now(timezone.utc).isoformat()
                self._total_flushes += 1
                return True
            except Exception:
                if attempt < max_attempts - 1:
                    logger.warning(
                        "Flush attempt %d/%d failed for %s, retrying in %.1fs...",
                        attempt + 1, max_attempts, user_id, self._flush_retry_delay,
                    )
                    await asyncio.sleep(self._flush_retry_delay)
                else:
                    logger.warning(
                        "Failed to flush ledger to vault for %s after %d attempt(s).",
                        user_id, max_attempts,
                    )
        return False

    async def flush_dirty(self) -> int:
        """Flush all dirty entries to vault. Returns count of flushed entries."""
        flushed = 0
        for user_id, entry in list(self._entries.items()):
            if entry.dirty:
                if await self._flush_entry(user_id, entry):
                    flushed += 1
        return flushed

    async def snapshot_all(self, timestamp: str) -> int:
        """Snapshot all cached ledgers to vault. Returns count of snapshots created."""
        snapped = 0
        for user_id, entry in list(self._entries.items()):
            try:
                result = await self._vault.snapshot_ledger(
                    user_id, entry.ledger.to_json(), timestamp
                )
                if result is not None:
                    snapped += 1
            except Exception:
                logger.warning("Failed to snapshot ledger for %s.", user_id)
        return snapped

    async def flush_all(self) -> int:
        """Flush every dirty entry (used during shutdown). Returns flush count."""
        return await self.flush_dirty()

    async def start_background_flush(self) -> None:
        """Start the periodic background flush task."""
        if self._flush_task is not None:
            return
        self._flush_task = asyncio.create_task(self._background_flush_loop())

    async def _background_flush_loop(self) -> None:
        """Periodically flush dirty entries until cancelled."""
        logger.warning(
            "Background flush loop started (interval=%ds). "
            "In serverless envs, opportunistic flush handles persistence instead.",
            self._flush_interval,
        )
        cycles = 0
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                count = await self.flush_dirty()
                cycles += 1
                if count > 0:
                    logger.info(
                        "Background flush: wrote %d ledger(s) "
                        "(cycle %d, total flushes: %d).",
                        count, cycles, self._total_flushes,
                    )
                elif cycles % 10 == 0:
                    # Heartbeat every 10 cycles even when idle
                    logger.info(
                        "Background flush heartbeat: cycle %d, "
                        "cache size %d, dirty %d, total flushes %d.",
                        cycles, self.size, self.dirty_count, self._total_flushes,
                    )
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Cancel background flush and flush all remaining dirty entries."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self.flush_all()

    @property
    def size(self) -> int:
        """Number of entries currently in cache."""
        return len(self._entries)

    @property
    def dirty_count(self) -> int:
        """Number of dirty (unflushed) entries in cache."""
        return sum(1 for e in self._entries.values() if e.dirty)

    def health(self) -> dict[str, object]:
        """Return cache health metrics for monitoring."""
        return {
            "cache_size": self.size,
            "dirty_entries": self.dirty_count,
            "last_flush_at": self._last_flush_at,
            "total_flushes": self._total_flushes,
            "flush_retries": self._flush_retries,
            "flush_retry_delay": self._flush_retry_delay,
            "flush_batch_size": self._flush_batch_size,
            "flush_staleness_secs": self._flush_staleness_secs,
            "background_flush_running": self._flush_task is not None
                                        and not self._flush_task.done(),
        }
