"""The one write path for a patron's ledger, and a read cache beside it.

**Read, apply, compare-and-swap, re-apply; nothing else writes.**

``mutate(user_id, fn)`` is the only code that writes a ledger row. Under the
per-user lock it reads the current ledger *and the version it was read at*,
applies ``fn``, and writes back with that exact version. If another writer —
another replica, or another coroutine of this process — landed first, the
store refuses, and ``mutate`` re-reads and re-applies. Money functions are
idempotent against fresh state (a settled invoice is credited once, a debit
re-checks the balance), so re-application is always safe.

Usage counters for free calls do not get a write of their own. ``note_usage``
accumulates them in memory as deltas; the next ``mutate`` for that patron folds
them into the fresh ledger it is about to write, and a periodic ``fold_usage``
pass writes the deltas of patrons who only made free calls. A delta applied to
fresh state cannot overwrite anything; a fold that fails keeps its deltas for
the next pass; a process that dies loses a statistic, never a sat.

Reads (``get``, ``get_fresh``) never write and never change what they return.
No expiry is invented here: tranche lifetimes come from the pricing model and
are applied by the callers of ``mutate``.

History: 0.62.0 made money writes read-apply-CAS; 0.78.0 moved the last
money paths onto it; what remained was a write-behind flush of whole-ledger
snapshots for free-call counters, whose version came from a per-user cache
rather than from the snapshot. On 2026-09-27 such a flush landed after a
settlement and erased 1,000 sats. That flush, and every path that could write
a ledger it had not just read, is gone.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from tollbooth.ledger import UserLedger
from tollbooth.vault_backend import (
    LedgerUnavailableError,
    LedgerVersionConflict,
    LedgerWriteError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    from tollbooth.vault_backend import VaultBackend

    _T = TypeVar("_T")

logger = logging.getLogger(__name__)

# How many times a mutation re-fetches + re-applies after losing a CAS race
# before giving up. Conflicts only happen when writers hit the same ledger in
# the same instant, so a handful of retries is ample.
_MAX_WRITE_RETRIES = 6


class LedgerCache:
    """Read cache plus the single write path for patron ledgers.

    - ``get()`` returns the cached snapshot, loading it on a miss.
    - ``get_fresh()`` reloads from the store and replaces the snapshot.
    - ``mutate()`` is the only writer: lock, read with version, apply, CAS write.
    - ``note_usage()`` records a free call; ``fold_usage()`` writes pending
      counters through ``mutate``.
    """

    def __init__(
        self,
        vault: VaultBackend,
        maxsize: int = 20,
        fold_interval_secs: int = 60,
    ) -> None:
        self._vault = vault
        self._maxsize = maxsize
        self._fold_interval = fold_interval_secs
        self._entries: OrderedDict[str, UserLedger] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._usage: dict[str, dict[str, int]] = {}
        self._fold_task: asyncio.Task[None] | None = None
        self._folds: int = 0

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    # -- reads -----------------------------------------------------------------

    async def get(self, user_id: str) -> UserLedger:
        """Return the cached ledger, loading it from the store on a miss.

        Never writes. A store that cannot be read yields an empty ledger flagged
        ``_vault_unavailable`` and NOT cached, so the next call tries again.
        """
        async with self._get_lock(user_id):
            cached = self._entries.get(user_id)
            if cached is not None:
                self._entries.move_to_end(user_id)
                return cached
            loaded = await self._load(user_id)
            if loaded is None:
                ledger = UserLedger()
                ledger._vault_unavailable = True  # type: ignore[attr-defined]
                return ledger
            ledger, _version = loaded
            self._install(user_id, ledger)
            return ledger

    async def get_fresh(self, user_id: str) -> UserLedger:
        """Reload the ledger from the definitive store and replace the snapshot.

        On a read failure the returned ledger is flagged ``_vault_unavailable``
        (and not cached), so callers can refuse to show a phantom balance.
        """
        async with self._get_lock(user_id):
            loaded = await self._load(user_id)
            if loaded is None:
                ledger = UserLedger()
                ledger._vault_unavailable = True  # type: ignore[attr-defined]
                return ledger
            ledger, _version = loaded
            self._install(user_id, ledger)
            return ledger

    # -- the write path --------------------------------------------------------

    async def mutate(
        self,
        user_id: str,
        fn: Callable[[UserLedger], _T],
        *,
        retries: int = _MAX_WRITE_RETRIES,
    ) -> _T:
        """Read the current ledger, apply ``fn``, and write it back at the
        version it was read at.

        Pending free-call usage for ``user_id`` is folded into the ledger first,
        so it rides along with whatever ``fn`` does. ``fn`` returning ``False``
        means "nothing to persist" (a debit found the balance short): nothing
        is written, the deltas are kept for a later fold, and ``False`` is
        returned. Any other value is written through and returned.

        On a version conflict the whole step repeats against fresh state, up to
        ``retries`` times. The snapshot this cache serves is always the ledger
        last read or written here.

        Raises ``LedgerUnavailableError`` if the store cannot be read (a
        mutation is never applied to an empty fallback) and ``LedgerWriteError``
        when the retries are exhausted.
        """
        async with self._get_lock(user_id):
            deltas = self._usage.pop(user_id, None)
            try:
                for _ in range(retries):
                    loaded = await self._load(user_id)
                    if loaded is None:
                        raise LedgerUnavailableError(
                            f"definitive ledger store unreadable for {user_id[:20]}"
                        )
                    ledger, version = loaded
                    if deltas:
                        for tool, calls in deltas.items():
                            ledger.record_usage(tool, calls)
                    result = fn(ledger)
                    if result is False:
                        # Nothing to persist. The snapshot read is still the best
                        # one this process has; the deltas wait for a real write.
                        self._install(user_id, ledger)
                        self._requeue(user_id, deltas)
                        deltas = None
                        return result
                    try:
                        await self._vault.store_ledger(user_id, ledger.to_json(), version)
                    except LedgerVersionConflict:
                        continue  # lost the race — re-read, re-fold, re-apply
                    self._install(user_id, ledger)
                    deltas = None
                    return result
                raise LedgerWriteError(
                    f"ledger write for {user_id[:20]} lost {retries} consecutive CAS races"
                )
            finally:
                # A read failure or exhausted retries must not lose the counters.
                self._requeue(user_id, deltas)

    async def debit(self, user_id: str, tool_name: str, cost: int) -> bool:
        """Write-through debit against fresh state. False when the balance is short."""
        return await self.mutate(user_id, lambda ledger: ledger.debit(tool_name, cost))

    async def credit(
        self,
        user_id: str,
        api_sats: int,
        invoice_id: str,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Write-through credit — adds a tranche against fresh state."""
        await self.mutate(
            user_id,
            lambda ledger: ledger.credit_deposit(api_sats, invoice_id, ttl_seconds=ttl_seconds),
        )

    # -- usage accounting ------------------------------------------------------

    def note_usage(self, user_id: str, tool_name: str) -> None:
        """Record one free call. Nothing is written until the next fold or mutate."""
        per_user = self._usage.setdefault(user_id, {})
        per_user[tool_name] = per_user.get(tool_name, 0) + 1

    async def fold_usage(self) -> int:
        """Write every patron's pending usage counters through ``mutate``.

        Returns how many patrons were written. A patron whose write failed keeps
        its deltas for the next pass.
        """
        folded = 0
        for user_id in list(self._usage):
            try:
                await self.mutate(user_id, lambda _ledger: True)
                folded += 1
            except (LedgerUnavailableError, LedgerWriteError) as exc:
                logger.info("Usage fold deferred for %s: %s", user_id[:20], exc)
            except Exception:  # a fold is accounting, never a fault
                logger.warning("Usage fold failed for %s (kept for next pass).", user_id[:20], exc_info=True)
        if folded:
            self._folds += folded
        return folded

    async def start_usage_fold(self) -> None:
        """Start the periodic fold of pending usage counters."""
        if self._fold_task is None:
            self._fold_task = asyncio.create_task(self._fold_loop())

    async def _fold_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._fold_interval)
                if self._usage:
                    await self.fold_usage()
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the periodic fold and write whatever usage is still pending."""
        if self._fold_task is not None:
            self._fold_task.cancel()
            try:
                await self._fold_task
            except asyncio.CancelledError:
                pass
            self._fold_task = None
        await self.fold_usage()

    # -- internals -------------------------------------------------------------

    async def _load(self, user_id: str) -> tuple[UserLedger, int | None] | None:
        """``(ledger, version)`` from the store, ``(empty, None)`` for a patron
        with no row yet, or ``None`` when the store could not be read."""
        try:
            row = await self._vault.fetch_ledger(user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load ledger from vault for %s: %s: %s",
                user_id[:20], type(exc).__name__, exc,
            )
            return None
        if row is None:
            return UserLedger(), None
        ledger_json, version = row
        return UserLedger.from_json(ledger_json), version

    def _install(self, user_id: str, ledger: UserLedger) -> None:
        self._entries[user_id] = ledger
        self._entries.move_to_end(user_id)
        while len(self._entries) > self._maxsize:
            evicted, _ = self._entries.popitem(last=False)
            self._locks.pop(evicted, None)

    def _requeue(self, user_id: str, deltas: dict[str, int] | None) -> None:
        if not deltas:
            return
        per_user = self._usage.setdefault(user_id, {})
        for tool, calls in deltas.items():
            per_user[tool] = per_user.get(tool, 0) + calls

    # -- metrics ---------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of ledgers currently cached."""
        return len(self._entries)

    @property
    def pending_usage(self) -> int:
        """Number of patrons with usage counters not yet written."""
        return len(self._usage)

    def health(self) -> dict[str, object]:
        """Cache health for monitoring."""
        return {
            "cache_size": self.size,
            "pending_usage": self.pending_usage,
            "usage_folds": self._folds,
            "fold_interval_secs": self._fold_interval,
            "usage_fold_running": self._fold_task is not None and not self._fold_task.done(),
        }
