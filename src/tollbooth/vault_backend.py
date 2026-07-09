"""Abstract persistence interface for commerce state (ledger storage).

Defines the VaultBackend Protocol that LedgerCache depends on.
Concrete implementations (e.g., PersonalBrainVault) live elsewhere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class LedgerVersionConflict(Exception):
    """A CAS ledger write lost the optimistic-concurrency race.

    Raised by ``store_ledger`` when the definitive store holds a newer version
    than the writer's cached one. The store NEVER blind-overwrites — the caller
    must re-fetch the current ledger, re-apply its mutation, and retry. This is
    what keeps a horizontally-scaled fleet from clobbering each other's balance
    writes (see ``LedgerCache.mutate``).
    """


class LedgerUnavailableError(Exception):
    """The definitive ledger store could not be read before a mutation.

    Raised by ``LedgerCache.mutate`` so a cold/unreachable store can never cause
    a mutation to be applied to an empty fallback ledger and written back
    (which would zero a real balance).
    """


class LedgerWriteError(Exception):
    """A ledger mutation exhausted its conflict retries without persisting."""


@runtime_checkable
class VaultBackend(Protocol):
    """Async persistence backend for user ledger data.

    Any object implementing these three methods can serve as the
    durable backing store for LedgerCache.
    """

    async def store_ledger(self, user_id: str, ledger_json: str) -> str: ...

    async def fetch_ledger(self, user_id: str) -> str | None: ...

    async def snapshot_ledger(
        self, user_id: str, ledger_json: str, timestamp: str
    ) -> str | None: ...
