"""Abstract persistence interface for commerce state (ledger storage).

Defines the VaultBackend Protocol that LedgerCache depends on. Concrete
implementations (NeonVault, PersonalBrainVault) live elsewhere.

The protocol is optimistic concurrency with the version bound to the snapshot:
``fetch_ledger`` hands back the row's version with its JSON, and
``store_ledger`` lands only if the row still holds that exact version. A writer
can never pass the check with a version it did not read alongside the state
it is writing — which is the property a per-process version cache broke on
2026-09-27, when a stale snapshot flushed under a fresher writer's version and
erased a settled credit.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class LedgerVersionConflict(Exception):
    """A CAS ledger write lost the optimistic-concurrency race.

    Raised by ``store_ledger`` when the definitive store no longer holds the
    version the writer read its snapshot at. The store NEVER blind-overwrites —
    the caller re-fetches the current ledger, re-applies its mutation, and
    retries. This is what keeps a horizontally-scaled fleet, and the coroutines
    of one process, from clobbering each other's balance writes (see
    ``LedgerCache.mutate``).
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

    ``fetch_ledger`` returns ``(ledger_json, version)`` or ``None`` when the
    user has no row. ``store_ledger`` writes ``ledger_json`` only if the row's
    version is still ``expected_version`` (``None`` means "there is no row yet":
    insert, and conflict if one appeared) and returns the new version.
    ``snapshot_ledger`` appends a timestamped copy to the journal; it never
    touches the live row.
    """

    async def fetch_ledger(self, user_id: str) -> tuple[str, int] | None: ...

    async def store_ledger(
        self, user_id: str, ledger_json: str, expected_version: int | None,
    ) -> int: ...

    async def snapshot_ledger(
        self, user_id: str, ledger_json: str, timestamp: str
    ) -> str | None: ...
