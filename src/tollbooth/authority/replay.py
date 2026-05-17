"""In-memory anti-replay tracker using OrderedDict + TTL."""

from __future__ import annotations

import time
from collections import OrderedDict


class ReplayTracker:
    """Tracks seen JTI values to prevent certificate replay.

    Defence-in-depth: the certificate ``expiration`` tag is the primary
    expiry mechanism.  This tracker provides an additional layer of
    protection against replay within the TTL window.
    """

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl = ttl_seconds
        self._seen: OrderedDict[str, float] = OrderedDict()

    def check_and_record(self, jti: str) -> bool:
        """Return True if *jti* is new (accepted), False if replayed.

        Automatically prunes expired entries on each call.
        """
        self._prune()
        if jti in self._seen:
            return False
        self._seen[jti] = time.monotonic()
        return True

    @property
    def size(self) -> int:
        return len(self._seen)

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._ttl
        while self._seen:
            oldest_jti, oldest_ts = next(iter(self._seen.items()))
            if oldest_ts > cutoff:
                break
            self._seen.pop(oldest_jti)
