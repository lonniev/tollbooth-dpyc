"""PatronSessionCache — in-memory session cache with Neon vault persistence.

Wraps ``SessionCache[T]`` with automatic vault restore on cache miss and
vault persist on session store. Operators provide a ``restore`` callback
that constructs a domain-specific session object from vault credentials.

Usage::

    from tollbooth.patron_session import PatronSessionCache

    async def _restore(creds: dict) -> MySession | None:
        if "api_key" not in creds:
            return None
        return MySession(api_key=creds["api_key"], ...)

    sessions = PatronSessionCache[MySession](
        runtime=runtime,
        service="my-service",
        restore=_restore,
    )

    # Get or restore session (checks memory, then vault)
    session = await sessions.get_or_restore(npub)

    # Store session (memory + vault)
    await sessions.store(npub, session, vault_data={"api_key": "..."})

    # Invalidate (memory only — vault cleanup via on_forget callback)
    sessions.invalidate(npub)
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Generic, TypeVar

from tollbooth.session_cache import SessionCache

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PatronSessionCache(Generic[T]):
    """Session cache with Neon vault persistence.

    Type parameter T is the operator's session object.

    Args:
        runtime: The ``OperatorRuntime`` instance.
        service: Service name for vault namespacing (e.g., ``"schwab"``).
        restore: Async callback ``(creds: dict) -> T | None`` that
            constructs a session from vault credentials. Return None
            if the credentials are insufficient.
        ttl_seconds: In-memory TTL (default 3600).
    """

    def __init__(
        self,
        runtime: Any,
        service: str,
        restore: Callable[[dict[str, Any]], Awaitable[T | None]],
        ttl_seconds: int = 3600,
    ) -> None:
        self._runtime = runtime
        self._service = service
        self._restore = restore
        self._cache: SessionCache[T] = SessionCache(ttl_seconds=ttl_seconds)

    async def get_or_restore(self, npub: str) -> T | None:
        """Get session by npub from memory, falling back to Neon vault restore."""
        if not npub:
            return None

        session = self._cache.get(npub)
        if session is not None:
            return session

        try:
            creds = await self._runtime.load_patron_session(
                npub, service=self._service,
            )
            if not creds:
                return None

            session = await self._restore(creds)
            if session is None:
                return None

            self._cache.set(npub, session)
            logger.info(
                "Restored %s session for %s from vault.",
                self._service, npub[:20],
            )
            return session
        except Exception as exc:
            logger.warning("Vault session restore failed: %s", exc)
            return None

    async def store(
        self,
        npub: str,
        session: T,
        vault_data: dict[str, Any],
    ) -> T:
        """Store session in memory and persist credentials to Neon vault."""
        self._cache.set(npub, session)
        await self._runtime.store_patron_session(
            npub, vault_data, service=self._service,
        )
        return session

    def get(self, npub: str) -> T | None:
        """Get session from memory only (no vault restore)."""
        return self._cache.get(npub)

    def set_local(self, npub: str, session: T) -> T:
        """Store session in memory only (no vault persist)."""
        self._cache.set(npub, session)
        return session

    def invalidate(self, npub: str) -> None:
        """Remove session from memory."""
        self._cache.clear(npub)

    def clear_local(self, npub: str) -> None:
        """Remove session from memory only (e.g., on forget callback)."""
        self._cache.clear(npub)

    @property
    def cache(self) -> SessionCache[T]:
        """Access the underlying SessionCache directly."""
        return self._cache
