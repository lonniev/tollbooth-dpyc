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
    session = await sessions.get_or_restore(user_id, npub)

    # Store session (memory + vault)
    await sessions.store(user_id, npub, session, vault_data={"api_key": "..."})

    # Invalidate (memory + vault)
    await sessions.invalidate(user_id, npub)
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
        self._npub_for_user: dict[str, str] = {}

    async def get_or_restore(
        self,
        user_id: str,
        npub: str = "",
    ) -> T | None:
        """Get session from memory, falling back to Neon vault restore.

        Returns None if no session exists in either location.
        """
        # Check memory first
        session = self._cache.get(user_id)
        if session is not None:
            return session

        # Resolve npub for vault lookup
        patron_npub = self._npub_for_user.get(user_id) or npub
        if not patron_npub:
            return None

        # Try vault restore
        try:
            creds = await self._runtime.load_patron_session(
                patron_npub, service=self._service,
            )
            if not creds:
                return None

            session = await self._restore(creds)
            if session is None:
                return None

            self._cache.set(user_id, session)
            self._npub_for_user[user_id] = patron_npub
            logger.info(
                "Restored %s session for %s from vault.",
                self._service, patron_npub[:20],
            )
            return session
        except Exception as exc:
            logger.warning("Vault session restore failed: %s", exc)
            return None

    async def store(
        self,
        user_id: str,
        npub: str,
        session: T,
        vault_data: dict[str, Any],
    ) -> T:
        """Store session in memory and persist credentials to Neon vault.

        Args:
            user_id: Horizon user ID.
            npub: Patron's Nostr public key.
            session: The domain session object to cache.
            vault_data: Dict of credentials to persist in vault.
        """
        self._cache.set(user_id, session)
        self._npub_for_user[user_id] = npub
        await self._runtime.store_patron_session(
            npub, vault_data, service=self._service,
        )
        return session

    def get(self, user_id: str) -> T | None:
        """Get session from memory only (no vault restore)."""
        return self._cache.get(user_id)

    def set_local(self, user_id: str, session: T, npub: str = "") -> T:
        """Store session in memory only (no vault persist)."""
        self._cache.set(user_id, session)
        if npub:
            self._npub_for_user[user_id] = npub
        return session

    def invalidate(self, user_id: str) -> None:
        """Remove session from memory.

        Vault cleanup is handled by the ``on_forget`` callback registered
        with OperatorRuntime. Call this from your ``on_forget`` handler.
        """
        self._cache.clear(user_id)
        self._npub_for_user.pop(user_id, None)

    def clear_local(self, user_id: str) -> None:
        """Remove session from memory only (e.g., on forget callback)."""
        self._cache.clear(user_id)
        self._npub_for_user.pop(user_id, None)

    @property
    def cache(self) -> SessionCache[T]:
        """Access the underlying SessionCache directly."""
        return self._cache
