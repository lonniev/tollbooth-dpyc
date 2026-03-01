"""Reusable Secure Courier Service — operators import and wire 3 MCP tools.

Wraps ``NostrCredentialExchange`` + ``CredentialVaultBackend`` + profile
publishing into a single high-level service with an operator callback hook
for post-receive logic.

Typical operator wiring::

    from tollbooth.secure_courier import SecureCourierService

    courier = SecureCourierService(
        operator_nsec=settings.nsec,
        relays=["wss://relay.primal.net"],
        templates={"x": my_template},
        credential_vault=my_vault,
        profile=NostrProfile(name="my-mcp"),
        on_credentials_received=my_callback,
    )

    # Wire 3 MCP tools:
    result = await courier.open_channel("x", recipient_npub=npub)
    result = await courier.receive(sender_npub, service="x")
    result = await courier.forget(sender_npub, service="x")

Dependencies are optional — install with ``pip install tollbooth-dpyc[nostr]``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation mirrors nostr_credentials.py
try:
    from tollbooth.credential_templates import CredentialTemplate
    from tollbooth.credential_vault_backend import CredentialVaultBackend
    from tollbooth.nostr_credentials import (
        CourierError,
        CourierNotReady,
        NostrCredentialExchange,
        NostrProfile,
    )

    _HAS_COURIER = True
except ImportError:
    _HAS_COURIER = False


class SecureCourierService:
    """Reusable Secure Courier -- operators import and wire 3 MCP tools.

    Handles the full credential exchange lifecycle:

    - Channel opening with welcome DM
    - Credential receiving (relay polling, decryption, validation, vault storage)
    - Success/error DM replies
    - Credential forgetting (vault deletion + key rotation support)

    Operator-specific logic (session activation, identity mapping, seed balance)
    is handled via the ``on_credentials_received`` callback.
    """

    def __init__(
        self,
        *,
        operator_nsec: str,
        relays: list[str],
        templates: dict[str, CredentialTemplate],
        credential_vault: CredentialVaultBackend | None = None,
        profile: NostrProfile | None = None,
        on_credentials_received: (
            Callable[[str, dict[str, str], str], Awaitable[dict[str, Any] | None]]
            | None
        ) = None,
    ) -> None:
        """Initialise the Secure Courier Service.

        Args:
            operator_nsec: Nostr secret key (bech32 nsec) for DM encryption.
            relays: List of relay WebSocket URLs.
            templates: ``{service_name: CredentialTemplate}`` mapping.
            credential_vault: Optional vault for cross-session persistence.
            profile: Optional Nostr kind 0 profile to publish on init.
            on_credentials_received: Async callback ``(sender_npub, credentials,
                service) -> dict | None``.  Called after successful credential
                receipt.  The returned dict (if any) is merged into the tool
                result.  ``credentials`` contains the validated fields.
                The callback handles operator-specific actions like session
                activation and identity mapping.
        """
        if not _HAS_COURIER:
            raise RuntimeError(
                "Secure Courier dependencies not installed. "
                "Install with: pip install tollbooth-dpyc[nostr]"
            )

        self._exchange = NostrCredentialExchange(
            nsec=operator_nsec,
            relays=relays,
            templates=templates,
            credential_vault=credential_vault,
        )

        self._on_received = on_credentials_received

        # Publish operator profile (kind 0) so patrons see a friendly
        # name and avatar instead of a raw npub string.
        if profile is not None:
            self._exchange.publish_profile(profile)

    # -- Public properties --------------------------------------------------

    @property
    def exchange(self) -> NostrCredentialExchange:
        """The underlying ``NostrCredentialExchange`` instance."""
        return self._exchange

    @property
    def enabled(self) -> bool:
        """Whether the underlying exchange is active and ready."""
        return self._exchange.enabled

    @property
    def npub(self) -> str:
        """Operator's npub (bech32)."""
        return self._exchange.npub

    @property
    def relays(self) -> list[str]:
        """Configured relay URLs."""
        return self._exchange.relays

    # -- Tool-level methods -------------------------------------------------

    async def open_channel(
        self,
        service: str = "x",
        *,
        greeting: str,
        recipient_npub: str | None = None,
    ) -> dict[str, Any]:
        """Open a Secure Courier channel.  Sends welcome DM if *recipient_npub* provided.

        Delegates directly to the underlying exchange.

        Args:
            service: Template service name (e.g. ``"x"``).
            greeting: Operator-provided banner shown at the top of the
                welcome DM.
            recipient_npub: Optional patron npub to send welcome DM to.

        Returns:
            Dict with npub, relays, template instructions, and freshness window.
        """
        return await self._exchange.open_channel(
            service, greeting=greeting, recipient_npub=recipient_npub,
        )

    async def receive(
        self,
        sender_npub: str,
        service: str = "x",
    ) -> dict[str, Any]:
        """Receive credentials from Secure Courier.

        1. Delegates to the exchange for relay polling / vault lookup.
        2. On success, calls ``on_credentials_received`` if set.
        3. Merges callback result into the response.
        4. **Always** strips credential values before returning.

        Args:
            sender_npub: Patron's npub (bech32).
            service: Credential template service name.

        Returns:
            Dict with success, service, field count, and any callback-added
            metadata.  **NEVER** returns credential values.
        """
        result = await self._exchange.receive(sender_npub, service=service)

        if result.get("success") and result.get("credentials"):
            credentials: dict[str, str] = result["credentials"]
            matched_service: str = result.get("service", service)

            # Invoke operator callback
            if self._on_received is not None:
                try:
                    extra = await self._on_received(
                        sender_npub, credentials, matched_service,
                    )
                    if extra and isinstance(extra, dict):
                        result.update(extra)
                except Exception as exc:
                    logger.warning(
                        "on_credentials_received callback failed: %s", exc,
                    )
                    result["callback_error"] = str(exc)

            # NEVER echo credential values
            result.pop("credentials", None)

        return result

    async def forget(
        self,
        sender_npub: str,
        service: str = "x",
    ) -> dict[str, Any]:
        """Delete vaulted credentials for key rotation.

        Args:
            sender_npub: Patron's npub (bech32).
            service: Service name whose credentials to forget.

        Returns:
            Dict with success and whether credentials were found.
        """
        return await self._exchange.forget(sender_npub, service=service)
