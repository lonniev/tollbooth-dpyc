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
    from tollbooth.credential_vault_backend import (
        CredentialVaultBackend,
        SessionBindingBackend,
    )
    from tollbooth.nostr_credentials import (
        CourierError,  # noqa: F401
        CourierNotReady,  # noqa: F401
        NostrCredentialExchange,
        NostrProfile,
    )

    _HAS_COURIER = True
except ImportError:
    _HAS_COURIER = False

try:
    from tollbooth.credential_card import (
        CredentialCardError,  # noqa: F401
        encode_credential_card,
        render_qr,
    )

    _HAS_CREDENTIAL_CARD = True
except ImportError:
    _HAS_CREDENTIAL_CARD = False

try:
    from tollbooth.identity_credential import sign_identity_credential

    _HAS_IDENTITY_CREDENTIAL = True
except ImportError:
    _HAS_IDENTITY_CREDENTIAL = False


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
        send_card_dm: bool = True,
        credential_ttl_seconds: int = 30 * 24 * 3600,
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
            send_card_dm: If True (default), send the credential card as a
                Nostr DM to the patron after first-time credential receipt.
                The DM contains the ``ncred1...`` string so the patron can
                save it for scan-and-paste reuse.  Only sent on fresh relay
                delivery, not on vault restores.
            credential_ttl_seconds: TTL for issued identity credentials
                (default 30 days).
        """
        if not _HAS_COURIER:
            raise RuntimeError(
                "Secure Courier dependencies not installed. "
                "Install with: pip install tollbooth-dpyc[nostr]"
            )

        self._operator_nsec = operator_nsec
        self._send_card_dm = send_card_dm
        self._credential_ttl = credential_ttl_seconds
        self._sessions: dict[str, str] = {}  # caller_id → npub (in-memory)
        self._credentials: dict[str, str] = {}  # npub → signed credential JSON

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

    def get_identity_credential(self, npub: str) -> str | None:
        """Return the signed identity credential for *npub*, or None."""
        return self._credentials.get(npub)

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
        *,
        caller_id: str | None = None,
    ) -> dict[str, Any]:
        """Receive credentials from Secure Courier.

        1. Delegates to the exchange for relay polling / vault lookup.
        2. On success, calls ``on_credentials_received`` if set.
        3. Merges callback result into the response.
        4. If *caller_id* provided, persists a session binding for cold-start
           restoration.
        5. **Always** strips credential values before returning.

        Args:
            sender_npub: Patron's npub (bech32).
            service: Credential template service name.
            caller_id: Optional transport-layer caller ID (e.g. Horizon
                user_id).  When provided and the vault supports session
                bindings, the ``(caller_id, service) → npub`` mapping is
                persisted for silent cold-start restoration.

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

            # Issue identity credential (kind 30080) for this citizen
            if _HAS_IDENTITY_CREDENTIAL:
                try:
                    credential = sign_identity_credential(
                        citizen_npub=sender_npub,
                        operator_nsec=self._operator_nsec,
                        ttl_seconds=self._credential_ttl,
                    )
                    self._credentials[sender_npub] = credential
                    result["identity_credential_issued"] = True
                except Exception as exc:
                    logger.warning(
                        "Identity credential issuance failed: %s", exc,
                    )

            # Persist session binding for cold-start restoration
            if caller_id is not None:
                self._sessions[caller_id] = sender_npub
                await self._store_binding(caller_id, matched_service, sender_npub)

            # Generate credential card for scan-and-paste reuse
            if _HAS_CREDENTIAL_CARD:
                try:
                    ncred, qr_png = self.generate_credential_card(
                        sender_npub, matched_service, credentials,
                    )
                    result["credential_card"] = ncred
                    if qr_png is not None:
                        import base64

                        result["credential_card_qr_base64"] = base64.b64encode(
                            qr_png
                        ).decode()

                    # Send credential card via Nostr DM on fresh relay receipt
                    is_vault_restore = result.get("encryption") == "vault"
                    if self._send_card_dm and not is_vault_restore:
                        await self._deliver_card_dm(
                            sender_npub, matched_service, ncred, qr_png=qr_png,
                        )
                        result["credential_card_dm_sent"] = True

                except Exception as exc:
                    logger.warning("Credential card generation failed: %s", exc)

            # NEVER echo credential values
            result.pop("credentials", None)

        return result

    async def forget(
        self,
        sender_npub: str,
        service: str = "x",
        *,
        caller_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete vaulted credentials for key rotation.

        Args:
            sender_npub: Patron's npub (bech32).
            service: Service name whose credentials to forget.
            caller_id: Optional transport-layer caller ID.  When provided,
                also deletes the session binding so cold-start restoration
                won't revive a forgotten identity.

        Returns:
            Dict with success and whether credentials were found.
        """
        result = await self._exchange.forget(sender_npub, service=service)

        if caller_id is not None:
            self._sessions.pop(caller_id, None)
            await self._delete_binding(caller_id, service)

        return result

    async def restore_session(
        self,
        caller_id: str,
        service: str = "x",
    ) -> str | None:
        """Attempt silent session restoration from a persisted binding.

        On serverless cold starts the in-memory session dict is lost.
        This method:

        1. Looks up the persisted ``(caller_id, service) → npub`` binding.
        2. Calls ``self.receive(npub, service, caller_id=caller_id)`` which
           hits the vault-first fast path (no relay I/O).
        3. The ``on_credentials_received`` callback fires, repopulating
           the in-memory session.

        Returns:
            The restored npub on success, ``None`` on any failure.
        """
        vault = self._exchange._credential_vault
        if vault is None or not isinstance(vault, SessionBindingBackend):
            return None

        try:
            npub = await vault.fetch_session_binding(caller_id, service)
        except Exception as exc:
            logger.debug("Session binding lookup failed: %s", exc)
            return None

        if npub is None:
            return None

        try:
            result = await self.receive(npub, service, caller_id=caller_id)
            if result.get("success"):
                return npub
        except Exception as exc:
            logger.debug("Session restoration failed: %s", exc)

        return None

    async def ensure_identity(
        self,
        caller_id: str,
        service: str = "x",
    ) -> str:
        """Return the npub for *caller_id*, auto-restoring on cold start.

        Every operator MCP server should call this instead of maintaining
        its own in-memory session dict.

        Fast path: in-memory ``_sessions`` cache hit → return immediately.
        Cold-start path: ``restore_session()`` → vault binding lookup →
        vault-first ``receive()`` → ``on_credentials_received`` callback →
        operator session re-established.

        Raises:
            ValueError: No identity could be established (first-time user
                or credentials were forgotten).
        """
        npub = self._sessions.get(caller_id)
        if npub:
            return npub

        restored = await self.restore_session(caller_id, service)
        if restored:
            return restored

        raise ValueError(
            "No DPYC identity active. Credit operations require an npub. "
            "Follow the Secure Courier onboarding flow: call "
            "request_credential_channel(recipient_npub=<npub>), reply via Nostr DM, "
            "then call receive_credentials(sender_npub=<npub>). "
            "Get your npub from the dpyc-oracle's how_to_join() tool."
        )

    # -- Private helpers -----------------------------------------------------

    async def _store_binding(
        self, caller_id: str, service: str, npub: str,
    ) -> None:
        """Persist a session binding if the vault supports it."""
        vault = self._exchange._credential_vault
        if vault is not None and isinstance(vault, SessionBindingBackend):
            try:
                await vault.store_session_binding(caller_id, service, npub)
            except Exception as exc:
                logger.warning("Failed to store session binding: %s", exc)

    async def _delete_binding(
        self, caller_id: str, service: str,
    ) -> None:
        """Delete a session binding if the vault supports it."""
        vault = self._exchange._credential_vault
        if vault is not None and isinstance(vault, SessionBindingBackend):
            try:
                await vault.delete_session_binding(caller_id, service)
            except Exception as exc:
                logger.warning("Failed to delete session binding: %s", exc)

    # -- Credential card -----------------------------------------------------

    def generate_credential_card(
        self,
        sender_npub: str,
        service: str,
        credentials: dict[str, str],
        *,
        expiry_days: int = 90,
    ) -> tuple[str, bytes | None]:
        """Generate an ``ncred1...`` credential card with optional QR PNG.

        Args:
            sender_npub: Patron's npub (bech32).
            service: Service name for the card.
            credentials: Validated credential dict.
            expiry_days: Card validity in days.

        Returns:
            Tuple of (ncred_string, qr_png_bytes_or_None).
        """
        if not _HAS_CREDENTIAL_CARD:
            raise RuntimeError("credential_card module not available")

        ncred = encode_credential_card(
            credentials, service, self._operator_nsec, sender_npub,
            expiry_days=expiry_days,
        )

        qr_png: bytes | None = None
        try:
            qr_png = render_qr(ncred)
        except Exception as exc:
            logger.debug("QR rendering skipped: %s", exc)

        return ncred, qr_png

    async def _deliver_card_dm(
        self,
        recipient_npub: str,
        service: str,
        ncred: str,
        *,
        qr_png: bytes | None = None,
    ) -> None:
        """Send the credential card string to the patron via Nostr DM.

        Runs in a fire-and-forget manner — DM delivery failures are
        logged but never propagated to the caller.

        When *qr_png* is provided, the QR image is uploaded to
        nostr.build (NIP-96) and the resulting URL is included in the
        DM so Nostr clients auto-render the image inline.

        Args:
            recipient_npub: Patron's npub (bech32).
            service: Service name (for the human-readable header).
            ncred: The ``ncred1...`` credential card string.
            qr_png: Optional PNG bytes of the credential card QR code.
        """
        image_url: str | None = None
        if qr_png is not None:
            try:
                from tollbooth.media_upload import upload_to_nostr_build

                image_url = await upload_to_nostr_build(qr_png)
            except Exception as exc:
                logger.warning(
                    "QR upload to nostr.build failed (non-fatal): %s", exc,
                )

        if image_url:
            dm_text = (
                f"Your credential card for {service}:\n\n"
                f"{image_url}\n\n"
                f"{ncred}\n\n"
                "Save this string. To reactivate your session later, "
                "paste it into the credential_card parameter — "
                "no Courier exchange needed."
            )
        else:
            dm_text = (
                f"Your credential card for {service}:\n\n"
                f"{ncred}\n\n"
                "Save this string. To reactivate your session later, "
                "paste it into the credential_card parameter — "
                "no Courier exchange needed."
            )

        try:
            self._exchange.send_dm(recipient_npub, dm_text)
            logger.info(
                "Credential card DM sent to %s for service %s",
                recipient_npub[:16] + "...",
                service,
            )
        except Exception as exc:
            logger.warning(
                "Credential card DM delivery failed (non-fatal): %s", exc,
            )

    async def redeem_card(
        self,
        ncred: str,
        service: str | None = None,
    ) -> dict[str, Any]:
        """Redeem an ``ncred1...`` credential card — bypasses relay DM flow.

        1. Delegates to the exchange for decode/verify/decrypt.
        2. On success, calls ``on_credentials_received`` if set.
        3. Merges callback result into the response.
        4. **Always** strips credential values before returning.

        Args:
            ncred: The ``ncred1...`` bech32 credential card string.
            service: Optional service hint for template matching.

        Returns:
            Dict with success, service, field count, and any callback-added
            metadata.  **NEVER** returns credential values.
        """
        result = await self._exchange.redeem_credential_card(
            ncred, service=service,
        )

        if result.get("success") and result.get("credentials"):
            credentials: dict[str, str] = result["credentials"]
            matched_service: str = result.get("service", service or "")

            # Invoke operator callback (same pattern as receive())
            if self._on_received is not None:
                try:
                    extra = await self._on_received(
                        result.get("user_npub", ""),
                        credentials,
                        matched_service,
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
