"""Nostr DM credential exchange — the Secure Courier Service.

Out-of-band credential delivery via NIP-44 (or NIP-04 fallback) encrypted
Nostr DMs.  Patrons send credentials from their Nostr client to the MCP
operator's npub; the operator picks them up, validates, stores, and
destroys the relay copy.

Two-tool operator interface:

    exchange = NostrCredentialExchange(nsec, relays, templates)

    # Tool 1: Tell patron where to send
    await exchange.open_channel("x")

    # Tool 2: Pick up the sealed pouch
    await exchange.receive("npub1user...")

Dependencies are optional — install with ``pip install tollbooth-dpyc[nostr]``.

Reference: https://github.com/nostr-protocol/nips/blob/master/44.md (NIP-44)
           https://github.com/nostr-protocol/nips/blob/master/04.md (NIP-04)
           https://github.com/nostr-protocol/nips/blob/master/09.md (NIP-09)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from tollbooth.credential_templates import (
    CredentialTemplate,
    FieldSpec,
    TemplateValidationError,
    render_template_instructions,
    validate_payload,
)
from tollbooth.credential_vault_backend import CredentialVaultBackend

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation
try:
    from pynostr.event import Event  # type: ignore[import-untyped]
    from pynostr.key import PrivateKey, PublicKey  # type: ignore[import-untyped]

    _HAS_PYNOSTR = True
except ImportError:
    _HAS_PYNOSTR = False

try:
    from websocket import create_connection  # type: ignore[import-untyped]

    _HAS_WEBSOCKET = True
except ImportError:
    _HAS_WEBSOCKET = False

try:
    from tollbooth.nip44 import decrypt as _nip44_decrypt

    _HAS_NIP44 = True
except ImportError:
    _HAS_NIP44 = False

try:
    from tollbooth.nip04 import decrypt as _nip04_decrypt

    _HAS_NIP04 = True
except ImportError:
    _HAS_NIP04 = False


# Nostr event kinds
_KIND_ENCRYPTED_DM = 4  # NIP-04 legacy DMs
_KIND_GIFT_WRAP = 1059  # NIP-17/NIP-59 gift-wrapped DMs
_KIND_SEAL = 13  # NIP-59 seal (inner layer of gift wrap)
_KIND_PRIVATE_DM = 14  # NIP-17 private DM (innermost content)
_KIND_DELETION = 5  # NIP-09 event deletion

# Default freshness window (seconds)
_DEFAULT_FRESHNESS = 600  # 10 minutes
_DEFAULT_SUBSCRIBE_TIMEOUT = 10


def _npub_to_hex(npub: str) -> str:
    """Convert npub bech32 to 32-byte hex pubkey."""
    if not _HAS_PYNOSTR:
        raise RuntimeError("pynostr required for npub conversion")
    return PublicKey.from_npub(npub).hex()


def _hex_to_npub(hex_pubkey: str) -> str:
    """Convert 32-byte hex pubkey to npub bech32."""
    if not _HAS_PYNOSTR:
        raise RuntimeError("pynostr required for npub conversion")
    return PublicKey(bytes.fromhex(hex_pubkey)).bech32()


class CourierError(Exception):
    """Base error for credential exchange operations."""


class CourierNotReady(CourierError):
    """Raised when dependencies are missing or exchange is not configured."""


class CourierTimeout(CourierError):
    """Raised when no matching DM is found within the freshness window."""


class CourierValidationError(CourierError):
    """Raised when credential payload fails template validation."""


class NostrCredentialExchange:
    """Secure Courier Service — out-of-band credential delivery via Nostr.

    Operators instantiate with their nsec, relay list, and credential
    templates.  Exposes two async methods for the MCP tool layer:
    ``open_channel()`` and ``receive()``.

    All relay I/O runs in background threads (sync websocket-client)
    to avoid blocking the async MCP event loop.
    """

    def __init__(
        self,
        nsec: str,
        relays: list[str],
        templates: dict[str, CredentialTemplate],
        *,
        freshness_window: int = _DEFAULT_FRESHNESS,
        nip44_only: bool = False,
        credential_vault: CredentialVaultBackend | None = None,
    ) -> None:
        """Initialize the credential exchange.

        Args:
            nsec: Operator's Nostr nsec (bech32).
            relays: List of relay WebSocket URLs.
            templates: Map of service name → CredentialTemplate.
            freshness_window: Max age in seconds for accepted DMs.
            nip44_only: If True, reject NIP-04 DMs entirely.
            credential_vault: Optional vault for persisting credentials
                across sessions.  If provided, ``receive()`` checks the
                vault first and stores credentials after successful pickup.
        """
        self._freshness_window = freshness_window
        self._nip44_only = nip44_only
        self._templates = templates
        self._relays = [r.strip() for r in relays if r.strip()]
        self._credential_vault = credential_vault

        # Key material
        self._privkey_hex: str = ""
        self._pubkey_hex: str = ""
        self._npub: str = ""
        self._enabled = True

        if not _HAS_PYNOSTR:
            logger.warning(
                "pynostr not installed — Secure Courier disabled. "
                "Install with: pip install tollbooth-dpyc[nostr]"
            )
            self._enabled = False
            return

        if not _HAS_WEBSOCKET:
            logger.warning(
                "websocket-client not installed — Secure Courier disabled. "
                "Install with: pip install tollbooth-dpyc[nostr]"
            )
            self._enabled = False
            return

        if not self._relays:
            logger.warning("No Nostr relays configured — Secure Courier disabled.")
            self._enabled = False
            return

        try:
            pk = PrivateKey.from_nsec(nsec)
            self._privkey_hex = pk.hex()
            self._pubkey_hex = pk.public_key.hex()
            self._npub = pk.public_key.bech32()
        except Exception as exc:
            logger.warning("Invalid operator nsec — Secure Courier disabled: %s", exc)
            self._enabled = False

        # Thread-safe buffer for received events
        self._lock = threading.Lock()
        self._received_events: list[dict[str, Any]] = []
        # Track consumed event IDs (prevent double-pickup)
        self._consumed_ids: set[str] = set()

    @property
    def enabled(self) -> bool:
        """Whether the exchange is active and ready."""
        return self._enabled

    @property
    def npub(self) -> str:
        """Operator's npub (bech32)."""
        return self._npub

    @property
    def relays(self) -> list[str]:
        """Configured relay URLs."""
        return list(self._relays)

    async def open_channel(self, service: str) -> dict[str, Any]:
        """Open a credential delivery channel for a service.

        Returns the operator's npub, relay list, and template instructions
        so the patron knows where and what to send.  Also starts a
        background relay subscription to listen for incoming DMs.

        Args:
            service: Template service name (e.g., "x", "openai").

        Returns:
            Dict with npub, relays, template instructions, and freshness window.
        """
        if not self._enabled:
            raise CourierNotReady(
                "Secure Courier not available. Check logs for missing dependencies."
            )

        if service not in self._templates:
            available = ", ".join(sorted(self._templates.keys()))
            raise CourierValidationError(
                f"Unknown service '{service}'. Available: {available}"
            )

        template = self._templates[service]
        instructions = render_template_instructions(template)

        # Start background subscription
        self._start_subscription()

        return {
            "success": True,
            "npub": self._npub,
            "relays": self._relays,
            "service": service,
            "freshness_window_seconds": self._freshness_window,
            "instructions": instructions,
            "message": (
                f"Send your {template.service} credentials as a Nostr DM "
                f"to {self._npub} from your Nostr client. "
                f"The DM must contain a JSON object matching the template above. "
                f"Then call receive_credentials with your npub."
            ),
        }

    async def receive(
        self, sender_npub: str, *, service: str | None = None,
    ) -> dict[str, Any]:
        """Pick up and validate credentials from a sender.

        If a credential vault is configured, checks it first.  On a vault
        hit the credentials are returned immediately without any relay I/O.
        On a miss (or no vault), falls back to the relay DM flow.

        After a successful relay pickup the validated credentials are
        encrypted and stored in the vault for future sessions.

        Args:
            sender_npub: Patron's npub (bech32).
            service: Optional service hint for vault lookup.  If omitted
                and only one template is configured, that service is used.

        Returns:
            Dict with success, service name, and field count.
            The actual credential values are returned under a
            ``credentials`` key for the caller to store.
        """
        if not self._enabled:
            raise CourierNotReady(
                "Secure Courier not available. Check logs for missing dependencies."
            )

        try:
            sender_hex = _npub_to_hex(sender_npub)
        except Exception as exc:
            raise CourierValidationError(f"Invalid sender npub: {exc}") from exc

        # Resolve service for vault lookup
        vault_service = self._resolve_service(service)

        # ── Vault-first lookup ──────────────────────────────────────
        if self._credential_vault is not None and vault_service:
            vaulted = await self._vault_fetch(vault_service, sender_npub)
            if vaulted is not None:
                template = self._templates[vault_service]
                sensitive_count = sum(
                    1 for name in vaulted
                    if template.fields.get(name, FieldSpec()).sensitive
                )
                return {
                    "success": True,
                    "service": vault_service,
                    "fields_received": len(vaulted),
                    "sensitive_fields": sensitive_count,
                    "encryption": "vault",
                    "credentials": vaulted,
                    "message": (
                        f"Credentials for {vault_service} restored from vault "
                        f"({len(vaulted)} fields). No relay I/O needed."
                    ),
                }

        # ── Relay DM flow (existing) ────────────────────────────────
        dm = self._find_dm_in_buffer(sender_hex)
        if dm is None:
            self._fetch_dms_from_relays()
            dm = self._find_dm_in_buffer(sender_hex)

        if dm is None:
            raise CourierTimeout(
                f"No DM found from {sender_npub} within the "
                f"{self._freshness_window}-second freshness window. "
                f"Make sure you sent the DM from your Nostr client and "
                f"try again."
            )

        # Decrypt content
        plaintext = self._decrypt_dm(dm, sender_hex)

        # Parse JSON payload
        try:
            payload = json.loads(plaintext)
        except json.JSONDecodeError as exc:
            raise CourierValidationError(
                f"DM content is not valid JSON: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise CourierValidationError(
                "DM content must be a JSON object, "
                f"got {type(payload).__name__}"
            )

        # Match template
        payload_service = payload.get("service")
        template = self._match_template(payload_service, payload)

        # Validate against template
        try:
            validated = validate_payload(payload, template)
        except TemplateValidationError as exc:
            raise CourierValidationError(str(exc)) from exc

        # Mark event as consumed
        event_id = dm.get("id", "")
        with self._lock:
            self._consumed_ids.add(event_id)

        # Publish NIP-09 deletion request (fire-and-forget)
        if event_id:
            self._request_deletion(event_id)

        # Store in vault for next session
        if self._credential_vault is not None:
            await self._vault_store(template.service, sender_npub, validated)

        # Count sensitive vs non-sensitive fields
        sensitive_count = sum(
            1 for name in validated if template.fields.get(name, FieldSpec()).sensitive
        )

        return {
            "success": True,
            "service": template.service,
            "fields_received": len(validated),
            "sensitive_fields": sensitive_count,
            "encryption": dm.get("encryption", "unknown"),
            "credentials": validated,
            "message": (
                f"Credentials received for {template.service} "
                f"({len(validated)} fields). "
                f"Relay copy deletion requested."
                + (
                    " Credentials stored in vault for future sessions."
                    if self._credential_vault is not None else ""
                )
            ),
        }

    async def forget(
        self, sender_npub: str, *, service: str | None = None,
    ) -> dict[str, Any]:
        """Delete vaulted credentials for a patron.

        Useful when a patron rotates API keys and needs to re-deliver
        via the courier.

        Args:
            sender_npub: Patron's npub (bech32).
            service: Service name to forget.  If omitted and only one
                template is configured, that service is used.

        Returns:
            Dict with success and whether credentials were found.
        """
        if self._credential_vault is None:
            return {
                "success": False,
                "message": "No credential vault configured.",
            }

        resolved = self._resolve_service(service)
        if not resolved:
            return {
                "success": False,
                "message": "Cannot determine service. Provide a service name.",
            }

        deleted = await self._credential_vault.delete_credentials(
            resolved, sender_npub,
        )
        return {
            "success": True,
            "service": resolved,
            "deleted": deleted,
            "message": (
                f"Credentials for {resolved} from {sender_npub} "
                + ("deleted from vault." if deleted else "not found in vault.")
            ),
        }

    def _resolve_service(self, service: str | None) -> str | None:
        """Resolve service name, defaulting to the single template if only one."""
        if service and service in self._templates:
            return service
        if len(self._templates) == 1:
            return next(iter(self._templates.keys()))
        return service

    # ── Vault helpers ──────────────────────────────────────────────────

    async def _vault_fetch(
        self, service: str, sender_npub: str,
    ) -> dict[str, str] | None:
        """Fetch and decrypt credentials from the vault."""
        assert self._credential_vault is not None
        blob = await self._credential_vault.fetch_credentials(service, sender_npub)
        if blob is None:
            return None
        try:
            plaintext = self._vault_decrypt(blob)
            creds = json.loads(plaintext)
            if isinstance(creds, dict):
                return creds
            logger.warning("Vault blob decoded to non-dict, ignoring")
            return None
        except Exception as exc:
            logger.warning("Vault credential decryption failed: %s", exc)
            return None

    async def _vault_store(
        self, service: str, sender_npub: str, credentials: dict[str, str],
    ) -> None:
        """Encrypt and store credentials in the vault."""
        assert self._credential_vault is not None
        try:
            plaintext = json.dumps(credentials)
            blob = self._vault_encrypt(plaintext)
            await self._credential_vault.store_credentials(
                service, sender_npub, blob,
            )
            logger.info(
                "Credentials for %s/%s stored in vault", service, sender_npub[:16],
            )
        except Exception as exc:
            logger.warning("Vault credential storage failed: %s", exc)

    def _vault_encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext for vault storage using operator's key.

        Uses NIP-04 AES-256-CBC with the operator as both sender and
        recipient (self-encryption).
        """
        if not _HAS_NIP04:
            raise RuntimeError("NIP-04 module required for vault encryption")
        from tollbooth.nip04 import encrypt as _nip04_encrypt
        return _nip04_encrypt(plaintext, self._privkey_hex, self._pubkey_hex)

    def _vault_decrypt(self, blob: str) -> str:
        """Decrypt a vault blob using operator's key."""
        if not _HAS_NIP04:
            raise RuntimeError("NIP-04 module required for vault decryption")
        return _nip04_decrypt(blob, self._privkey_hex, self._pubkey_hex)

    # ── Template matching ───────────────────────────────────────────

    def _match_template(
        self, service: str | None, payload: dict,
    ) -> CredentialTemplate:
        """Find the matching template for a payload."""
        # Exact match by service name
        if service and service in self._templates:
            return self._templates[service]

        # Single-template shortcut: if only one template, use it
        if len(self._templates) == 1:
            return next(iter(self._templates.values()))

        # Try to match by field names
        payload_fields = {
            k for k in payload.keys() if k not in {"service", "version"}
        }
        for tmpl in self._templates.values():
            template_fields = set(tmpl.fields.keys())
            if payload_fields.issubset(template_fields):
                return tmpl

        available = ", ".join(sorted(self._templates.keys()))
        raise CourierValidationError(
            f"Cannot match payload to a template. "
            f"Include a 'service' field or available services: {available}"
        )

    def _start_subscription(self) -> None:
        """Start relay subscription and wait for it to complete.

        Runs synchronously so that ``open_channel()`` returns with the
        buffer already populated.  Relay timeouts keep this bounded.
        """
        self._subscribe_to_relays()

    def _subscribe_to_relays(self) -> None:
        """Subscribe to all relays for DMs addressed to us."""
        since = int(time.time()) - self._freshness_window
        # REQ filter: kind 4 (NIP-04) and kind 1059 (NIP-17 gift wrap)
        kinds = [_KIND_ENCRYPTED_DM]
        if not self._nip44_only:
            kinds.append(_KIND_GIFT_WRAP)
        else:
            # For NIP-44-only, still listen for gift wraps
            kinds = [_KIND_GIFT_WRAP]
            # Also listen for kind 4 in case sender used NIP-04
            # (we'll reject it during decrypt if nip44_only is set)
            kinds.append(_KIND_ENCRYPTED_DM)

        # NIP-04 DMs are tagged with p-tag for recipient
        # NIP-17 gift wraps are tagged with p-tag for recipient
        req_filter = {
            "kinds": kinds,
            "#p": [self._pubkey_hex],
            "since": since,
            "limit": 50,
        }
        sub_id = f"courier-{int(time.time())}"

        for relay_url in self._relays:
            try:
                self._subscribe_one_relay(relay_url, sub_id, req_filter)
            except Exception as exc:
                logger.debug(
                    "Relay subscription %s failed (non-fatal): %s",
                    relay_url, exc,
                )

    def _subscribe_one_relay(
        self,
        relay_url: str,
        sub_id: str,
        req_filter: dict[str, Any],
    ) -> None:
        """Subscribe to one relay and collect events."""
        sslopt: dict[str, Any] = {}
        ws = create_connection(relay_url, timeout=_DEFAULT_SUBSCRIBE_TIMEOUT, sslopt=sslopt)
        try:
            # Send REQ
            req_msg = json.dumps(["REQ", sub_id, req_filter])
            ws.send(req_msg)

            # Read events until EOSE or timeout
            ws.settimeout(_DEFAULT_SUBSCRIBE_TIMEOUT)
            while True:
                try:
                    raw = ws.recv()
                except Exception:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if not isinstance(msg, list) or len(msg) < 2:
                    continue

                msg_type = msg[0]

                if msg_type == "EVENT" and len(msg) >= 3:
                    event_data = msg[2]
                    if isinstance(event_data, dict):
                        event_data["_relay"] = relay_url
                        with self._lock:
                            self._received_events.append(event_data)

                elif msg_type == "EOSE":
                    break

                elif msg_type == "CLOSED" or msg_type == "NOTICE":
                    break

            # Send CLOSE
            close_msg = json.dumps(["CLOSE", sub_id])
            ws.send(close_msg)
        finally:
            ws.close()

    def _fetch_dms_from_relays(self) -> None:
        """One-shot fetch of recent DMs from all relays."""
        self._subscribe_to_relays()

    def _find_dm_in_buffer(self, sender_hex: str) -> dict[str, Any] | None:
        """Find the most recent unconsumed DM from a sender."""
        now = int(time.time())
        cutoff = now - self._freshness_window

        with self._lock:
            candidates = []
            for event in self._received_events:
                event_id = event.get("id", "")
                if event_id in self._consumed_ids:
                    continue

                created_at = event.get("created_at", 0)
                if created_at < cutoff:
                    continue

                kind = event.get("kind", 0)

                if kind == _KIND_ENCRYPTED_DM:
                    # NIP-04: sender is event.pubkey
                    if event.get("pubkey", "") == sender_hex:
                        candidates.append(event)

                elif kind == _KIND_GIFT_WRAP:
                    # NIP-17: sender is hidden inside the seal
                    # We can't filter by sender until we unwrap, so
                    # include all gift wraps as candidates
                    candidates.append(event)

            if not candidates:
                return None

            # Return the most recent
            candidates.sort(key=lambda e: e.get("created_at", 0), reverse=True)
            return candidates[0]

    def _decrypt_dm(
        self, event: dict[str, Any], sender_hex: str,
    ) -> str:
        """Decrypt a DM event (NIP-04 or NIP-17 gift wrap).

        Returns the plaintext content string.
        """
        kind = event.get("kind", 0)

        if kind == _KIND_GIFT_WRAP:
            return self._unwrap_gift_wrap(event, sender_hex)

        if kind == _KIND_ENCRYPTED_DM:
            return self._decrypt_nip04_dm(event, sender_hex)

        raise CourierValidationError(f"Unexpected event kind: {kind}")

    def _decrypt_nip04_dm(
        self, event: dict[str, Any], sender_hex: str,
    ) -> str:
        """Decrypt a NIP-04 kind 4 DM."""
        if self._nip44_only:
            raise CourierValidationError(
                "NIP-04 DMs rejected — this operator requires NIP-44 "
                "(NIP-17 gift wrap). Please use a modern Nostr client."
            )

        if not _HAS_NIP04:
            raise CourierValidationError(
                "NIP-04 decryption not available. Install cryptography."
            )

        content = event.get("content", "")
        try:
            plaintext = _nip04_decrypt(content, self._privkey_hex, sender_hex)
        except Exception as exc:
            raise CourierValidationError(
                f"NIP-04 decryption failed: {exc}"
            ) from exc

        event["encryption"] = "nip04"
        logger.info(
            "Received NIP-04 DM (legacy). Consider upgrading to a "
            "NIP-17/NIP-44 capable client."
        )
        return plaintext

    def _unwrap_gift_wrap(
        self, event: dict[str, Any], sender_hex: str,
    ) -> str:
        """Unwrap a NIP-17 gift wrap (kind 1059 → kind 13 → kind 14).

        NIP-17 uses three layers:
        1. Gift wrap (kind 1059): encrypted to recipient with random key
        2. Seal (kind 13): encrypted to recipient with sender's key
        3. Private DM (kind 14): plaintext content

        The gift wrap is encrypted with NIP-44 using a random one-time key,
        so we decrypt with our privkey + the gift wrap's pubkey.
        """
        if not _HAS_NIP44:
            raise CourierValidationError(
                "NIP-44 decryption not available for gift wrap unwrapping."
            )

        # Layer 1: Decrypt gift wrap content with our privkey + wrap pubkey
        wrap_pubkey = event.get("pubkey", "")
        wrap_content = event.get("content", "")

        try:
            seal_json = _nip44_decrypt(
                wrap_content, self._privkey_hex, wrap_pubkey,
            )
        except Exception as exc:
            raise CourierValidationError(
                f"Gift wrap decryption failed: {exc}"
            ) from exc

        # Layer 2: Parse seal event
        try:
            seal = json.loads(seal_json)
        except json.JSONDecodeError as exc:
            raise CourierValidationError(
                f"Seal is not valid JSON: {exc}"
            ) from exc

        if seal.get("kind") != _KIND_SEAL:
            raise CourierValidationError(
                f"Expected kind {_KIND_SEAL} seal, got kind {seal.get('kind')}"
            )

        # Verify sender identity from seal
        seal_pubkey = seal.get("pubkey", "")
        if seal_pubkey != sender_hex:
            raise CourierValidationError(
                f"Seal sender {_hex_to_npub(seal_pubkey) if _HAS_PYNOSTR else seal_pubkey} "
                f"does not match expected sender. "
                f"DM may be from a different npub."
            )

        # Layer 3: Decrypt seal content with our privkey + sender pubkey
        try:
            dm_json = _nip44_decrypt(
                seal.get("content", ""), self._privkey_hex, seal_pubkey,
            )
        except Exception as exc:
            raise CourierValidationError(
                f"Seal decryption failed: {exc}"
            ) from exc

        try:
            dm = json.loads(dm_json)
        except json.JSONDecodeError as exc:
            raise CourierValidationError(
                f"DM is not valid JSON: {exc}"
            ) from exc

        if dm.get("kind") != _KIND_PRIVATE_DM:
            raise CourierValidationError(
                f"Expected kind {_KIND_PRIVATE_DM} DM, got kind {dm.get('kind')}"
            )

        event["encryption"] = "nip44"
        return dm.get("content", "")

    def _request_deletion(self, event_id: str) -> None:
        """Publish a NIP-09 deletion request for a consumed event.

        Fire-and-forget in a daemon thread.
        """
        if not _HAS_PYNOSTR:
            return

        try:
            event = Event(
                kind=_KIND_DELETION,
                content="credential received",
                tags=[["e", event_id]],
                pubkey=self._pubkey_hex,
                created_at=int(time.time()),
            )
            event.sign(self._privkey_hex)
            message = event.to_message()

            thread = threading.Thread(
                target=self._publish_to_relays,
                args=(message,),
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            logger.debug("Failed to create deletion event: %s", exc)

    def _publish_to_relays(self, message: str) -> None:
        """Send an event to all configured relays."""
        sslopt: dict[str, Any] = {}
        for relay_url in self._relays:
            try:
                ws = create_connection(
                    relay_url, timeout=_DEFAULT_SUBSCRIBE_TIMEOUT, sslopt=sslopt,
                )
                try:
                    ws.send(message)
                    ws.recv()
                finally:
                    ws.close()
            except Exception as exc:
                logger.debug(
                    "Relay publish %s failed (non-fatal): %s", relay_url, exc,
                )
