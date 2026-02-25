"""Nostr audit event publisher for Tollbooth ledger operations.

Publishes kind 30078 (NIP-78 parameterized replaceable) events to Nostr
relays on every vault write, providing an independent, cryptographically
signed audit trail for NeonVault cutover diagnostics.

**Privacy**: When the patron's identity is an npub, the ``content`` field
is encrypted via NIP-44v2 (ECDH + ChaCha20-Poly1305).  Only the patron's
nsec can decrypt it.  Observers see the event metadata (kind, tags, pubkey)
but not the balance data.

Architecture:

    LedgerCache → AuditedVault → NeonVault (or TheBrainVault)
                       ↓
                NostrAuditPublisher → Nostr relays (fire-and-forget)

Dependencies are optional — install with ``pip install tollbooth-dpyc[nostr]``.
If pynostr or websocket-client are not installed, the publisher becomes a
silent no-op.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation if not installed
try:
    from pynostr.event import Event  # type: ignore[import-untyped]
    from pynostr.key import PrivateKey  # type: ignore[import-untyped]

    _HAS_PYNOSTR = True
except ImportError:
    _HAS_PYNOSTR = False

try:
    from websocket import create_connection  # type: ignore[import-untyped]

    _HAS_WEBSOCKET = True
except ImportError:
    _HAS_WEBSOCKET = False

try:
    from tollbooth.nip44 import encrypt as _nip44_encrypt

    _HAS_NIP44 = True
except ImportError:
    _HAS_NIP44 = False


def _extract_audit_fields(ledger_json: str) -> dict[str, Any]:
    """Extract balance fields from a ledger JSON string.

    Returns a dict with balance, deposited, consumed, expired, and tranches
    count. Returns zeroes on parse error.
    """
    try:
        obj = json.loads(ledger_json)
        balance = sum(
            t.get("remaining_sats", 0) for t in obj.get("tranches", [])
        )
        return {
            "balance": balance,
            "deposited": obj.get("total_deposited_api_sats", 0),
            "consumed": obj.get("total_consumed_api_sats", 0),
            "expired": obj.get("total_expired_api_sats", 0),
            "tranches": len(obj.get("tranches", [])),
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {
            "balance": 0,
            "deposited": 0,
            "consumed": 0,
            "expired": 0,
            "tranches": 0,
        }


def _npub_to_hex(user_id: str) -> str:
    """Convert an npub bech32 string to hex pubkey.

    If the user_id is not an npub, returns it as-is (best effort for
    the p-tag). Requires pynostr for bech32 decoding.
    """
    if not _HAS_PYNOSTR or not user_id.startswith("npub1"):
        return user_id
    try:
        from pynostr.key import PublicKey  # type: ignore[import-untyped]

        pk = PublicKey.from_npub(user_id)
        return pk.hex()
    except Exception:
        return user_id


class NostrAuditPublisher:
    """Fire-and-forget Nostr event publisher for ledger audit.

    Constructs kind 30078 events with balance state and publishes to
    configured relays. All relay errors are caught and logged — never
    blocks or raises.

    If pynostr or websocket-client are not installed, becomes a silent
    no-op with a one-time warning.
    """

    def __init__(
        self,
        operator_nsec: str,
        relays: list[str],
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._relays = [r.strip() for r in relays if r.strip()]
        self._private_key: Any = None
        self._pubkey_hex: str = ""

        if not enabled:
            logger.debug("NostrAuditPublisher disabled by configuration.")
            return

        if not _HAS_PYNOSTR:
            logger.warning(
                "pynostr not installed — Nostr audit events disabled. "
                "Install with: pip install tollbooth-dpyc[nostr]"
            )
            self._enabled = False
            return

        if not _HAS_WEBSOCKET:
            logger.warning(
                "websocket-client not installed — Nostr audit events disabled. "
                "Install with: pip install tollbooth-dpyc[nostr]"
            )
            self._enabled = False
            return

        if not self._relays:
            logger.warning("No Nostr relays configured — audit events disabled.")
            self._enabled = False
            return

        try:
            self._private_key = PrivateKey.from_nsec(operator_nsec)
            self._pubkey_hex = self._private_key.public_key.hex()
        except Exception as exc:
            logger.warning("Invalid operator nsec — audit events disabled: %s", exc)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Whether the publisher is active and will publish events."""
        return self._enabled

    def publish_ledger_update(
        self,
        user_id: str,
        ledger_json: str,
        event_type: str = "flush",
    ) -> None:
        """Construct and publish a kind 30078 event with balance state.

        When *user_id* is an npub and NIP-44 is available, the ``content``
        field is encrypted to the patron's public key so only the patron's
        nsec can read the balance data.  An ``["encrypted", "nip44"]`` tag
        is added so clients know to decrypt.

        Fire-and-forget: relay publishing runs in a daemon thread so it
        never blocks the vault write path. All exceptions are caught.
        """
        if not self._enabled:
            return

        try:
            fields = _extract_audit_fields(ledger_json)
            user_hex = _npub_to_hex(user_id)
            # Short npub prefix for d-tag (first 12 chars of hex pubkey)
            user_short = user_hex[:12] if len(user_hex) >= 12 else user_hex

            plaintext = json.dumps(
                {
                    "user_id": user_id,
                    "balance": fields["balance"],
                    "deposited": fields["deposited"],
                    "consumed": fields["consumed"],
                    "expired": fields["expired"],
                    "tranches": fields["tranches"],
                    "event_type": event_type,
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                },
                separators=(",", ":"),
            )

            tags = [
                ["d", f"tollbooth-audit-{user_short}"],
                ["p", user_hex],
                ["t", "tollbooth-audit"],
                ["L", "dpyc.tollbooth"],
            ]

            # Encrypt content to patron's npub via NIP-44v2
            can_encrypt = (
                _HAS_NIP44
                and user_id.startswith("npub1")
                and self._private_key is not None
            )
            if can_encrypt:
                content = _nip44_encrypt(
                    plaintext, self._private_key.hex(), user_hex,
                )
                tags.append(["encrypted", "nip44"])
            else:
                # Fallback: skip publishing entirely if encryption unavailable
                # for npub patrons (never publish plaintext for npub users)
                if user_id.startswith("npub1"):
                    logger.warning(
                        "NIP-44 encryption unavailable — skipping audit "
                        "event for npub patron (refusing plaintext fallback)."
                    )
                    return
                content = plaintext

            event = Event(
                kind=30078,
                content=content,
                tags=tags,
                pubkey=self._pubkey_hex,
                created_at=int(time.time()),
            )
            event.sign(self._private_key.hex())

            message = event.to_message()

            # Fire-and-forget in a daemon thread
            thread = threading.Thread(
                target=self._publish_to_relays,
                args=(message,),
                daemon=True,
            )
            thread.start()

        except Exception as exc:
            logger.warning("Failed to construct Nostr audit event: %s", exc)

    def _publish_to_relays(self, message: str) -> None:
        """Send event to all configured relays. Catches all exceptions."""
        sslopt: dict[str, Any] = {}
        for relay_url in self._relays:
            try:
                ws = create_connection(relay_url, timeout=10, sslopt=sslopt)
                try:
                    ws.send(message)
                    resp = ws.recv()
                    logger.debug("Nostr relay %s: %s", relay_url, resp)
                finally:
                    ws.close()
            except Exception as exc:
                logger.debug(
                    "Nostr relay %s failed (non-fatal): %s", relay_url, exc
                )

    def close(self) -> None:
        """Cleanup (no persistent state to release)."""
        pass


class AuditedVault:
    """VaultBackend decorator that publishes Nostr audit events on writes.

    Wraps any VaultBackend. On ``store_ledger()`` and ``snapshot_ledger()``,
    delegates to the inner vault first, then fires a Nostr audit event.
    ``fetch_ledger()`` is a pure passthrough.

    If the publisher is disabled or encounters errors, the inner vault
    write always succeeds — audit is strictly best-effort.
    """

    def __init__(self, inner: Any, publisher: NostrAuditPublisher) -> None:
        self._inner = inner
        self._publisher = publisher

    async def store_ledger(self, user_id: str, ledger_json: str) -> str:
        """Delegate to inner vault, then publish audit event."""
        result = await self._inner.store_ledger(user_id, ledger_json)
        self._publisher.publish_ledger_update(user_id, ledger_json, "flush")
        return result

    async def fetch_ledger(self, user_id: str) -> str | None:
        """Pure passthrough to inner vault."""
        return await self._inner.fetch_ledger(user_id)

    async def snapshot_ledger(
        self, user_id: str, ledger_json: str, timestamp: str,
    ) -> str | None:
        """Delegate to inner vault, then publish audit event."""
        result = await self._inner.snapshot_ledger(
            user_id, ledger_json, timestamp,
        )
        self._publisher.publish_ledger_update(
            user_id, ledger_json, "snapshot",
        )
        return result
