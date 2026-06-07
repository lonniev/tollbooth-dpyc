"""Nostr DM credential exchange — the Secure Courier Service.

Out-of-band credential delivery via NIP-44 (or NIP-04 fallback) encrypted
Nostr DMs.  Patrons send credentials from their Nostr client to the MCP
operator's npub; the operator picks them up, validates, stores, and
destroys the relay copy.

Two-tool operator interface:

    exchange = NostrCredentialExchange(nsec, relays, templates)

    # Tool 1: Send welcome DM to patron (they just reply)
    await exchange.open_channel("x", recipient_npub="npub1user...")

    # Tool 2: Pick up the sealed pouch
    await exchange.receive("npub1user...")

Dependencies are optional — install with ``pip install tollbooth-dpyc[nostr]``.

Reference: https://github.com/nostr-protocol/nips/blob/master/44.md (NIP-44)
           https://github.com/nostr-protocol/nips/blob/master/04.md (NIP-04)
           https://github.com/nostr-protocol/nips/blob/master/09.md (NIP-09)
           https://github.com/nostr-protocol/nips/blob/master/01.md (NIP-01 kind 0)
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from tollbooth.credential_templates import (
    CredentialTemplate,
    FieldSpec,
    TemplateValidationError,
    render_delimited_instructions,
    validate_payload,
)
from tollbooth.credential_vault_backend import CredentialVaultBackend
from tollbooth.constants import ErrorCode

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
    from tollbooth.nip44 import encrypt as _nip44_encrypt

    _HAS_NIP44 = True
except ImportError:
    _HAS_NIP44 = False

try:
    from tollbooth.nip04 import decrypt as _nip04_decrypt

    _HAS_NIP04 = True
except ImportError:
    _HAS_NIP04 = False


# Nostr event kinds
_KIND_METADATA = 0  # NIP-01 kind 0 (user metadata / profile)
_KIND_ENCRYPTED_DM = 4  # NIP-04 legacy DMs
_KIND_GIFT_WRAP = 1059  # NIP-17/NIP-59 gift-wrapped DMs
_KIND_SEAL = 13  # NIP-59 seal (inner layer of gift wrap)
_KIND_PRIVATE_DM = 14  # NIP-17 private DM (innermost content)
_KIND_DELETION = 5  # NIP-09 event deletion

# NIP-17 timestamp randomization window (0–48 hours into the past)
_TIMESTAMP_FUZZ_SECONDS = 2 * 24 * 60 * 60


def _randomize_timestamp() -> int:
    """Return a timestamp fuzzed 0–48 hours into the past per NIP-17.

    NIP-17 recommends randomizing ``created_at`` to thwart time-analysis.
    We only fuzz *backwards* because Nostr relays reject events whose
    ``created_at`` is in the future (anti-spam policy on Primal, Damus,
    nos.lol, and most others).
    """
    return int(time.time()) - random.randint(0, _TIMESTAMP_FUZZ_SECONDS)


# Anti-replay poison slug — memorable adjective-noun-number phrases
_ADJECTIVES = [
    "amber", "bold", "calm", "dark", "eager", "faint", "glad", "hazy",
    "iron", "jade", "keen", "live", "mild", "neat", "opal", "pale",
    "quick", "rare", "soft", "true", "ultra", "vivid", "warm", "zinc",
]
_NOUNS = [
    "arrow", "blade", "cloud", "dusk", "eagle", "flame", "grove", "hawk",
    "iris", "jewel", "knot", "lake", "mesa", "nest", "orbit", "peak",
    "quill", "reef", "spark", "tide", "vale", "wave", "yarn", "zero",
]


def _generate_poison() -> str:
    """Generate a memorable anti-replay phrase like ``bold-hawk-42``."""
    adj = random.choice(_ADJECTIVES)
    noun = random.choice(_NOUNS)
    num = random.randint(10, 99)
    return f"{adj}-{noun}-{num}"


# @@@ delimiter extraction — mobile-friendly credential format
# Tempered greedy: match any non-@ char, or @ not followed by @@.
# [^@] matches newlines naturally (no DOTALL needed), so this handles
# line-wrapped values from mobile Nostr clients without bleeding across
# adjacent @@@ delimiters.
_DELIMITED_FIELD = re.compile(r"(\w+)\s*=\s*@@@((?:[^@]|@(?!@@))*)@@@")


def _parse_delimited_credentials(text: str) -> dict[str, str] | None:
    """Extract ``key = @@@value@@@`` pairs from *text*.

    Returns a dict of field→value mappings, or ``None`` if no @@@ patterns
    were found (indicating the message doesn't contain credentials).
    """
    matches = _DELIMITED_FIELD.findall(text)
    if not matches:
        return None
    # Strip newlines that mobile clients may inject; preserve intentional spaces
    return {key.strip(): re.sub(r"\r?\n", "", value).strip() for key, value in matches}


@dataclass
class NostrProfile:
    """Nostr kind 0 profile metadata for the operator's npub.

    Published so patrons see a friendly name and avatar in their
    Nostr client instead of a raw npub string.
    """

    name: str
    display_name: str | None = None
    about: str | None = None
    picture: str | None = None
    nip05: str | None = None
    banner: str | None = None
    website: str | None = None
    lud16: str | None = None  # Lightning address
    extra: dict[str, str] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, str]:
        """Serialize to the NIP-01 kind 0 content dict."""
        meta: dict[str, str] = {"name": self.name}
        if self.display_name:
            meta["display_name"] = self.display_name
        if self.about:
            meta["about"] = self.about
        if self.picture:
            meta["picture"] = self.picture
        if self.nip05:
            meta["nip05"] = self.nip05
        if self.banner:
            meta["banner"] = self.banner
        if self.website:
            meta["website"] = self.website
        if self.lud16:
            meta["lud16"] = self.lud16
        meta.update(self.extra)
        return meta

# Default freshness window (seconds)
_DEFAULT_FRESHNESS = 900  # 15 minutes
_DEFAULT_SUBSCRIBE_TIMEOUT = 10

# NACK copy sent to the sender of a popped courier DM whose anti-replay
# token did NOT match the open channel. Deliberately does NOT reveal the
# expected poison phrase — telling an unknown sender the token they should
# have used would defeat the anti-replay guarantee.
_NACK_TOKEN = (
    "Your message was popped from the courier queue, but its session "
    "phrase didn't match an open request on this channel, so it has been "
    "removed. Ask the service for the correct session phrase, then resend "
    "your reply using the @@@ format from the welcome message."
)

# Cap on how many NACK DMs a single drain will send, so a flood of junk
# DMs from one sender can't be amplified into a flood of outbound replies.
# Beyond the cap, mismatched DMs are still popped + NIP-09 deleted silently.
_MAX_NACKS_PER_DRAIN = 5

# Patron-actionable prose for each courier-resolution failure. Keyed by the
# ErrorCode the resolver returns; the agent branches on error_code, the human
# reads the message. ``{request_tool}`` is filled per flow (the proof flow
# names request_npub_proof, credential flows name request_*credentials*) so
# the recovery hint always points at the tool that actually opens THIS channel.
_COURIER_RESOLVE_ERRORS = {
    ErrorCode.COURIER_NO_PENDING_RECORD: (
        "No open courier channel for this npub. Call {request_tool} first, "
        "then retrieve with the poison it returns."
    ),
    ErrorCode.COURIER_POISON_MISMATCH: (
        "The supplied session phrase does not match the open channel for this "
        "npub. Use the exact poison returned by {request_tool}, or open a "
        "fresh channel."
    ),
    ErrorCode.COURIER_TOKEN_EXPIRED: (
        "This courier channel's freshness window has elapsed. Call "
        "{request_tool} again for a fresh session phrase."
    ),
    ErrorCode.COURIER_NO_PINNED_RELAY: (
        "The open channel has no pinned rendezvous relay on record, so there "
        "is no single relay to drain. Call {request_tool} for a fresh channel."
    ),
}


def _courier_resolve_error(error_code: str, request_tool: str) -> str:
    """Format the patron-facing resolve-error prose for the given flow.

    ``request_tool`` is the request tool that opens this kind of channel
    (request_credential_channel / request_patron_credentials /
    request_npub_proof) so the recovery hint names the right one.
    """
    template = _COURIER_RESOLVE_ERRORS.get(
        error_code, "Could not resolve the courier channel.",
    )
    return template.replace("{request_tool}", request_tool)


def _npub_to_hex(npub: str) -> str:
    """Convert npub bech32 to 32-byte hex pubkey."""
    if not _HAS_PYNOSTR:
        raise RuntimeError("pynostr required for npub conversion")
    if not npub or not npub.startswith("npub1"):
        raise ValueError(
            f"Expected an npub (npub1...), got: {npub!r}"
        )
    try:
        return PublicKey.from_npub(npub).hex()
    except TypeError:
        # pynostr raises TypeError when bech32 checksum fails
        # (bech32_decode returns None, which convertbits iterates)
        raise ValueError(
            f"Invalid npub — bech32 checksum failed for "
            f"'{npub[:16]}...'. Verify the npub is correct."
        )


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


class CourierUnreachableError(CourierError):
    """Raised when no configured relay accepted the challenge publish.

    Surfaces as a lifecycle state to the caller: the courier exhausted
    every candidate relay without a successful publish, so no rendezvous
    could be pinned and the responder has nowhere to reply. The caller
    should re-issue request_credential_channel / request_npub_proof
    after checking relay connectivity.
    """


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
            self._nsec = nsec
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
        # Poison slugs: {(recipient_npub, service): (phrase, expiry_timestamp)}
        self._pending_poisons: dict[tuple[str, str], tuple[str, float]] = {}
        # Per-conversation rendezvous relay pins: {(recipient_npub, service): wss_url}
        # The wheel commits one relay per challenge (the one that accepted the
        # publish) and embeds it in the DM body so the responder knows where
        # to send their reply. Lives alongside _pending_poisons; cleared
        # together on receive success / expiry.
        self._pinned_relays: dict[tuple[str, str], str] = {}
        # Ephemeral agent keys for self-DM avoidance:
        # {(recipient_npub, service): PrivateKey}
        self._ephemeral_agents: dict[tuple[str, str], PrivateKey] = {}

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

    def publish_profile(self, profile: NostrProfile) -> None:
        """Publish a NIP-01 kind 0 profile event for the operator's npub.

        Sends the profile metadata to all configured relays so patrons
        see a friendly name and avatar instead of a raw npub string.

        Args:
            profile: Operator profile metadata.
        """
        if not self._enabled:
            logger.warning("Cannot publish profile — Secure Courier not enabled.")
            return

        metadata = profile.to_metadata()
        content = json.dumps(metadata)

        try:
            event = Event(
                kind=_KIND_METADATA,
                content=content,
                tags=[],
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
            logger.info("Published kind 0 profile for %s", self._npub[:20])
        except Exception as exc:
            logger.warning("Failed to publish profile: %s", exc)

    def send_dm(
        self,
        recipient_npub: str,
        message_text: str,
        *,
        target_relay: str | None = None,
    ) -> None:
        """Send a DM via both NIP-17 gift wrap and NIP-04 legacy.

        Sends the same message twice — once as a NIP-17 kind 1059 gift
        wrap and once as a NIP-04 kind 4 encrypted DM — so the recipient
        sees whichever their client supports.  The receive side already
        handles both protocols.

        Raises ``CourierError`` only if *both* protocols fail to get at
        least one relay acceptance.

        Args:
            recipient_npub: Patron's npub (bech32).
            message_text: Plaintext message to encrypt and send.
            target_relay: If set, publish only to this single relay
                (rendezvous-pinned send used by ``open_channel``).
                When ``None`` (default), fan out to all configured relays
                — used for profile publishes, deletions, and ack DMs that
                don't need pinning.
        """
        if not self._enabled:
            raise CourierNotReady("Secure Courier not enabled.")

        if not _HAS_NIP44:
            raise CourierNotReady("NIP-44 module required for outbound DMs.")

        try:
            recipient_hex = _npub_to_hex(recipient_npub)
        except Exception as exc:
            raise CourierValidationError(
                f"Invalid recipient npub: {exc}"
            ) from exc

        nip17_ok = False
        nip04_ok = False
        errors: list[str] = []

        # ── NIP-17 gift wrap (kind 1059) ─────────────────────────────
        try:
            message = self._build_gift_wrap(recipient_hex, message_text)
            if target_relay:
                ok, detail = self._publish_to_one_relay(message, target_relay)
                nip17_ok = ok
                if nip17_ok:
                    logger.info(
                        "Sent NIP-17 gift-wrapped DM to %s via pinned relay %s",
                        recipient_npub[:20], target_relay,
                    )
                else:
                    errors.append(f"NIP-17 @ {target_relay}: {detail}")
            else:
                results = self._publish_to_relays(message) or []
                accepted = sum(1 for _, ok, _ in results if ok)
                nip17_ok = accepted > 0
                if nip17_ok:
                    logger.info(
                        "Sent NIP-17 gift-wrapped DM to %s (%d/%d relays)",
                        recipient_npub[:20], accepted, len(results),
                    )
                else:
                    details = "; ".join(
                        f"{url}: {err}" for url, ok, err in results if not ok
                    )
                    errors.append(f"NIP-17: {details}")
        except Exception as exc:
            errors.append(f"NIP-17: {exc}")
            logger.debug("NIP-17 send failed: %s", exc)

        # ── NIP-04 legacy DM (kind 4) ───────────────────────────────
        try:
            message = self._build_nip04_dm(recipient_hex, message_text)
            if target_relay:
                ok, detail = self._publish_to_one_relay(message, target_relay)
                nip04_ok = ok
                if nip04_ok:
                    logger.info(
                        "Sent NIP-04 DM to %s via pinned relay %s",
                        recipient_npub[:20], target_relay,
                    )
                else:
                    errors.append(f"NIP-04 @ {target_relay}: {detail}")
            else:
                results = self._publish_to_relays(message) or []
                accepted = sum(1 for _, ok, _ in results if ok)
                nip04_ok = accepted > 0
                if nip04_ok:
                    logger.info(
                        "Sent NIP-04 DM to %s (%d/%d relays)",
                        recipient_npub[:20], accepted, len(results),
                    )
                else:
                    details = "; ".join(
                        f"{url}: {err}" for url, ok, err in results if not ok
                    )
                    errors.append(f"NIP-04: {details}")
        except Exception as exc:
            errors.append(f"NIP-04: {exc}")
            logger.debug("NIP-04 send failed: %s", exc)

        if not nip17_ok and not nip04_ok:
            raise CourierError(
                "All relay sends failed for both protocols: "
                + "; ".join(errors)
            )

    def _send_dm_as(
        self,
        sender_key: PrivateKey,
        recipient_npub: str,
        message_text: str,
        *,
        target_relay: str | None = None,
    ) -> None:
        """Send a DM using an explicit sender key instead of the operator key.

        Same dual-protocol logic as ``send_dm()`` but signs and encrypts
        with the provided ``sender_key``.  Used for ephemeral agent DMs
        when the operator is self-onboarding (recipient == operator npub).

        Args:
            sender_key: The PrivateKey to use for signing/encrypting.
            recipient_npub: Recipient's npub (bech32).
            message_text: Plaintext message to encrypt and send.
            target_relay: If set, publish only to this single relay
                (used by the rendezvous-pinned open_channel path).
        """
        if not self._enabled:
            raise CourierNotReady("Secure Courier not enabled.")

        if not _HAS_NIP44:
            raise CourierNotReady("NIP-44 module required for outbound DMs.")

        try:
            recipient_hex = _npub_to_hex(recipient_npub)
        except Exception as exc:
            raise CourierValidationError(
                f"Invalid recipient npub: {exc}"
            ) from exc

        sender_privkey_hex = sender_key.hex()
        sender_pubkey_hex = sender_key.public_key.hex()

        nip17_ok = False
        nip04_ok = False
        errors: list[str] = []

        # ── NIP-17 gift wrap (kind 1059) ─────────────────────────────
        try:
            message = self._build_gift_wrap_with(
                sender_privkey_hex, sender_pubkey_hex,
                recipient_hex, message_text,
            )
            if target_relay:
                ok, detail = self._publish_to_one_relay(message, target_relay)
                nip17_ok = ok
                if nip17_ok:
                    logger.info(
                        "Sent NIP-17 agent DM to %s via pinned relay %s",
                        recipient_npub[:20], target_relay,
                    )
                else:
                    errors.append(f"NIP-17 @ {target_relay}: {detail}")
            else:
                results = self._publish_to_relays(message) or []
                accepted = sum(1 for _, ok, _ in results if ok)
                nip17_ok = accepted > 0
                if nip17_ok:
                    logger.info(
                        "Sent NIP-17 agent DM to %s (%d/%d relays)",
                        recipient_npub[:20], accepted, len(results),
                    )
                else:
                    details = "; ".join(
                        f"{url}: {err}" for url, ok, err in results if not ok
                    )
                    errors.append(f"NIP-17: {details}")
        except Exception as exc:
            errors.append(f"NIP-17: {exc}")
            logger.debug("NIP-17 agent send failed: %s", exc)

        # ── NIP-04 legacy DM (kind 4) ───────────────────────────────
        try:
            message = self._build_nip04_dm_with(
                sender_privkey_hex, sender_pubkey_hex,
                recipient_hex, message_text,
            )
            if target_relay:
                ok, detail = self._publish_to_one_relay(message, target_relay)
                nip04_ok = ok
                if nip04_ok:
                    logger.info(
                        "Sent NIP-04 agent DM to %s via pinned relay %s",
                        recipient_npub[:20], target_relay,
                    )
                else:
                    errors.append(f"NIP-04 @ {target_relay}: {detail}")
            else:
                results = self._publish_to_relays(message) or []
                accepted = sum(1 for _, ok, _ in results if ok)
                nip04_ok = accepted > 0
                if nip04_ok:
                    logger.info(
                        "Sent NIP-04 agent DM to %s (%d/%d relays)",
                        recipient_npub[:20], accepted, len(results),
                    )
                else:
                    details = "; ".join(
                        f"{url}: {err}" for url, ok, err in results if not ok
                    )
                    errors.append(f"NIP-04: {details}")
        except Exception as exc:
            errors.append(f"NIP-04: {exc}")
            logger.debug("NIP-04 agent send failed: %s", exc)

        if not nip17_ok and not nip04_ok:
            raise CourierError(
                "All relay sends failed for both protocols: "
                + "; ".join(errors)
            )

    def _build_gift_wrap(
        self, recipient_hex: str, message_text: str,
    ) -> str:
        """Build a NIP-17 gift-wrapped DM (kind 1059 → 13 → 14).

        Returns the Nostr protocol message (``["EVENT", {...}]``) ready
        for relay publishing.
        """
        return self._build_gift_wrap_with(
            self._privkey_hex, self._pubkey_hex, recipient_hex, message_text,
        )

    def _build_gift_wrap_with(
        self,
        sender_privkey_hex: str,
        sender_pubkey_hex: str,
        recipient_hex: str,
        message_text: str,
    ) -> str:
        """Build a NIP-17 gift-wrapped DM with explicit sender keys.

        Returns the Nostr protocol message (``["EVENT", {...}]``) ready
        for relay publishing.
        """
        # Layer 3: Kind 14 rumor — unsigned private DM
        rumor = {
            "kind": _KIND_PRIVATE_DM,
            "content": message_text,
            "tags": [["p", recipient_hex]],
            "pubkey": sender_pubkey_hex,
            "created_at": int(time.time()),
        }

        # Layer 2: Kind 13 seal — signed by sender, content NIP-44 encrypted
        seal_content = _nip44_encrypt(
            json.dumps(rumor), sender_privkey_hex, recipient_hex,
        )
        seal = Event(
            kind=_KIND_SEAL,
            content=seal_content,
            tags=[],  # No tags on seal (metadata protection)
            pubkey=sender_pubkey_hex,
            created_at=_randomize_timestamp(),
        )
        seal.sign(sender_privkey_hex)

        # Layer 1: Kind 1059 gift wrap — random ephemeral key
        ephemeral_sk = PrivateKey()
        wrap_content = _nip44_encrypt(
            json.dumps(seal.to_dict()), ephemeral_sk.hex(), recipient_hex,
        )
        expiry = str(int(time.time()) + 600)  # NIP-40: expire after 10 minutes
        wrap = Event(
            kind=_KIND_GIFT_WRAP,
            content=wrap_content,
            tags=[
                ["p", recipient_hex],  # p-tag for relay routing
                ["expiration", expiry],  # NIP-40 auto-delete
            ],
            pubkey=ephemeral_sk.public_key.hex(),
            created_at=_randomize_timestamp(),
        )
        wrap.sign(ephemeral_sk.hex())

        return wrap.to_message()

    def _build_nip04_dm(
        self, recipient_hex: str, message_text: str,
    ) -> str:
        """Build a NIP-04 kind 4 encrypted DM.

        Returns the Nostr protocol message (``["EVENT", {...}]``) ready
        for relay publishing.
        """
        return self._build_nip04_dm_with(
            self._privkey_hex, self._pubkey_hex, recipient_hex, message_text,
        )

    def _build_nip04_dm_with(
        self,
        sender_privkey_hex: str,
        sender_pubkey_hex: str,
        recipient_hex: str,
        message_text: str,
    ) -> str:
        """Build a NIP-04 kind 4 encrypted DM with explicit sender keys.

        Returns the Nostr protocol message (``["EVENT", {...}]``) ready
        for relay publishing.
        """
        if not _HAS_NIP04:
            raise CourierNotReady("NIP-04 module required for legacy DMs.")

        from tollbooth.nip04 import encrypt as _nip04_encrypt

        encrypted = _nip04_encrypt(
            message_text, sender_privkey_hex, recipient_hex,
        )
        now = int(time.time())
        expiry = str(now + 600)  # NIP-40: expire after 10 minutes
        event = Event(
            kind=_KIND_ENCRYPTED_DM,
            content=encrypted,
            tags=[["p", recipient_hex], ["expiration", expiry]],
            pubkey=sender_pubkey_hex,
            created_at=now,
        )
        event.sign(sender_privkey_hex)
        return event.to_message()

    async def open_channel(
        self,
        service: str,
        *,
        greeting: str,
        recipient_npub: str | None = None,
    ) -> dict[str, Any]:
        """Open a credential delivery channel for a service.

        If ``recipient_npub`` is provided, sends a welcome DM to the
        patron with template instructions.  The patron just replies
        to the thread instead of composing a new DM to an unfamiliar npub.

        If ``recipient_npub`` is omitted, falls back to returning the
        operator's npub and instructions for manual DM initiation.

        Args:
            service: Template service name (e.g., "x", "openai").
            greeting: Operator-provided banner shown at the top of the
                welcome DM.
            recipient_npub: Optional patron npub to send welcome DM to.

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
        instructions = render_delimited_instructions(template)

        # Generate anti-replay poison slug
        poison = _generate_poison()

        # Start background subscription
        self._start_subscription()

        # Build the welcome message for the patron
        from datetime import datetime, timezone
        from tollbooth import __version__ as _tb_version
        from tollbooth.credential_templates import render_credential_payload_lines

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        payload_lines = render_credential_payload_lines(template)

        # Detect self-DM (operator onboarding themselves)
        is_self_dm = recipient_npub == self._npub
        agent_key: PrivateKey | None = None

        if is_self_dm:
            agent_key = PrivateKey()
            self._ephemeral_agents[(recipient_npub, service)] = agent_key
            logger.info(
                "Self-DM detected — using ephemeral agent %s for %s/%s",
                agent_key.public_key.bech32()[:20],
                recipient_npub[:20], service,
            )

        operator_line_value = (
            agent_key.public_key.bech32() if agent_key is not None else self._npub
        )

        def build_welcome(rendezvous_relay: str) -> str:
            """Build the welcome DM with the rendezvous relay embedded.

            The rendezvous_relay field tells the responder which relay to
            publish their reply to, so it lands where the courier is
            listening. Called once per candidate relay during the
            sequential publish loop.
            """
            return (
                f"{greeting}\n\n"
                f"--- Credential Payload ---\n"
                + "\n".join(payload_lines) + "\n"
                f"  poison = @@@{poison}@@@\n"
                f"  rendezvous_relay = @@@{rendezvous_relay}@@@\n\n"
                f"IMPORTANT: include the anti-replay token exactly as shown.\n"
                f"IMPORTANT: reply via the rendezvous_relay above — the "
                f"courier listens there.\n\n"
                f"--- Message Provenance ---\n"
                f"Service: {template.description or template.service}\n"
                f"Operator: {operator_line_value}\n"
                f"Sent: {timestamp}\n"
                f"Protocol: DPYC Secure Courier v{_tb_version}\n\n"
                f"Your reply is end-to-end encrypted — "
                f"only this service can read it.\n"
                f"If you didn't request this, simply ignore this message."
            )

        result: dict[str, Any] = {
            "success": True,
            "npub": self._npub,
            "relays": self._relays,
            "service": service,
            "freshness_window_seconds": self._freshness_window,
            "instructions": instructions,
            "poison": poison,
        }

        if agent_key is not None:
            result["agent_npub"] = agent_key.public_key.bech32()

        if recipient_npub:
            # ── Rendezvous-pinned publish ─────────────────────────────
            # Try each configured relay in order. The first one that
            # accepts at least one protocol (NIP-17 or NIP-04) becomes
            # the per-conversation rendezvous. Embed that relay's URL
            # in the DM body before signing so the responder knows
            # where to reply.
            committed_relay: str | None = None
            attempt_errors: list[str] = []
            for candidate_relay in self._relays:
                welcome_text = build_welcome(candidate_relay)
                try:
                    if agent_key is not None:
                        self._send_dm_as(
                            agent_key, recipient_npub, welcome_text,
                            target_relay=candidate_relay,
                        )
                    else:
                        self.send_dm(
                            recipient_npub, welcome_text,
                            target_relay=candidate_relay,
                        )
                    committed_relay = candidate_relay
                    break
                except Exception as exc:
                    # Catch broadly: relay socket errors, NACKs, NIP-44
                    # encryption issues all surface as different types.
                    # Move on to the next candidate; if none accept, the
                    # caller gets CourierUnreachableError below.
                    attempt_errors.append(f"{candidate_relay}: {exc}")
                    logger.info(
                        "Relay %s rejected challenge for %s/%s — trying next",
                        candidate_relay, recipient_npub[:20], service,
                    )
                    continue

            if committed_relay is None:
                # No relay accepted — lifecycle state, not a stack trace.
                # Record nothing in pending state; caller must re-issue
                # after checking relay connectivity.
                raise CourierUnreachableError(
                    f"No configured relay accepted the challenge for "
                    f"{recipient_npub} ({len(self._relays)} attempted). "
                    f"Re-issue request after verifying relay connectivity. "
                    f"Errors: {'; '.join(attempt_errors)}"
                )

            # Pin committed. Persist poison + rendezvous so receive can
            # match the reply and the caller can display the URL.
            expiry = time.time() + self._freshness_window
            self._pending_poisons[(recipient_npub, service)] = (poison, expiry)
            self._pinned_relays[(recipient_npub, service)] = committed_relay

            # Persist pending state to survive cold starts
            if self._credential_vault is not None:
                pending_blob: dict[str, Any] = {
                    "poison": poison,
                    "expiry": expiry,
                    "rendezvous_relay": committed_relay,
                }
                if agent_key is not None:
                    pending_blob["agent_nsec_hex"] = agent_key.hex()
                await self._vault_store(
                    f"__pending__{service}", recipient_npub, pending_blob,
                )

            result["welcome_dm_sent"] = True
            result["rendezvous_relay"] = committed_relay
            result["message"] = (
                f"A welcome DM has been sent to {recipient_npub} via {committed_relay}. "
                f"Tell the user to look for a DM containing the session phrase "
                f"\"{poison}\" — share this phrase in chat so they can identify "
                f"the correct message. It is not sensitive. "
                f"They MUST reply on the rendezvous relay {committed_relay} "
                f"(included in the DM body) using the @@@ format shown. "
                f"Then call receive_credentials with their npub, the service, "
                f"and the session phrase \"{poison}\" as the poison argument."
            )
            result["relay_propagation_note"] = (
                f"Reply must land on {committed_relay} — the courier listens there. "
                f"Nostr relay propagation may take 30-60 seconds."
            )
        else:
            # No recipient — pin-less manual flow. Caller will need to
            # initiate the DM themselves, so we can't commit a relay
            # pin. Embed every configured relay in the instructions so
            # the caller can pick one.
            result["welcome_dm_sent"] = False
            result["message"] = (
                f"The user should send their {template.service} credentials as a Nostr DM "
                f"to {self._npub} from their Nostr client, including the session phrase "
                f"\"{poison}\" — share this phrase in chat so they can include it. "
                f"Reply with credentials using the @@@ format shown above. "
                f"Then call receive_credentials with their npub, the service, "
                f"and the session phrase \"{poison}\" as the poison argument."
            )

        return result

    async def receive_from_vault(
        self, sender_npub: str, *, service: str | None = None,
    ) -> dict[str, Any]:
        """Vault-only credential lookup — no relay I/O, no poison.

        This is the cold-start / returning-session path: it answers "do we
        already hold complete credentials for this (service, npub)?" without
        touching the Secure Courier relay protocol. Used by
        ``SecureCourier.restore_session`` to re-establish an operator session
        on a serverless cold start. It is deliberately separate from
        ``receive()`` — the agent-facing courier retrieve is strict and
        poison-scoped, while this is a plain encrypted-store read.

        Returns a success dict (``encryption: "vault"``) on a complete vault
        hit, else ``{"success": False}``.
        """
        vault_service = self._resolve_service(service)
        if self._credential_vault is None or not vault_service:
            return {"success": False}

        vaulted = await self._vault_fetch(vault_service, sender_npub)
        if vaulted is None:
            return {"success": False}

        template = self._templates[vault_service]
        required_fields = {
            name for name, spec in template.fields.items() if spec.required
        }
        if not required_fields.issubset(vaulted.keys()):
            logger.info(
                "Vault blob for %s/%s has %d fields but template requires %s.",
                vault_service, sender_npub[:16],
                len(vaulted), required_fields - vaulted.keys(),
            )
            return {"success": False}

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

    async def receive(
        self, sender_npub: str, *, service: str, poison: str,
        request_tool: str = "request_credential_channel",
    ) -> dict[str, Any]:
        """Pick up validated credentials via the deterministic courier drain.

        The client names exactly which response it wants — ``(sender_npub,
        service, poison)`` — and the drain is unambiguous:

        1. Resolve the pending channel for ``(sender_npub, service)``, verify
           ``poison`` matches it and hasn't expired, and read the pinned
           rendezvous relay the original request committed to.
        2. Drain ONLY that relay. Every popped DM whose poison is wrong (or is
           undecryptable / lacks ``@@@`` markers) is NIP-09 deleted and the
           sender is NACK'd (without revealing the expected poison). The first
           DM whose poison matches is deleted, the sender is ACK'd, and the
           scan STOPS (stop-at-match).
        3. On a match the credentials are validated, vaulted, and returned.
           If the relay drains with no match, a structured not-found result
           is returned and the queue is now empty of the sought DM.

        All three arguments are required — there is no vault-first fallback
        and no guessing. (Returning-session vault reads go through
        ``receive_from_vault``.)

        Returns:
            On success, a dict with ``success: True``, ``credentials``, and
            field counts. On failure, ``success: False`` with an
            ``error_code`` from the ``COURIER_*`` family and ``popped``.
        """
        if not self._enabled:
            raise CourierNotReady(
                "Secure Courier not available. Check logs for missing dependencies."
            )

        if not poison:
            return {
                "success": False,
                "error_code": ErrorCode.POISON_MISSING,
                "popped": 0,
                "error": (
                    f"poison is required — pass the session phrase returned by "
                    f"{request_tool}."
                ),
            }

        try:
            sender_hex = _npub_to_hex(sender_npub)
        except Exception as exc:
            raise CourierValidationError(f"Invalid sender npub: {exc}") from exc

        # Channel state (poison, pin, ephemeral agent, __pending__ vault blob)
        # is keyed by the RAW service to match open_channel's write. The
        # resolved service is used only for template matching/validation later.
        resolved_service = self._resolve_service(service)
        poison_key = (sender_npub, service)

        # Resolve + verify the pinned rendezvous relay for this exact
        # (npub, service, poison). Handles cold-start vault rehydration and
        # returns a structured error (popping nothing) when the channel can't
        # be resolved.
        pinned, error_code = await self._resolve_pinned_record(
            sender_npub, service, poison,
        )
        if error_code is not None:
            return {
                "success": False,
                "error_code": error_code,
                "popped": 0,
                "error": _courier_resolve_error(error_code, request_tool),
            }

        error_template = self._resolve_error_template(service)

        # Self-DM decryption uses the ephemeral agent key restored by the
        # resolver; falls back to the operator nsec for patron→operator DMs.
        agent_key = self._ephemeral_agents.get(poison_key)
        decrypt_privkey = agent_key.hex() if agent_key else None

        # ── Strict pinned-relay drain ──────────────────────────────────
        # Fetch from ONLY the pinned relay, then consider only candidates that
        # actually arrived on it (the buffer may hold cross-relay events from
        # an earlier subscription). Events with no relay tag are not excluded:
        # every real subscription stamps ``_relay`` (see _subscribe_one_relay),
        # so an untagged event only appears in tests.
        self._fetch_dms_from_relays([pinned])
        candidates = [
            c for c in self._find_dm_candidates(sender_hex)
            if c.get("_relay") in (pinned, None)
        ]

        dm = None
        plaintext = None
        popped = 0
        nacks_sent = 0

        for candidate in candidates:
            event_id = candidate.get("id", "")
            kind = candidate.get("kind", 0)

            # Classify the candidate: either it matches (pt set) or it earns a
            # NACK reason. The pop happens once, below, so the outbound-NACK
            # cap is applied uniformly.
            nack_reason: str | None = None
            pt: str | None = None

            if kind == _KIND_ENCRYPTED_DM and self._nip44_only:
                nack_reason = (
                    "Your NIP-04 message was popped and removed — this channel "
                    "accepts only NIP-44. Use a modern Nostr client and resend."
                )
            else:
                try:
                    pt = self._decrypt_dm(
                        candidate, sender_hex, decrypt_privkey_hex=decrypt_privkey,
                    )
                except Exception:
                    pt = None
                    nack_reason = _NACK_TOKEN
                if pt is not None:
                    payload = _parse_delimited_credentials(pt)
                    if payload is None:
                        nack_reason = (
                            "Your message was popped but had no @@@ field "
                            "markers. Resend using: field_name = @@@value@@@"
                        )
                    elif payload.get("poison", "") != poison:
                        nack_reason = _NACK_TOKEN

            if nack_reason is None:
                # Valid match — pop it (ACK sent after a successful vault write)
                dm = candidate
                plaintext = pt
                self._pop_event(event_id)
                popped += 1
                break  # stop-at-match

            # Mismatch — pop + NIP-09 delete, NACK the sender up to the cap so
            # a flood of junk DMs can't be amplified into a flood of replies.
            if nacks_sent < _MAX_NACKS_PER_DRAIN:
                self._pop_event(event_id, sender_npub, nack_reason, target_relay=pinned)
                nacks_sent += 1
            else:
                self._pop_event(event_id)
            popped += 1

        logger.info(
            "Courier drain on %s: popped %d, match=%s",
            pinned, popped, "found" if dm else "none",
        )

        if dm is None:
            return {
                "success": False,
                "error_code": ErrorCode.COURIER_NOT_FOUND,
                "popped": popped,
                "error": (
                    f"Drained the pinned relay ({pinned}); none of the "
                    f"{popped} message(s) carried the expected session phrase. "
                    f"The queue is now empty of the sought reply — confirm the "
                    f"patron replied on {pinned}, or call {request_tool} for a "
                    f"fresh channel."
                ),
            }

        # ── Match: consume one-time channel state (keyed by raw service) ──
        self._pending_poisons.pop(poison_key, None)
        self._pinned_relays.pop(poison_key, None)
        self._ephemeral_agents.pop(poison_key, None)
        if self._credential_vault is not None:
            try:
                await self._credential_vault.delete_credentials(
                    f"__pending__{service}", sender_npub,
                )
            except Exception:
                pass  # best-effort cleanup

        payload = _parse_delimited_credentials(plaintext)
        if payload is None:
            self._send_error_dm(sender_npub, error_template)
            raise CourierValidationError(
                "Could not parse credentials from DM. "
                "Use the @@@ format from the welcome message, e.g.: "
                "field_name = @@@your_value@@@"
            )

        payload.pop("poison", None)
        template = self._match_template(resolved_service, payload)

        try:
            validated = validate_payload(payload, template)
        except TemplateValidationError as exc:
            self._send_error_dm(sender_npub, template)
            raise CourierValidationError(str(exc)) from exc

        # Store in vault for next session, honestly report outcome.
        persisted = True
        if self._credential_vault is not None:
            persisted = await self._vault_store(template.service, sender_npub, validated)

        # ACK only if persistence actually worked — telling the patron "got it"
        # when the vault write failed is the root of the lag-and-confusion bugs.
        if persisted:
            self._send_success_dm(sender_npub, target_relay=pinned)

        sensitive_count = sum(
            1 for name in validated if template.fields.get(name, FieldSpec()).sensitive
        )

        return {
            "success": persisted or self._credential_vault is None,
            "service": template.service,
            "fields_received": len(validated),
            "sensitive_fields": sensitive_count,
            "encryption": dm.get("encryption", "unknown"),
            "credentials": validated,
            "persisted": persisted,
            "popped": popped,
            "error_code": None if persisted else "credential_vault_unavailable",
            "error": None if persisted else (
                "Credentials received and validated but the vault write "
                "failed. Retry receive_credentials in 10–15 seconds — the "
                "DM is gone from the relay; the agent must resend."
            ),
            "message": (
                f"Credentials received for {template.service} "
                f"({len(validated)} fields). "
                f"Relay copy deletion requested."
                + (
                    " Credentials stored in vault for future sessions."
                    if persisted and self._credential_vault is not None else
                    " Vault write failed — patron must resend; retry receive_credentials in a moment."
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

    async def redeem_credential_card(
        self, ncred: str, *, service: str | None = None,
    ) -> dict[str, Any]:
        """Redeem an ``ncred1...`` credential card — bypass relay DM flow.

        Decodes the card, verifies signature and expiration, validates
        against the matching template, and stores in vault if configured.

        Args:
            ncred: The ``ncred1...`` bech32 credential card string.
            service: Optional service hint for template matching.
                If omitted, uses the service from the card.

        Returns:
            Dict with same shape as ``receive()`` — success, service,
            field counts, and credentials.

        Raises:
            CourierNotReady: If dependencies are missing.
            CourierValidationError: If card fails validation.
        """
        if not self._enabled:
            raise CourierNotReady(
                "Secure Courier not available. Check logs for missing dependencies."
            )

        try:
            from tollbooth.credential_card import (
                CredentialCardError,
                CredentialCardExpired,
                CredentialCardInvalid,
                decode_credential_card,
            )
        except ImportError:
            raise CourierNotReady(
                "credential_card module not available"
            )

        try:
            decoded = decode_credential_card(ncred, self._nsec)
        except CredentialCardExpired as exc:
            raise CourierValidationError(f"Credential card expired: {exc}") from exc
        except (CredentialCardInvalid, CredentialCardError) as exc:
            raise CourierValidationError(f"Invalid credential card: {exc}") from exc

        card_service = decoded["service"]
        credentials = decoded["credentials"]
        user_npub = decoded["user_npub"]

        # Resolve service: prefer explicit param, fall back to card's service
        resolved = self._resolve_service(service) or card_service

        # Validate against template
        if resolved in self._templates:
            template = self._templates[resolved]
            try:
                credentials = validate_payload(credentials, template)
            except TemplateValidationError as exc:
                raise CourierValidationError(str(exc)) from exc
        elif service is not None:
            # Explicit service requested but no template found
            raise CourierValidationError(
                f"No template found for service '{service}'"
            )

        # Store in vault, honestly report persistence outcome. Pre-0.33.0
        # this was a fire-and-forget that always claimed success — root
        # cause of the ncred state-lag bug class.
        persisted = True
        if self._credential_vault is not None and resolved:
            persisted = await self._vault_store(resolved, user_npub, credentials)

        # Count sensitive fields
        sensitive_count = 0
        if resolved and resolved in self._templates:
            template = self._templates[resolved]
            sensitive_count = sum(
                1 for name in credentials
                if template.fields.get(name, FieldSpec()).sensitive
            )

        return {
            "success": persisted or self._credential_vault is None,
            "service": resolved or card_service,
            "fields_received": len(credentials),
            "sensitive_fields": sensitive_count,
            "encryption": "credential_card",
            "credentials": credentials,
            "user_npub": user_npub,
            "persisted": persisted,
            "error_code": None if persisted else "credential_vault_unavailable",
            "error": None if persisted else (
                "Credential card was valid but the vault write failed. "
                "Retry receive_credentials in 10–15 seconds once the MCP has "
                "warmed up; the ncred is still good."
            ),
            "message": (
                f"Credentials for {resolved or card_service} redeemed from "
                f"credential card ({len(credentials)} fields)."
                + (
                    " Credentials stored in vault for future sessions."
                    if persisted and self._credential_vault is not None else
                    " Vault write failed — retry receive_credentials in a moment."
                    if self._credential_vault is not None else ""
                )
            ),
        }

    # ── Conversational DM helpers ─────────────────────────────────────

    def _send_success_dm(
        self, sender_npub: str, *, target_relay: str | None = None,
    ) -> None:
        """Send a success (ACK) DM to the patron after credential pickup.

        Non-fatal -- DM failure is logged but does not block the receive flow.
        When *target_relay* is set the ACK is published only on that relay
        (the pinned rendezvous relay the patron is already listening on).
        """
        success_text = (
            "\u2705 Your credentials have been securely stored. "
            "Your AI agent can now use them on your behalf.\n\n"
            "This message thread is no longer needed — "
            "the relay copy of your credentials has been deleted."
        )
        try:
            self.send_dm(sender_npub, success_text, target_relay=target_relay)
        except Exception as exc:
            logger.debug(
                "Success DM to %s failed (non-fatal): %s",
                sender_npub[:20], exc,
            )

    def _pop_event(
        self,
        event_id: str,
        reply_npub: str | None = None,
        reason: str | None = None,
        *,
        target_relay: str | None = None,
    ) -> None:
        """Pop a DM from the local queue and relay.

        Unconditionally consumes the event: adds to ``_consumed_ids``,
        removes from ``_received_events``, and sends a NIP-09 deletion
        request to all relays.

        If *reply_npub* and *reason* are provided, sends an acknowledgement
        (NACK) DM to the sender explaining why the message was popped.
        When *target_relay* is set the NACK is published only on that relay.
        Acknowledgement failures are non-fatal.
        """
        if not event_id:
            return

        with self._lock:
            self._consumed_ids.add(event_id)
            self._received_events = [
                e for e in self._received_events
                if e.get("id", "") != event_id
            ]

        self._request_deletion(event_id)

        if reply_npub and reason:
            try:
                self.send_dm(reply_npub, reason, target_relay=target_relay)
            except Exception as exc:
                logger.debug(
                    "Ack DM to %s failed (non-fatal): %s",
                    reply_npub[:20], exc,
                )

    def _send_error_dm(
        self,
        sender_npub: str,
        template: CredentialTemplate | None,
    ) -> None:
        """Send an error DM to the patron when credential parsing fails.

        Non-fatal -- DM failure is logged but does not mask the real error.
        """
        if template is not None:
            instructions = render_delimited_instructions(template)
        else:
            instructions = "(see the welcome message for the expected format)"

        error_text = (
            "\u26a0\ufe0f I couldn\u2019t process your credentials. "
            "The message didn\u2019t match the expected format.\n\n"
            "Please reply using the @@@ format \u2014 paste each value "
            "between the markers:\n\n"
            f"{instructions}\n\n"
            "Copy the template above, paste your values between "
            "the @@@ markers, and send it back."
        )
        try:
            self.send_dm(sender_npub, error_text)
        except Exception as exc:
            logger.debug(
                "Error DM to %s failed (non-fatal): %s",
                sender_npub[:20], exc,
            )

    def _resolve_error_template(
        self, service: str | None,
    ) -> CredentialTemplate | None:
        """Resolve a template for error DM instructions (best-effort)."""
        resolved = self._resolve_service(service)
        if resolved and resolved in self._templates:
            return self._templates[resolved]
        return None

    def _resolve_service(self, service: str | None) -> str | None:
        """Resolve service name, defaulting to the single template if only one."""
        if service and service in self._templates:
            return service
        if len(self._templates) == 1:
            return next(iter(self._templates.keys()))
        return service

    async def _resolve_pinned_record(
        self, sender_npub: str, service: str, poison: str,
    ) -> tuple[str | None, str | None]:
        """Resolve and verify the pinned rendezvous relay for a courier channel.

        Looks up the pending record for ``(sender_npub, service)``, verifies
        the supplied *poison* matches it and has not expired, and returns the
        pinned rendezvous relay the original request committed to. Performs
        cold-start recovery from the vault (restoring the in-memory poison,
        pin, and ephemeral agent key) when the in-memory state is gone after
        a process restart.

        This is the gate for the deterministic Secure Courier retrieve path:
        the client names ``(sender_npub, service, poison)`` and the drain
        targets exactly the one relay returned here — no guessing, no fan-out.

        Returns:
            ``(pinned_relay, None)`` on success, or ``(None, error_code)``
            where ``error_code`` is one of ``ErrorCode.COURIER_NO_PENDING_RECORD``
            / ``COURIER_POISON_MISMATCH`` / ``COURIER_TOKEN_EXPIRED`` /
            ``COURIER_NO_PINNED_RELAY``.
        """
        # Key by the RAW service — open_channel stores the pending record
        # under the exact service string the request used (see open_channel:
        # self._pending_poisons[(recipient_npub, service)] and
        # __pending__{service}). _resolve_service is intentionally NOT applied
        # here: for the proof service it would collapse to a credential
        # template name when only one is configured, breaking the lookup.
        poison_key = (sender_npub, service)

        expected = self._pending_poisons.get(poison_key)
        pinned = self._pinned_relays.get(poison_key)

        # Cold-start recovery: rehydrate from the __pending__ vault blob when
        # in-memory state was lost to a restart (mirrors open_channel's write).
        if (expected is None or pinned is None) and self._credential_vault is not None:
            pending = await self._vault_fetch(
                f"__pending__{service}", sender_npub,
            )
            if pending and "poison" in pending:
                p_expiry = pending.get("expiry", 0)
                if time.time() > p_expiry:
                    try:
                        await self._credential_vault.delete_credentials(
                            f"__pending__{service}", sender_npub,
                        )
                    except Exception:
                        pass
                    return None, ErrorCode.COURIER_TOKEN_EXPIRED
                self._pending_poisons[poison_key] = (pending["poison"], p_expiry)
                expected = self._pending_poisons[poison_key]
                pin_url = pending.get("rendezvous_relay")
                if pin_url:
                    self._pinned_relays[poison_key] = pin_url
                    pinned = pin_url
                agent_nsec_hex = pending.get("agent_nsec_hex")
                if agent_nsec_hex and poison_key not in self._ephemeral_agents:
                    restored_key = PrivateKey(bytes.fromhex(agent_nsec_hex))
                    self._ephemeral_agents[poison_key] = restored_key
                    logger.info(
                        "Cold-start: restored ephemeral agent %s for %s/%s",
                        restored_key.public_key.bech32()[:20],
                        sender_npub[:16], service,
                    )

        if expected is None:
            return None, ErrorCode.COURIER_NO_PENDING_RECORD

        expected_phrase, expiry = expected
        if time.time() > expiry:
            self._pending_poisons.pop(poison_key, None)
            self._pinned_relays.pop(poison_key, None)
            if self._credential_vault is not None:
                try:
                    await self._credential_vault.delete_credentials(
                        f"__pending__{service}", sender_npub,
                    )
                except Exception:
                    pass
            return None, ErrorCode.COURIER_TOKEN_EXPIRED

        if poison != expected_phrase:
            return None, ErrorCode.COURIER_POISON_MISMATCH

        if not pinned:
            return None, ErrorCode.COURIER_NO_PINNED_RELAY

        return pinned, None

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
    ) -> bool:
        """Encrypt and store credentials in the vault.

        Returns True if the write actually reached Neon, False if it failed
        (vault unreachable, schema not provisioned, permission denied, etc.).
        Callers MUST propagate this honestly in their response — same lesson
        as the 0.30.0 `_vault_unavailable` guards for the ledger path:
        telling a caller "stored" when the row isn't there leads to the
        ncred-state-lag class of bugs where `get_patron_onboarding_status`
        keeps reporting "missing" while the user sees "accepted" messaging.
        """
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
            return True
        except Exception as exc:
            logger.warning(
                "Vault credential storage failed for %s/%s: %s: %s. "
                "Caller will surface this as persisted=false; the credential "
                "is NOT in Neon and the agent should retry receive_credentials.",
                service, sender_npub[:16], type(exc).__name__, exc,
            )
            return False

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

    def _subscribe_to_relays(self, relays: list[str] | None = None) -> None:
        """Subscribe to relays for DMs addressed to us.

        Uses multiple REQ filters to cover both NIP-04 (kind 4) and
        NIP-17 (kind 1059) DMs.  NIP-17 gift wraps may use a random
        ``p`` tag for metadata protection (per the NIP-17 spec), so we
        include a broad filter without a ``#p`` constraint — tightly
        limited — to catch those events too.

        Args:
            relays: Specific relays to subscribe to. Defaults to all
                configured relays. The Secure Courier retrieve path passes
                a single pinned rendezvous relay here so the drain touches
                only the relay the original request committed to.
        """
        target_relays = relays if relays is not None else self._relays
        since = int(time.time()) - self._freshness_window

        # Always subscribe to both NIP-04 and NIP-17 events.
        # Even in nip44_only mode we listen for kind 4 so we can
        # surface a clear rejection message during decrypt.

        # Collect pubkeys to listen for: operator + any active ephemeral agents
        listen_pubkeys = [self._pubkey_hex]
        for agent_key in self._ephemeral_agents.values():
            listen_pubkeys.append(agent_key.public_key.hex())

        # Filter 1: NIP-04 DMs addressed to us (p-tag match)
        filter_nip04: dict[str, Any] = {
            "kinds": [_KIND_ENCRYPTED_DM],
            "#p": listen_pubkeys,
            "since": since,
            "limit": 50,
        }
        # Filter 2: NIP-17 gift wraps addressed to us (p-tag match)
        # NIP-17 spec requires the outer gift wrap p-tag to be the real
        # recipient (for relay routing).  A previous broad filter without
        # a #p constraint pulled in wraps addressed to other users, causing
        # ECDH/MAC failures on decrypt.
        filter_giftwrap_tagged: dict[str, Any] = {
            "kinds": [_KIND_GIFT_WRAP],
            "#p": listen_pubkeys,
            "since": since - _TIMESTAMP_FUZZ_SECONDS,  # 48h wider for NIP-17 fuzz
            "limit": 50,
        }

        filters = [filter_nip04, filter_giftwrap_tagged]
        sub_id = f"courier-{int(time.time())}"

        for relay_url in target_relays:
            try:
                self._subscribe_one_relay(relay_url, sub_id, filters)
            except Exception as exc:
                logger.debug(
                    "Relay subscription %s failed (non-fatal): %s",
                    relay_url, exc,
                )

    def _subscribe_one_relay(
        self,
        relay_url: str,
        sub_id: str,
        filters: list[dict[str, Any]] | dict[str, Any],
    ) -> None:
        """Subscribe to one relay and collect events.

        Args:
            relay_url: WebSocket URL of the relay.
            sub_id: Subscription identifier.
            filters: One filter dict or a list of filter dicts.
                The Nostr protocol supports multiple filters in a single
                REQ — events matching ANY filter are returned.
        """
        if isinstance(filters, dict):
            filters = [filters]

        sslopt: dict[str, Any] = {}
        ws = create_connection(relay_url, timeout=_DEFAULT_SUBSCRIBE_TIMEOUT, sslopt=sslopt)
        try:
            # Send REQ with all filters: ["REQ", sub_id, filter1, filter2, ...]
            req_msg = json.dumps(["REQ", sub_id, *filters])
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

    def _fetch_dms_from_relays(self, relays: list[str] | None = None) -> None:
        """One-shot fetch of recent DMs.

        Defaults to all configured relays; pass a single-element list to
        drain only a pinned rendezvous relay.
        """
        self._subscribe_to_relays(relays)

    def _find_dm_candidates(self, sender_hex: str) -> list[dict[str, Any]]:
        """Find unconsumed DM candidates from a sender, newest first.

        NIP-04 events are filtered by sender pubkey.  NIP-17 gift wraps
        cannot be filtered by sender until decrypted (the real sender is
        inside the encrypted seal), so all addressed wraps are included.
        """
        logger.info(
            "_find_dm_candidates: sender=%s, total_events=%d, consumed=%d",
            sender_hex[:16], len(self._received_events),
            len(self._consumed_ids),
        )

        with self._lock:
            candidates = []
            for event in self._received_events:
                event_id = event.get("id", "")
                if event_id in self._consumed_ids:
                    continue

                kind = event.get("kind", 0)

                if kind == _KIND_ENCRYPTED_DM:
                    # NIP-04: sender is event.pubkey
                    if event.get("pubkey", "") == sender_hex:
                        candidates.append(event)

                elif kind == _KIND_GIFT_WRAP:
                    # NIP-17: sender is hidden inside the seal —
                    # include all gift wraps, filter after decryption
                    candidates.append(event)

            # Newest first
            candidates.sort(key=lambda e: e.get("created_at", 0), reverse=True)
            logger.info(
                "_find_dm_candidates: %d candidates from %d unconsumed events",
                len(candidates),
                sum(1 for e in self._received_events
                    if e.get("id", "") not in self._consumed_ids),
            )
            return candidates

    def _decrypt_dm(
        self,
        event: dict[str, Any],
        sender_hex: str,
        *,
        decrypt_privkey_hex: str | None = None,
    ) -> str:
        """Decrypt a DM event (NIP-04 or NIP-17 gift wrap).

        Args:
            event: The raw Nostr event dict.
            sender_hex: Expected sender pubkey (hex).
            decrypt_privkey_hex: If provided, use this private key for
                decryption instead of the operator's key.  Used when an
                ephemeral agent key was used for the channel.

        Returns the plaintext content string.
        """
        kind = event.get("kind", 0)

        if kind == _KIND_GIFT_WRAP:
            return self._unwrap_gift_wrap(
                event, sender_hex,
                decrypt_privkey_hex=decrypt_privkey_hex,
            )

        if kind == _KIND_ENCRYPTED_DM:
            return self._decrypt_nip04_dm(
                event, sender_hex,
                decrypt_privkey_hex=decrypt_privkey_hex,
            )

        raise CourierValidationError(f"Unexpected event kind: {kind}")

    def _decrypt_nip04_dm(
        self,
        event: dict[str, Any],
        sender_hex: str,
        *,
        decrypt_privkey_hex: str | None = None,
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

        privkey_hex = decrypt_privkey_hex or self._privkey_hex
        content = event.get("content", "")
        try:
            plaintext = _nip04_decrypt(content, privkey_hex, sender_hex)
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
        self,
        event: dict[str, Any],
        sender_hex: str,
        *,
        decrypt_privkey_hex: str | None = None,
    ) -> str:
        """Unwrap a NIP-17 gift wrap (kind 1059 → kind 13 → kind 14).

        NIP-17 uses three layers:
        1. Gift wrap (kind 1059): encrypted to recipient with random key
        2. Seal (kind 13): encrypted to recipient with sender's key
        3. Private DM (kind 14): plaintext content

        The gift wrap is encrypted with NIP-44 using a random one-time key,
        so we decrypt with our privkey + the gift wrap's pubkey.

        Args:
            event: The raw gift wrap event dict.
            sender_hex: Expected sender pubkey (hex).
            decrypt_privkey_hex: If provided, use this private key for
                decryption instead of the operator's key.
        """
        if not _HAS_NIP44:
            raise CourierValidationError(
                "NIP-44 decryption not available for gift wrap unwrapping."
            )

        privkey_hex = decrypt_privkey_hex or self._privkey_hex

        # Layer 1: Decrypt gift wrap content with our privkey + wrap pubkey
        wrap_pubkey = event.get("pubkey", "")
        wrap_content = event.get("content", "")

        try:
            seal_json = _nip44_decrypt(
                wrap_content, privkey_hex, wrap_pubkey,
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
                seal.get("content", ""), privkey_hex, seal_pubkey,
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
        # Propagate the real timestamp from the inner DM to the outer event
        # so callers see when the message was actually sent, not the fuzzed
        # gift-wrap timestamp
        inner_created_at = dm.get("created_at")
        if inner_created_at:
            event["created_at"] = inner_created_at
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

    def _publish_to_relays(
        self, message: str,
    ) -> list[tuple[str, bool, str]]:
        """Send an event to all configured relays.

        Returns:
            List of ``(relay_url, accepted, detail)`` tuples.
            *accepted* is ``True`` when the relay returned ``["OK", id, true, ...]``.
        """
        results: list[tuple[str, bool, str]] = []
        sslopt: dict[str, Any] = {}
        for relay_url in self._relays:
            try:
                ws = create_connection(
                    relay_url, timeout=_DEFAULT_SUBSCRIBE_TIMEOUT, sslopt=sslopt,
                )
                try:
                    ws.send(message)
                    raw = ws.recv()
                    try:
                        ack = json.loads(raw)
                        if isinstance(ack, list) and len(ack) >= 3 and ack[0] == "OK":
                            ok = bool(ack[2])
                            detail = ack[3] if len(ack) > 3 else ""
                            results.append((relay_url, ok, detail))
                        else:
                            # Non-OK response (e.g., NOTICE) — treat as accepted
                            results.append((relay_url, True, str(raw)))
                    except (json.JSONDecodeError, IndexError):
                        results.append((relay_url, True, str(raw)))
                finally:
                    ws.close()
            except Exception as exc:
                logger.debug(
                    "Relay publish %s failed (non-fatal): %s", relay_url, exc,
                )
                results.append((relay_url, False, str(exc)))
        return results

    def _publish_to_one_relay(
        self, message: str, relay_url: str,
    ) -> tuple[bool, str]:
        """Send an event to exactly one relay.

        Used by the rendezvous-pinning publish path: the courier commits
        the first relay that accepts the challenge as the per-conversation
        rendezvous and embeds it in the DM body before publishing.

        Returns:
            ``(accepted, detail)`` — ``accepted`` is True when the relay
            returned ``["OK", id, true, ...]`` (or any non-explicit
            rejection).
        """
        sslopt: dict[str, Any] = {}
        try:
            ws = create_connection(
                relay_url, timeout=_DEFAULT_SUBSCRIBE_TIMEOUT, sslopt=sslopt,
            )
            try:
                ws.send(message)
                raw = ws.recv()
                try:
                    ack = json.loads(raw)
                    if isinstance(ack, list) and len(ack) >= 3 and ack[0] == "OK":
                        ok = bool(ack[2])
                        detail = ack[3] if len(ack) > 3 else ""
                        return ok, detail
                    return True, str(raw)
                except (json.JSONDecodeError, IndexError):
                    return True, str(raw)
            finally:
                ws.close()
        except Exception as exc:
            logger.debug(
                "Single-relay publish %s failed (non-fatal): %s", relay_url, exc,
            )
            return False, str(exc)
