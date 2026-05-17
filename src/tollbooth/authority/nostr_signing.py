"""Nostr event signing for certificates — Schnorr/BIP-340 via pynostr.

Signs kind 30079 parameterized replaceable events that serve as
Authority certificates. The Schnorr-native replacement for the
the previous Ed25519 JWT signing module (removed).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey, PublicKey  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

NOSTR_CERT_KIND = 30079
"""NIP-33 parameterized replaceable event kind for tollbooth certificates."""


def _npub_to_hex(npub: str) -> str:
    """Convert a bech32 npub to hex pubkey string."""
    return PublicKey.from_npub(npub).hex()


class AuthorityNostrSigner:
    """Signs certificate events using a Nostr private key (nsec)."""

    def __init__(self, nsec: str) -> None:
        """Load Nostr key from nsec bech32 string."""
        self._private_key = PrivateKey.from_nsec(nsec)
        self._pubkey_hex: str = self._private_key.public_key.hex()

    @property
    def nsec(self) -> str:
        """Authority's private key as hex string."""
        return self._private_key.hex()

    @property
    def npub(self) -> str:
        """Authority's public key in bech32 npub format."""
        return self._private_key.public_key.bech32()

    @property
    def pubkey_hex(self) -> str:
        """Authority's public key in hex format."""
        return self._pubkey_hex

    def sign_certificate_event(
        self,
        claims: dict[str, Any],
        jti: str,
        operator_npub: str,
        expiration: int,
    ) -> str:
        """Build and sign a kind 30079 Nostr event.

        Args:
            claims: Certificate claims dict (sub, amount_sats, fee_sats,
                net_sats, dpyc_protocol). Serialized as the event content.
            jti: Unique certificate ID (goes in the d-tag for NIP-33).
            operator_npub: Operator's npub (goes in the p-tag).
            expiration: Unix timestamp for NIP-40 expiration tag.

        Returns:
            JSON string of the signed event.
        """
        # Build content from claims (only the fields the verifier needs)
        content_claims = {
            "sub": claims.get("sub", ""),
            "amount_sats": claims.get("amount_sats", 0),
            "fee_sats": claims.get("fee_sats", 0),
            "net_sats": claims.get("net_sats", 0),
            "dpyc_protocol": claims.get("dpyc_protocol", ""),
        }

        # Convert operator npub to hex for p-tag (best effort)
        try:
            operator_hex = _npub_to_hex(operator_npub)
        except Exception:
            operator_hex = operator_npub  # fallback: use raw value

        event = Event(
            kind=NOSTR_CERT_KIND,
            content=json.dumps(content_claims, separators=(",", ":")),
            tags=[
                ["d", jti],
                ["p", operator_hex],
                ["t", "tollbooth-cert"],
                ["L", "dpyc.tollbooth"],
                ["expiration", str(expiration)],
            ],
            pubkey=self._pubkey_hex,
            created_at=int(time.time()),
        )
        event.sign(self._private_key.hex())

        return json.dumps(event.to_dict())
