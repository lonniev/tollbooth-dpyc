"""Tests for tollbooth.authority.nostr_signing (audit M2.4).

The Authority signs kind-30079 certificate events with Schnorr/BIP-340. A real
signing round-trip (sign → verify) is the core security guarantee. Was ~59%
covered (the sign path untested).
"""

from __future__ import annotations

import json

from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey, PublicKey  # type: ignore[import-untyped]

from tollbooth.authority.nostr_signing import (
    NOSTR_CERT_KIND,
    AuthorityNostrSigner,
    _npub_to_hex,
)


def test_signer_properties_match_key():
    pk = PrivateKey()
    signer = AuthorityNostrSigner(pk.nsec)
    assert signer.npub == pk.public_key.bech32()
    assert signer.pubkey_hex == pk.public_key.hex()
    assert signer.nsec == pk.hex()


def test_npub_to_hex_roundtrip():
    pub = PrivateKey().public_key
    assert _npub_to_hex(pub.bech32()) == pub.hex()


def test_sign_certificate_event_verifies_and_carries_claims():
    signer = AuthorityNostrSigner(PrivateKey().nsec)
    operator_npub = PrivateKey().public_key.bech32()

    signed = signer.sign_certificate_event(
        claims={
            "sub": "npub1patron", "amount_sats": 1000, "fee_sats": 20,
            "net_sats": 980, "dpyc_protocol": "dpyp-01-base-certificate",
            "ignored_extra": "dropped",
        },
        jti="cert-abc-123",
        operator_npub=operator_npub,
        expiration=2_000_000_000,
    )

    event = Event.from_dict(json.loads(signed))
    # Schnorr signature verifies and is signed by the Authority key
    assert event.verify() is True
    assert event.kind == NOSTR_CERT_KIND == 30079
    assert event.pubkey == signer.pubkey_hex

    tags = {t[0]: t[1] for t in event.tags}
    assert tags["d"] == "cert-abc-123"                 # NIP-33 d-tag = jti
    assert tags["expiration"] == "2000000000"          # NIP-40
    assert tags["t"] == "tollbooth-cert"
    assert tags["p"] == PublicKey.from_npub(operator_npub).hex()

    content = json.loads(event.content)
    assert content == {
        "sub": "npub1patron", "amount_sats": 1000, "fee_sats": 20,
        "net_sats": 980, "dpyc_protocol": "dpyp-01-base-certificate",
    }
    assert "ignored_extra" not in content              # only verifier fields kept


def test_sign_defaults_missing_claims_to_zero_and_empty():
    signer = AuthorityNostrSigner(PrivateKey().nsec)
    signed = signer.sign_certificate_event(
        claims={}, jti="j", operator_npub=PrivateKey().public_key.bech32(),
        expiration=1,
    )
    content = json.loads(Event.from_dict(json.loads(signed)).content)
    assert content == {"sub": "", "amount_sats": 0, "fee_sats": 0, "net_sats": 0, "dpyc_protocol": ""}


def test_sign_falls_back_to_raw_operator_value_on_bad_npub():
    signer = AuthorityNostrSigner(PrivateKey().nsec)
    signed = signer.sign_certificate_event(
        claims={}, jti="j", operator_npub="not-a-valid-npub", expiration=1,
    )
    tags = {t[0]: t[1] for t in Event.from_dict(json.loads(signed)).tags}
    assert tags["p"] == "not-a-valid-npub"   # raw fallback when conversion fails
