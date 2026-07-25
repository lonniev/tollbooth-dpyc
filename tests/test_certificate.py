"""Tests for shared certificate infrastructure: JTI store, error class, and auto-verify routing."""

import json
import time

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.certificate import (
    CertificateError,
    _JTIStore,
    reset_jti_store,
    verify_certificate_auto,
)
from tollbooth.nostr_certificate import NOSTR_CERT_KIND, NOSTR_CERT_LABEL, NOSTR_CERT_TAG

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_jti_store():
    """Reset the JTI store before each test."""
    reset_jti_store()
    yield
    reset_jti_store()


@pytest.fixture()
def nostr_keypair():
    """Generate a Nostr keypair for testing."""
    private_key = PrivateKey()
    npub = private_key.public_key.bech32()
    return private_key, npub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_jti_counter = 0


def _sign_nostr_certificate(
    private_key: PrivateKey,
    *,
    operator_id: str = "npub1operator",
    amount_sats: int = 1000,
    fee_sats: int = 20,
    net_sats: int = 980,
    jti: str | None = None,
    exp_offset: int = 600,
) -> str:
    """Sign a test Nostr certificate event. Returns JSON string."""
    global _jti_counter
    if jti is None:
        _jti_counter += 1
        jti = f"cert-jti-{_jti_counter}-{time.time_ns()}"

    claims = {
        "sub": operator_id,
        "amount_sats": amount_sats,
        "fee_sats": fee_sats,
        "net_sats": net_sats,
        "dpyc_protocol": "dpyp-01-base-certificate",
    }

    expiration = int(time.time()) + exp_offset
    tags: list[list[str]] = [
        ["d", jti],
        ["p", "deadbeef" * 8],
        ["t", NOSTR_CERT_TAG],
        ["L", NOSTR_CERT_LABEL],
        ["expiration", str(expiration)],
    ]

    event = Event(
        kind=NOSTR_CERT_KIND,
        content=json.dumps(claims),
        tags=tags,
        pubkey=private_key.public_key.hex(),
        created_at=int(time.time()),
    )
    event.sign(private_key.hex())
    return json.dumps(event.to_dict())


# ---------------------------------------------------------------------------
# _JTIStore
# ---------------------------------------------------------------------------


class TestJTIStore:
    def test_check_and_record_new(self):
        """New JTI is accepted."""
        store = _JTIStore()
        assert store.check_and_record("jti-1", time.time() + 600) is True

    def test_check_and_record_replay(self):
        """Same JTI is rejected (replay)."""
        store = _JTIStore()
        store.check_and_record("jti-1", time.time() + 600)
        assert store.check_and_record("jti-1", time.time() + 600) is False

    def test_cleanup_expired(self):
        """Expired JTIs are cleaned up and can be re-used."""
        store = _JTIStore()
        # Record with expiry in the past
        store.check_and_record("jti-old", time.time() - 10)
        # After cleanup, the JTI should be accepted again
        assert store.check_and_record("jti-old", time.time() + 600) is True

    def test_different_jtis_accepted(self):
        """Different JTIs are all accepted."""
        store = _JTIStore()
        assert store.check_and_record("jti-a", time.time() + 600) is True
        assert store.check_and_record("jti-b", time.time() + 600) is True
        assert store.check_and_record("jti-c", time.time() + 600) is True


# ---------------------------------------------------------------------------
# reset_jti_store
# ---------------------------------------------------------------------------


class TestResetJTIStore:
    def test_reset_clears_store(self, nostr_keypair):
        """After reset, previously seen JTIs are forgotten."""
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="jti-reset-test")
        verify_certificate_auto(event_json, authority_npub=npub)

        # Reset and re-verify the same JTI — should succeed after reset
        reset_jti_store()
        event_json2 = _sign_nostr_certificate(private_key, jti="jti-reset-test")
        result = verify_certificate_auto(event_json2, authority_npub=npub)
        assert result["jti"] == "jti-reset-test"


# ---------------------------------------------------------------------------
# CertificateError
# ---------------------------------------------------------------------------


class TestCertificateError:
    def test_can_be_raised(self):
        with pytest.raises(CertificateError, match="test error"):
            raise CertificateError("test error")

    def test_is_exception(self):
        assert issubclass(CertificateError, Exception)


# ---------------------------------------------------------------------------
# verify_certificate_auto — Nostr routing
# ---------------------------------------------------------------------------


class TestVerifyCertificateAuto:
    def test_nostr_event_verified(self, nostr_keypair):
        """Nostr event JSON is verified successfully."""
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="auto-nostr-1")
        result = verify_certificate_auto(event_json, authority_npub=npub)
        assert result["jti"] == "auto-nostr-1"
        assert result["operator_id"] == "npub1operator"
        assert result["net_sats"] == 980

    def test_no_authority_npub_fails(self, nostr_keypair):
        """Missing authority_npub raises CertificateError."""
        private_key, _ = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="auto-no-npub")
        with pytest.raises(CertificateError, match="No authority_npub configured"):
            verify_certificate_auto(event_json, authority_npub="")

    def test_empty_string_npub_fails(self, nostr_keypair):
        """Empty string authority_npub raises CertificateError."""
        private_key, _ = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="auto-empty-npub")
        with pytest.raises(CertificateError, match="No authority_npub configured"):
            verify_certificate_auto(event_json)

    def test_wrong_signer_fails(self, nostr_keypair):
        """Certificate signed by wrong key is rejected."""
        _, npub = nostr_keypair
        other_key = PrivateKey()
        event_json = _sign_nostr_certificate(other_key, jti="auto-wrong-signer")
        with pytest.raises(CertificateError, match="not signed by registered Authority"):
            verify_certificate_auto(event_json, authority_npub=npub)

    def test_replay_detected(self, nostr_keypair):
        """Same JTI is rejected on second use."""
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="auto-replay")
        verify_certificate_auto(event_json, authority_npub=npub)
        event_json2 = _sign_nostr_certificate(private_key, jti="auto-replay")
        with pytest.raises(CertificateError, match="replay"):
            verify_certificate_auto(event_json2, authority_npub=npub)
