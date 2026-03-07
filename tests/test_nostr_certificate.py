"""Tests for Nostr event certificate verification: Schnorr/BIP-340 validation and anti-replay."""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.certificate import CertificateError, reset_jti_store, verify_certificate_auto
from tollbooth.nostr_certificate import (
    NOSTR_CERT_KIND,
    NOSTR_CERT_LABEL,
    NOSTR_CERT_TAG,
    verify_nostr_certificate,
)
from tollbooth.ledger import UserLedger
from tollbooth.ledger_cache import LedgerCache
from tollbooth.btcpay_client import BTCPayClient
from tollbooth.tools.credits import purchase_credits_tool


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
    kind: int = NOSTR_CERT_KIND,
    extra_tags: list[list[str]] | None = None,
    content_override: str | None = None,
    include_expiration: bool = True,
    include_d_tag: bool = True,
) -> str:
    """Sign a test Nostr certificate event. Returns JSON string."""
    global _jti_counter
    if jti is None:
        _jti_counter += 1
        jti = f"nostr-jti-{_jti_counter}-{time.time_ns()}"

    claims = {
        "sub": operator_id,
        "amount_sats": amount_sats,
        "fee_sats": fee_sats,
        "net_sats": net_sats,
        "dpyc_protocol": "dpyp-01-base-certificate",
    }

    content = content_override if content_override is not None else json.dumps(claims)
    expiration = int(time.time()) + exp_offset

    tags: list[list[str]] = []
    if include_d_tag:
        tags.append(["d", jti])
    tags.append(["p", "deadbeef" * 8])
    tags.append(["t", NOSTR_CERT_TAG])
    tags.append(["L", NOSTR_CERT_LABEL])
    if include_expiration:
        tags.append(["expiration", str(expiration)])
    if extra_tags:
        tags.extend(extra_tags)

    event = Event(
        kind=kind,
        content=content,
        tags=tags,
        pubkey=private_key.public_key.hex(),
        created_at=int(time.time()),
    )
    event.sign(private_key.hex())

    return json.dumps(event.to_dict())


def _mock_btcpay(invoice_response: dict | None = None):
    client = AsyncMock(spec=BTCPayClient)
    resp = invoice_response or {"id": "inv-1", "checkoutLink": "https://pay.example.com/inv-1"}
    client.create_invoice = AsyncMock(return_value=resp)
    return client


def _mock_cache(ledger: UserLedger | None = None):
    cache = AsyncMock(spec=LedgerCache)
    cache.get = AsyncMock(return_value=ledger or UserLedger())
    cache.mark_dirty = MagicMock()
    return cache


# ---------------------------------------------------------------------------
# verify_nostr_certificate — valid
# ---------------------------------------------------------------------------


class TestVerifyNostrCertificateValid:
    def test_valid_certificate(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="jti-valid-1")
        result = verify_nostr_certificate(event_json, npub)
        assert result["operator_id"] == "npub1operator"
        assert result["amount_sats"] == 1000
        assert result["fee_sats"] == 20
        assert result["net_sats"] == 980
        assert result["jti"] == "jti-valid-1"
        assert result["dpyc_protocol"] == "dpyp-01-base-certificate"

    def test_extracts_all_claims(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(
            private_key,
            operator_id="npub1custom",
            amount_sats=5000,
            fee_sats=100,
            net_sats=4900,
            jti="jti-custom-1",
        )
        result = verify_nostr_certificate(event_json, npub)
        assert result["operator_id"] == "npub1custom"
        assert result["amount_sats"] == 5000
        assert result["net_sats"] == 4900


# ---------------------------------------------------------------------------
# verify_nostr_certificate — invalid
# ---------------------------------------------------------------------------


class TestVerifyNostrCertificateInvalid:
    def test_expired_certificate(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, exp_offset=-10, jti="jti-expired")
        with pytest.raises(CertificateError, match="expired"):
            verify_nostr_certificate(event_json, npub)

    def test_wrong_signer(self, nostr_keypair):
        private_key, npub = nostr_keypair
        other_key = PrivateKey()
        event_json = _sign_nostr_certificate(other_key, jti="jti-wrong-signer")
        with pytest.raises(CertificateError, match="not signed by registered Authority"):
            verify_nostr_certificate(event_json, npub)

    def test_tampered_content(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="jti-tampered")
        # Tamper with content after signing
        event_dict = json.loads(event_json)
        event_dict["content"] = '{"sub":"hacker","amount_sats":999999}'
        tampered_json = json.dumps(event_dict)
        with pytest.raises(CertificateError, match="invalid|failed"):
            verify_nostr_certificate(tampered_json, npub)

    def test_wrong_kind(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, kind=1, jti="jti-wrong-kind")
        with pytest.raises(CertificateError, match="Not a tollbooth certificate"):
            verify_nostr_certificate(event_json, npub)

    def test_missing_expiration_tag(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(
            private_key, include_expiration=False, jti="jti-no-exp",
        )
        with pytest.raises(CertificateError, match="missing expiration"):
            verify_nostr_certificate(event_json, npub)

    def test_missing_d_tag(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(
            private_key, include_d_tag=False, jti="jti-no-d-tag",
        )
        with pytest.raises(CertificateError, match="missing d-tag"):
            verify_nostr_certificate(event_json, npub)

    def test_invalid_json(self, nostr_keypair):
        _, npub = nostr_keypair
        with pytest.raises(CertificateError, match="not valid JSON"):
            verify_nostr_certificate("not json at all", npub)

    def test_invalid_content_json(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(
            private_key, content_override="not json", jti="jti-bad-content",
        )
        with pytest.raises(CertificateError, match="content is not valid JSON"):
            verify_nostr_certificate(event_json, npub)

    def test_invalid_authority_npub(self, nostr_keypair):
        private_key, _ = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="jti-bad-npub")
        with pytest.raises(CertificateError, match="Invalid authority npub"):
            verify_nostr_certificate(event_json, "not-an-npub")


# ---------------------------------------------------------------------------
# Anti-replay (JTI via d-tag)
# ---------------------------------------------------------------------------


class TestNostrAntiReplay:
    def test_duplicate_jti_rejected(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="jti-nostr-dup")
        verify_nostr_certificate(event_json, npub)
        # Same JTI again — replay
        event_json2 = _sign_nostr_certificate(private_key, jti="jti-nostr-dup")
        with pytest.raises(CertificateError, match="replay"):
            verify_nostr_certificate(event_json2, npub)

    def test_different_jti_accepted(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json1 = _sign_nostr_certificate(private_key, jti="jti-nostr-a")
        event_json2 = _sign_nostr_certificate(private_key, jti="jti-nostr-b")
        verify_nostr_certificate(event_json1, npub)
        verify_nostr_certificate(event_json2, npub)  # should not raise


# ---------------------------------------------------------------------------
# Protocol versioning
# ---------------------------------------------------------------------------


class TestNostrProtocolVersioning:
    def test_missing_protocol_rejected(self, nostr_keypair):
        private_key, npub = nostr_keypair
        content = json.dumps({"sub": "op-1", "amount_sats": 1000, "net_sats": 980})
        event_json = _sign_nostr_certificate(
            private_key, content_override=content, jti="jti-no-proto",
        )
        with pytest.raises(CertificateError, match="missing dpyc_protocol"):
            verify_nostr_certificate(event_json, npub)

    def test_unknown_protocol_rejected(self, nostr_keypair):
        private_key, npub = nostr_keypair
        content = json.dumps({
            "sub": "op-1",
            "amount_sats": 1000,
            "net_sats": 980,
            "dpyc_protocol": "dpyp-99-future",
        })
        event_json = _sign_nostr_certificate(
            private_key, content_override=content, jti="jti-future-proto",
        )
        with pytest.raises(CertificateError, match="Unsupported protocol"):
            verify_nostr_certificate(event_json, npub)


# ---------------------------------------------------------------------------
# verify_certificate_auto — format auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_nostr_event_routed_to_nostr_verifier(self, nostr_keypair):
        """Nostr event JSON is verified via verify_certificate_auto."""
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="auto-nostr-1")
        result = verify_certificate_auto(event_json, authority_npub=npub)
        assert result["jti"] == "auto-nostr-1"

    def test_nostr_without_npub_fails(self, nostr_keypair):
        """No npub configured raises CertificateError."""
        private_key, _ = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, jti="auto-no-npub")
        with pytest.raises(CertificateError, match="No authority_npub"):
            verify_certificate_auto(event_json)


# ---------------------------------------------------------------------------
# purchase_credits_tool with Nostr certificate
# ---------------------------------------------------------------------------


class TestPurchaseWithNostrCertificate:
    @pytest.mark.asyncio
    async def test_valid_nostr_certificate_creates_invoice(self, nostr_keypair):
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, net_sats=980)
        result = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate=event_json,
            authority_npub=npub,
        )
        assert result["success"] is True
        assert result["amount_sats"] == 1000  # full amount_sats, not net_sats

    @pytest.mark.asyncio
    async def test_npub_only_config_works(self, nostr_keypair):
        """Operator configured with only authority_npub (no public_key) accepts Nostr certs."""
        private_key, npub = nostr_keypair
        event_json = _sign_nostr_certificate(private_key, net_sats=500, jti="npub-only")
        result = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate=event_json,
            authority_npub=npub,
        )
        assert result["success"] is True
        assert result["amount_sats"] == 1000  # full amount_sats, not net_sats

    @pytest.mark.asyncio
    async def test_invalid_nostr_certificate_rejected(self, nostr_keypair):
        _, npub = nostr_keypair
        other_key = PrivateKey()
        event_json = _sign_nostr_certificate(other_key, jti="wrong-signer-purchase")
        result = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate=event_json,
            authority_npub=npub,
        )
        assert result["success"] is False
        assert "Certificate rejected" in result["error"]
