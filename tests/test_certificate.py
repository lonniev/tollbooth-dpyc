"""Tests for Authority certificate verification: Ed25519 JWT validation and anti-replay."""

import time
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from tollbooth.certificate import CertificateError, verify_certificate, reset_jti_store
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
def keypair():
    """Generate an Ed25519 keypair for testing."""
    private_key = Ed25519PrivateKey.generate()
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_key, public_pem


def _sign_certificate(
    private_key: Ed25519PrivateKey,
    *,
    operator_id: str = "op-1",
    amount_sats: int = 1000,
    tax_paid_sats: int = 20,
    net_sats: int = 980,
    jti: str = "jti-unique-1",
    exp_offset: int = 600,
    extra_claims: dict | None = None,
) -> str:
    """Sign a test certificate JWT."""
    claims = {
        "operator_id": operator_id,
        "amount_sats": amount_sats,
        "tax_paid_sats": tax_paid_sats,
        "net_sats": net_sats,
        "jti": jti,
        "iat": int(time.time()),
        "exp": int(time.time()) + exp_offset,
    }
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, private_key, algorithm="EdDSA")


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
# verify_certificate — valid
# ---------------------------------------------------------------------------


class TestVerifyCertificateValid:
    def test_valid_certificate(self, keypair):
        private_key, public_pem = keypair
        token = _sign_certificate(private_key)
        result = verify_certificate(token, public_pem)
        assert result["operator_id"] == "op-1"
        assert result["amount_sats"] == 1000
        assert result["tax_paid_sats"] == 20
        assert result["net_sats"] == 980
        assert result["jti"] == "jti-unique-1"

    def test_bare_base64_key_accepted(self, keypair):
        """Bare base64 key (no PEM headers) works for verification."""
        private_key, public_pem = keypair
        # Strip PEM headers to get bare base64
        lines = [ln for ln in public_pem.strip().splitlines() if not ln.startswith("-----")]
        bare_b64 = "".join(lines).strip()
        token = _sign_certificate(private_key, jti="jti-bare-b64")
        result = verify_certificate(token, bare_b64)
        assert result["operator_id"] == "op-1"
        assert result["jti"] == "jti-bare-b64"

    def test_extracts_all_claims(self, keypair):
        private_key, public_pem = keypair
        token = _sign_certificate(
            private_key,
            operator_id="op-42",
            amount_sats=5000,
            tax_paid_sats=100,
            net_sats=4900,
            jti="jti-42",
        )
        result = verify_certificate(token, public_pem)
        assert result["operator_id"] == "op-42"
        assert result["amount_sats"] == 5000
        assert result["net_sats"] == 4900


# ---------------------------------------------------------------------------
# verify_certificate — invalid
# ---------------------------------------------------------------------------


class TestVerifyCertificateInvalid:
    def test_expired_certificate(self, keypair):
        private_key, public_pem = keypair
        token = _sign_certificate(private_key, exp_offset=-10)
        with pytest.raises(CertificateError, match="expired"):
            verify_certificate(token, public_pem)

    def test_tampered_certificate(self, keypair):
        private_key, public_pem = keypair
        token = _sign_certificate(private_key)
        # Flip a character in the signature portion
        parts = token.split(".")
        sig = list(parts[2])
        sig[0] = "A" if sig[0] != "A" else "B"
        parts[2] = "".join(sig)
        tampered = ".".join(parts)
        with pytest.raises(CertificateError, match="invalid|decoded"):
            verify_certificate(tampered, public_pem)

    def test_wrong_key(self, keypair):
        private_key, public_pem = keypair
        # Sign with a different key
        other_key = Ed25519PrivateKey.generate()
        token = _sign_certificate(other_key)
        with pytest.raises(CertificateError, match="invalid|signature"):
            verify_certificate(token, public_pem)

    def test_garbage_token(self, keypair):
        _, public_pem = keypair
        with pytest.raises(CertificateError, match="decoded|Invalid"):
            verify_certificate("not.a.jwt", public_pem)

    def test_invalid_public_key(self):
        with pytest.raises(CertificateError, match="Invalid authority public key"):
            verify_certificate("some.jwt.token", "not a valid pem")

    def test_missing_jti(self, keypair):
        private_key, public_pem = keypair
        claims = {
            "operator_id": "op-1",
            "amount_sats": 1000,
            "net_sats": 980,
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
        }
        token = jwt.encode(claims, private_key, algorithm="EdDSA")
        with pytest.raises(CertificateError, match="missing jti"):
            verify_certificate(token, public_pem)


# ---------------------------------------------------------------------------
# Anti-replay (JTI)
# ---------------------------------------------------------------------------


class TestAntiReplay:
    def test_duplicate_jti_rejected(self, keypair):
        private_key, public_pem = keypair
        token = _sign_certificate(private_key, jti="jti-dup")
        # First call succeeds
        verify_certificate(token, public_pem)
        # Second call with same JTI — replay
        token2 = _sign_certificate(private_key, jti="jti-dup")
        with pytest.raises(CertificateError, match="replay"):
            verify_certificate(token2, public_pem)

    def test_different_jti_accepted(self, keypair):
        private_key, public_pem = keypair
        token1 = _sign_certificate(private_key, jti="jti-a")
        token2 = _sign_certificate(private_key, jti="jti-b")
        verify_certificate(token1, public_pem)
        verify_certificate(token2, public_pem)  # should not raise


# ---------------------------------------------------------------------------
# purchase_credits_tool with certificate
# ---------------------------------------------------------------------------


class TestPurchaseWithCertificate:
    @pytest.mark.asyncio
    async def test_missing_certificate_rejected(self, keypair):
        _, public_pem = keypair
        result = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate="",
            authority_public_key=public_pem,
        )
        assert result["success"] is False
        assert "certificate is required" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_certificate_rejected(self, keypair):
        _, public_pem = keypair
        result = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate="bad.jwt.token",
            authority_public_key=public_pem,
        )
        assert result["success"] is False
        assert "Certificate rejected" in result["error"]

    @pytest.mark.asyncio
    async def test_valid_certificate_creates_invoice(self, keypair):
        private_key, public_pem = keypair
        token = _sign_certificate(private_key, net_sats=980)
        result = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate=token,
            authority_public_key=public_pem,
        )
        assert result["success"] is True
        assert result["amount_sats"] == 980  # from certificate net_sats
        assert result["certificate_jti"] == "jti-unique-1"

    @pytest.mark.asyncio
    async def test_certificate_net_sats_overrides_amount(self, keypair):
        private_key, public_pem = keypair
        token = _sign_certificate(private_key, net_sats=500, jti="jti-override")
        btcpay = _mock_btcpay()
        result = await purchase_credits_tool(
            btcpay=btcpay,
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=9999,  # should be overridden by cert's net_sats
            certificate=token,
            authority_public_key=public_pem,
        )
        assert result["success"] is True
        assert result["amount_sats"] == 500
        btcpay.create_invoice.assert_called_once()
        call_args = btcpay.create_invoice.call_args
        assert call_args[0][0] == 500

    @pytest.mark.asyncio
    async def test_certificate_jti_in_invoice_metadata(self, keypair):
        private_key, public_pem = keypair
        token = _sign_certificate(private_key, jti="jti-meta-test")
        btcpay = _mock_btcpay()
        await purchase_credits_tool(
            btcpay=btcpay,
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate=token,
            authority_public_key=public_pem,
        )
        call_kwargs = btcpay.create_invoice.call_args[1]
        assert call_kwargs["metadata"]["certificate_jti"] == "jti-meta-test"

    @pytest.mark.asyncio
    async def test_replay_rejected_in_purchase(self, keypair):
        private_key, public_pem = keypair
        token = _sign_certificate(private_key, jti="jti-replay-purchase")
        result1 = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate=token,
            authority_public_key=public_pem,
        )
        assert result1["success"] is True
        # Replayed certificate fails
        token2 = _sign_certificate(private_key, jti="jti-replay-purchase")
        result2 = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate=token2,
            authority_public_key=public_pem,
        )
        assert result2["success"] is False
        assert "replay" in result2["error"].lower()


# ---------------------------------------------------------------------------
# Mandatory trust — no untrusted operation
# ---------------------------------------------------------------------------


class TestMandatoryTrust:
    @pytest.mark.asyncio
    async def test_no_public_key_rejects_purchase(self):
        """Operators cannot operate without a trusted Authority."""
        result = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate="some.jwt.token",
            authority_public_key="",
        )
        assert result["success"] is False
        assert "authority_public_key is required" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_certificate_and_empty_key(self):
        """Both missing — operator misconfigured."""
        result = await purchase_credits_tool(
            btcpay=_mock_btcpay(),
            cache=_mock_cache(),
            user_id="user-1",
            amount_sats=1000,
            certificate="",
            authority_public_key="",
        )
        assert result["success"] is False
        assert "authority_public_key is required" in result["error"]
