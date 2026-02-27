"""Tests for the SecureCourierService high-level wrapper."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.nostr_credentials import (
    CourierNotReady,
    NostrCredentialExchange,
    NostrProfile,
)
from tollbooth.secure_courier import SecureCourierService


# ── Fixtures ──────────────────────────────────────────────────────────


def _test_templates() -> dict[str, CredentialTemplate]:
    """Single-service test template."""
    return {
        "x": CredentialTemplate(
            service="x",
            version=1,
            fields={
                "access_token": FieldSpec(required=True, sensitive=True),
                "access_token_secret": FieldSpec(required=True, sensitive=True),
            },
            description="Test X API credentials",
        ),
    }


@pytest.fixture
def operator_nsec():
    return PrivateKey().nsec


@pytest.fixture
def relays():
    return ["wss://relay.test.com"]


@pytest.fixture
def templates():
    return _test_templates()


def _make_vault():
    """Create a mock CredentialVaultBackend."""
    vault = AsyncMock()
    vault.store_credentials = AsyncMock()
    vault.fetch_credentials = AsyncMock(return_value=None)
    vault.delete_credentials = AsyncMock(return_value=False)
    return vault


# ── Initialization tests ──────────────────────────────────────────────


class TestServiceInit:
    def test_init_with_all_options(self, operator_nsec, relays, templates):
        """Service initialises with all optional params provided."""
        vault = _make_vault()
        profile = NostrProfile(name="test-service")
        callback = AsyncMock(return_value=None)

        with patch.object(NostrCredentialExchange, "publish_profile"):
            service = SecureCourierService(
                operator_nsec=operator_nsec,
                relays=relays,
                templates=templates,
                credential_vault=vault,
                profile=profile,
                on_credentials_received=callback,
            )

        assert service.enabled is True
        assert service.npub.startswith("npub1")
        assert service.relays == relays

    def test_init_without_optional_params(self, operator_nsec, relays, templates):
        """Service initialises fine without vault, profile, or callback."""
        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
        )

        assert service.enabled is True
        assert service.npub.startswith("npub1")

    def test_profile_published_on_init(self, operator_nsec, relays, templates):
        """Profile is published when provided."""
        profile = NostrProfile(name="test-operator")

        with patch.object(
            NostrCredentialExchange, "publish_profile"
        ) as mock_publish:
            SecureCourierService(
                operator_nsec=operator_nsec,
                relays=relays,
                templates=templates,
                profile=profile,
            )

        mock_publish.assert_called_once_with(profile)

    def test_profile_not_published_when_none(self, operator_nsec, relays, templates):
        """Profile is NOT published when not provided."""
        with patch.object(
            NostrCredentialExchange, "publish_profile"
        ) as mock_publish:
            SecureCourierService(
                operator_nsec=operator_nsec,
                relays=relays,
                templates=templates,
                profile=None,
            )

        mock_publish.assert_not_called()

    def test_exchange_property_exposed(self, operator_nsec, relays, templates):
        """The underlying exchange is accessible."""
        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
        )
        assert isinstance(service.exchange, NostrCredentialExchange)


# ── Missing deps graceful degradation ─────────────────────────────────


class TestMissingDeps:
    def test_raises_runtime_error_when_courier_unavailable(
        self, operator_nsec, relays, templates,
    ):
        """Graceful error when nostr dependencies are not installed."""
        with patch("tollbooth.secure_courier._HAS_COURIER", False):
            with pytest.raises(RuntimeError, match="dependencies not installed"):
                SecureCourierService(
                    operator_nsec=operator_nsec,
                    relays=relays,
                    templates=templates,
                )


# ── open_channel tests ────────────────────────────────────────────────


class TestOpenChannel:
    @pytest.mark.asyncio
    async def test_delegates_to_exchange(self, operator_nsec, relays, templates):
        """open_channel delegates to the underlying exchange."""
        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
        )

        expected = {
            "success": True,
            "npub": service.npub,
            "service": "x",
        }

        with patch.object(
            service.exchange, "open_channel",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_open:
            result = await service.open_channel("x")

        mock_open.assert_called_once_with("x", recipient_npub=None)
        assert result == expected

    @pytest.mark.asyncio
    async def test_passes_recipient_npub(self, operator_nsec, relays, templates):
        """open_channel passes recipient_npub to exchange."""
        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
        )

        npub = "npub1" + "a" * 58

        with patch.object(
            service.exchange, "open_channel",
            new_callable=AsyncMock,
            return_value={"success": True},
        ) as mock_open:
            await service.open_channel("x", recipient_npub=npub)

        mock_open.assert_called_once_with("x", recipient_npub=npub)


# ── receive tests ─────────────────────────────────────────────────────


class TestReceive:
    _SENDER = "npub1" + "b" * 58

    @pytest.mark.asyncio
    async def test_credentials_stripped_from_result(
        self, operator_nsec, relays, templates,
    ):
        """Credential values are NEVER echoed in the result."""
        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
        )

        exchange_result = {
            "success": True,
            "service": "x",
            "fields_received": 2,
            "sensitive_fields": 2,
            "encryption": "nip04",
            "credentials": {
                "access_token": "secret-tok",
                "access_token_secret": "secret-ts",
            },
            "message": "Credentials received.",
        }

        with patch.object(
            service.exchange, "receive",
            new_callable=AsyncMock,
            return_value=exchange_result,
        ):
            result = await service.receive(self._SENDER, service="x")

        assert "credentials" not in result
        assert "secret-tok" not in str(result)
        assert result["success"] is True
        assert result["fields_received"] == 2

    @pytest.mark.asyncio
    async def test_callback_called_with_correct_args(
        self, operator_nsec, relays, templates,
    ):
        """on_credentials_received callback receives correct arguments."""
        callback = AsyncMock(return_value={"session_activated": True})

        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
            on_credentials_received=callback,
        )

        creds = {
            "access_token": "tok",
            "access_token_secret": "ts",
        }

        exchange_result = {
            "success": True,
            "service": "x",
            "fields_received": 2,
            "sensitive_fields": 2,
            "encryption": "nip04",
            "credentials": creds,
            "message": "Credentials received.",
        }

        with patch.object(
            service.exchange, "receive",
            new_callable=AsyncMock,
            return_value=exchange_result,
        ):
            result = await service.receive(self._SENDER, service="x")

        callback.assert_called_once_with(self._SENDER, creds, "x")
        assert result["session_activated"] is True
        assert "credentials" not in result

    @pytest.mark.asyncio
    async def test_callback_result_merged(
        self, operator_nsec, relays, templates,
    ):
        """Callback return dict is merged into the result."""
        callback = AsyncMock(return_value={
            "session_activated": True,
            "dpyc_npub": self._SENDER,
            "seed_applied": True,
        })

        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
            on_credentials_received=callback,
        )

        exchange_result = {
            "success": True,
            "service": "x",
            "fields_received": 2,
            "sensitive_fields": 2,
            "encryption": "vault",
            "credentials": {"access_token": "t", "access_token_secret": "ts"},
            "message": "From vault.",
        }

        with patch.object(
            service.exchange, "receive",
            new_callable=AsyncMock,
            return_value=exchange_result,
        ):
            result = await service.receive(self._SENDER, service="x")

        assert result["session_activated"] is True
        assert result["dpyc_npub"] == self._SENDER
        assert result["seed_applied"] is True

    @pytest.mark.asyncio
    async def test_callback_returns_none(
        self, operator_nsec, relays, templates,
    ):
        """Callback returning None still works."""
        callback = AsyncMock(return_value=None)

        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
            on_credentials_received=callback,
        )

        exchange_result = {
            "success": True,
            "service": "x",
            "fields_received": 2,
            "sensitive_fields": 2,
            "encryption": "nip04",
            "credentials": {"access_token": "t", "access_token_secret": "ts"},
            "message": "Received.",
        }

        with patch.object(
            service.exchange, "receive",
            new_callable=AsyncMock,
            return_value=exchange_result,
        ):
            result = await service.receive(self._SENDER, service="x")

        assert result["success"] is True
        assert "credentials" not in result

    @pytest.mark.asyncio
    async def test_callback_raises_error(
        self, operator_nsec, relays, templates,
    ):
        """Callback exception is caught and reported, not re-raised."""
        callback = AsyncMock(side_effect=ValueError("session activation failed"))

        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
            on_credentials_received=callback,
        )

        exchange_result = {
            "success": True,
            "service": "x",
            "fields_received": 2,
            "sensitive_fields": 2,
            "encryption": "nip04",
            "credentials": {"access_token": "t", "access_token_secret": "ts"},
            "message": "Received.",
        }

        with patch.object(
            service.exchange, "receive",
            new_callable=AsyncMock,
            return_value=exchange_result,
        ):
            result = await service.receive(self._SENDER, service="x")

        # Should not raise — error captured in result
        assert result["success"] is True
        assert "callback_error" in result
        assert "session activation failed" in result["callback_error"]
        assert "credentials" not in result

    @pytest.mark.asyncio
    async def test_receive_without_callback(
        self, operator_nsec, relays, templates,
    ):
        """Receive works fine without an on_credentials_received callback."""
        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
            # No callback
        )

        exchange_result = {
            "success": True,
            "service": "x",
            "fields_received": 2,
            "sensitive_fields": 2,
            "encryption": "vault",
            "credentials": {"access_token": "t", "access_token_secret": "ts"},
            "message": "From vault.",
        }

        with patch.object(
            service.exchange, "receive",
            new_callable=AsyncMock,
            return_value=exchange_result,
        ):
            result = await service.receive(self._SENDER, service="x")

        assert result["success"] is True
        assert "credentials" not in result

    @pytest.mark.asyncio
    async def test_receive_failure_passed_through(
        self, operator_nsec, relays, templates,
    ):
        """Exchange errors propagate unchanged (no callback, no stripping)."""
        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
        )

        exchange_result = {
            "success": False,
            "error": "No DM found",
        }

        with patch.object(
            service.exchange, "receive",
            new_callable=AsyncMock,
            return_value=exchange_result,
        ):
            result = await service.receive(self._SENDER, service="x")

        assert result["success"] is False
        assert result["error"] == "No DM found"

    @pytest.mark.asyncio
    async def test_receive_uses_service_from_exchange_result(
        self, operator_nsec, relays, templates,
    ):
        """Callback receives the matched service from the exchange result."""
        callback = AsyncMock(return_value=None)

        # Add a second template
        templates["openai"] = CredentialTemplate(
            service="openai",
            version=1,
            fields={"api_key": FieldSpec(required=True, sensitive=True)},
        )

        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
            on_credentials_received=callback,
        )

        exchange_result = {
            "success": True,
            "service": "openai",
            "fields_received": 1,
            "sensitive_fields": 1,
            "encryption": "nip44",
            "credentials": {"api_key": "sk-xxx"},
            "message": "Received.",
        }

        with patch.object(
            service.exchange, "receive",
            new_callable=AsyncMock,
            return_value=exchange_result,
        ):
            await service.receive(self._SENDER, service="openai")

        # Callback should receive "openai" as the service name
        callback.assert_called_once_with(self._SENDER, {"api_key": "sk-xxx"}, "openai")


# ── forget tests ──────────────────────────────────────────────────────


class TestForget:
    _SENDER = "npub1" + "c" * 58

    @pytest.mark.asyncio
    async def test_delegates_to_exchange(self, operator_nsec, relays, templates):
        """forget delegates to the underlying exchange."""
        service = SecureCourierService(
            operator_nsec=operator_nsec,
            relays=relays,
            templates=templates,
        )

        expected = {
            "success": True,
            "service": "x",
            "deleted": True,
            "message": "Credentials deleted.",
        }

        with patch.object(
            service.exchange, "forget",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_forget:
            result = await service.forget(self._SENDER, service="x")

        mock_forget.assert_called_once_with(self._SENDER, service="x")
        assert result == expected
