"""Tests for auto-restore session binding (cold-start resilience)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tollbooth.credential_vault_backend import (
    CredentialVaultBackend,
    SessionBindingBackend,
)
from tollbooth.vaults.neon import NeonCredentialVault


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_neon_vault_mock(execute_return=None):
    """Create a mock NeonVault with a controllable _execute."""
    neon = MagicMock()
    neon._execute = AsyncMock(return_value=execute_return or {})
    return neon


def _make_credential_vault(execute_return=None):
    """Create a NeonCredentialVault wrapping a mocked NeonVault."""
    neon = _make_neon_vault_mock(execute_return)
    vault = NeonCredentialVault(neon_vault=neon)
    return vault, neon


# ---------------------------------------------------------------------------
# NeonCredentialVault — SessionBindingBackend SQL operations
# ---------------------------------------------------------------------------

class TestSessionBindingSQL:
    """Verify SQL round-trips via mocked _execute."""

    @pytest.mark.asyncio
    async def test_store_and_fetch_binding(self):
        """Store a binding then fetch it back."""
        vault, neon = _make_credential_vault()

        # Store
        await vault.store_session_binding("user_01", "thebrain", "npub1abc")
        neon._execute.assert_called_once()
        call_args = neon._execute.call_args
        assert "INSERT INTO session_bindings" in call_args[0][0]
        assert call_args[0][1] == ["user_01", "thebrain", "npub1abc"]

        # Fetch
        neon._execute.reset_mock()
        neon._execute.return_value = {"rows": [{"npub": "npub1abc"}]}
        result = await vault.fetch_session_binding("user_01", "thebrain")
        assert result == "npub1abc"
        assert "SELECT npub FROM session_bindings" in neon._execute.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_missing_returns_none(self):
        """Empty rows → None."""
        vault, neon = _make_credential_vault({"rows": []})
        result = await vault.fetch_session_binding("user_01", "thebrain")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_overwrites_npub(self):
        """ON CONFLICT UPDATE behavior — SQL contains ON CONFLICT."""
        vault, neon = _make_credential_vault()
        await vault.store_session_binding("user_01", "thebrain", "npub1new")
        sql = neon._execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_delete_binding_found(self):
        """rowCount > 0 → True."""
        vault, neon = _make_credential_vault({"rowCount": 1})
        result = await vault.delete_session_binding("user_01", "thebrain")
        assert result is True
        assert "DELETE FROM session_bindings" in neon._execute.call_args[0][0]

    @pytest.mark.asyncio
    async def test_delete_binding_not_found(self):
        """rowCount == 0 → False."""
        vault, neon = _make_credential_vault({"rowCount": 0})
        result = await vault.delete_session_binding("user_01", "thebrain")
        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_schema_creates_session_bindings(self):
        """ensure_schema() creates both credentials and session_bindings tables."""
        vault, neon = _make_credential_vault()
        await vault.ensure_schema()
        # Should have been called twice: credentials + session_bindings
        assert neon._execute.call_count == 2
        sqls = [call[0][0] for call in neon._execute.call_args_list]
        assert any("credentials" in s for s in sqls)
        assert any("session_bindings" in s for s in sqls)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:

    def test_neon_credential_vault_is_session_binding_backend(self):
        """NeonCredentialVault satisfies the SessionBindingBackend protocol."""
        vault, _ = _make_credential_vault()
        assert isinstance(vault, SessionBindingBackend)

    def test_neon_credential_vault_still_is_credential_vault_backend(self):
        """NeonCredentialVault still satisfies CredentialVaultBackend."""
        vault, _ = _make_credential_vault()
        assert isinstance(vault, CredentialVaultBackend)

    def test_plain_dict_is_not_session_binding_backend(self):
        """A plain dict does NOT satisfy the protocol."""
        assert not isinstance({}, SessionBindingBackend)


# ---------------------------------------------------------------------------
# SecureCourierService — restore_session flow
# ---------------------------------------------------------------------------

def _make_courier_service(
    *,
    vault=None,
    on_received=None,
):
    """Create a SecureCourierService with mocked internals."""
    from tollbooth.secure_courier import SecureCourierService

    # We can't easily construct a real SecureCourierService without nostr deps,
    # so we build one with mocked exchange.
    svc = object.__new__(SecureCourierService)
    svc._operator_nsec = "nsec1fake"
    svc._send_card_dm = False
    svc._sessions = {}
    svc._on_received = on_received

    exchange = MagicMock()
    exchange._credential_vault = vault
    exchange.receive = AsyncMock(return_value={
        "success": True,
        "service": "thebrain",
        "credentials": {"api_key": "k", "brain_id": "b"},
        "encryption": "vault",
    })
    exchange.forget = AsyncMock(return_value={"success": True, "deleted": True})
    svc._exchange = exchange

    return svc, exchange


class TestRestoreSession:

    @pytest.mark.asyncio
    async def test_restore_calls_receive(self):
        """restore_session() calls receive() with the stored npub."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.fetch_session_binding = AsyncMock(return_value="npub1abc")
        vault.store_session_binding = AsyncMock()

        callback = AsyncMock(return_value={"session": "active"})
        svc, exchange = _make_courier_service(vault=vault, on_received=callback)

        result = await svc.restore_session("user_01", service="thebrain")
        assert result == "npub1abc"

        # receive() was called on the exchange
        exchange.receive.assert_called_once_with("npub1abc", service="thebrain")
        # Callback fired
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_no_binding_returns_none(self):
        """No binding in vault → None, no receive() call."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.fetch_session_binding = AsyncMock(return_value=None)

        svc, exchange = _make_courier_service(vault=vault)
        result = await svc.restore_session("user_01", service="thebrain")
        assert result is None
        exchange.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_no_vault_returns_none(self):
        """No vault at all → None."""
        svc, exchange = _make_courier_service(vault=None)
        result = await svc.restore_session("user_01", service="thebrain")
        assert result is None
        exchange.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_vault_without_binding_support(self):
        """Vault doesn't implement SessionBindingBackend → None."""
        # A plain CredentialVaultBackend without session binding methods
        vault = MagicMock(spec=CredentialVaultBackend)
        svc, exchange = _make_courier_service(vault=vault)
        result = await svc.restore_session("user_01", service="thebrain")
        assert result is None
        exchange.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_receive_failure_returns_none(self):
        """If receive() raises, restore returns None."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.fetch_session_binding = AsyncMock(return_value="npub1abc")

        svc, exchange = _make_courier_service(vault=vault)
        exchange.receive = AsyncMock(side_effect=RuntimeError("relay down"))

        result = await svc.restore_session("user_01", service="thebrain")
        assert result is None


# ---------------------------------------------------------------------------
# SecureCourierService — binding stored/deleted on receive/forget
# ---------------------------------------------------------------------------

class TestBindingOnReceiveForget:

    @pytest.mark.asyncio
    async def test_binding_stored_on_receive(self):
        """caller_id param → store_session_binding called."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.store_session_binding = AsyncMock()

        svc, _ = _make_courier_service(vault=vault)
        await svc.receive("npub1abc", "thebrain", caller_id="user_01")

        vault.store_session_binding.assert_called_once_with(
            "user_01", "thebrain", "npub1abc",
        )

    @pytest.mark.asyncio
    async def test_binding_not_stored_without_caller_id(self):
        """No caller_id → store_session_binding NOT called."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.store_session_binding = AsyncMock()

        svc, _ = _make_courier_service(vault=vault)
        await svc.receive("npub1abc", "thebrain")

        vault.store_session_binding.assert_not_called()

    @pytest.mark.asyncio
    async def test_binding_deleted_on_forget(self):
        """caller_id param → delete_session_binding called."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.delete_session_binding = AsyncMock()

        svc, _ = _make_courier_service(vault=vault)
        await svc.forget("npub1abc", "thebrain", caller_id="user_01")

        vault.delete_session_binding.assert_called_once_with(
            "user_01", "thebrain",
        )

    @pytest.mark.asyncio
    async def test_binding_not_deleted_without_caller_id(self):
        """No caller_id → delete_session_binding NOT called."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.delete_session_binding = AsyncMock()

        svc, _ = _make_courier_service(vault=vault)
        await svc.forget("npub1abc", "thebrain")

        vault.delete_session_binding.assert_not_called()

    @pytest.mark.asyncio
    async def test_receive_populates_in_memory_sessions(self):
        """receive() with caller_id populates _sessions dict."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.store_session_binding = AsyncMock()

        svc, _ = _make_courier_service(vault=vault)
        assert "user_01" not in svc._sessions
        await svc.receive("npub1abc", "thebrain", caller_id="user_01")
        assert svc._sessions["user_01"] == "npub1abc"

    @pytest.mark.asyncio
    async def test_forget_clears_in_memory_sessions(self):
        """forget() with caller_id clears _sessions dict."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.delete_session_binding = AsyncMock()

        svc, _ = _make_courier_service(vault=vault)
        svc._sessions["user_01"] = "npub1abc"
        await svc.forget("npub1abc", "thebrain", caller_id="user_01")
        assert "user_01" not in svc._sessions


# ---------------------------------------------------------------------------
# SecureCourierService — ensure_identity (operator-facing API)
# ---------------------------------------------------------------------------

class TestEnsureIdentity:

    @pytest.mark.asyncio
    async def test_fast_path_from_memory(self):
        """In-memory _sessions hit → returns npub immediately, no vault I/O."""
        svc, exchange = _make_courier_service(vault=None)
        svc._sessions["user_01"] = "npub1abc"

        result = await svc.ensure_identity("user_01", service="thebrain")
        assert result == "npub1abc"
        exchange.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_cold_start_restore(self):
        """No in-memory session → vault restore → returns npub."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.fetch_session_binding = AsyncMock(return_value="npub1abc")
        vault.store_session_binding = AsyncMock()

        callback = AsyncMock(return_value={"session": "active"})
        svc, exchange = _make_courier_service(vault=vault, on_received=callback)

        result = await svc.ensure_identity("user_01", service="thebrain")
        assert result == "npub1abc"
        # _sessions populated by receive() via restore_session()
        assert svc._sessions["user_01"] == "npub1abc"

    @pytest.mark.asyncio
    async def test_no_binding_raises(self):
        """No binding and no in-memory session → ValueError."""
        vault = MagicMock(spec=SessionBindingBackend)
        vault.fetch_session_binding = AsyncMock(return_value=None)

        svc, _ = _make_courier_service(vault=vault)

        with pytest.raises(ValueError, match="No DPYC identity active"):
            await svc.ensure_identity("user_01", service="thebrain")

    @pytest.mark.asyncio
    async def test_no_vault_raises(self):
        """No vault at all → ValueError."""
        svc, _ = _make_courier_service(vault=None)

        with pytest.raises(ValueError, match="No DPYC identity active"):
            await svc.ensure_identity("user_01", service="thebrain")
