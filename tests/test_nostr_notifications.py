"""Tests for proactive low-balance and expiration Nostr DM notifications."""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from pynostr.key import PrivateKey

from tollbooth.ledger import Tranche
from tollbooth.nostr_notifications import (
    NotificationLevel,
    NotificationManager,
    NotificationPreferences,
    _render_critical_message,
    _render_expiration_message,
    _render_warning_message,
)


# -- Fixtures ----------------------------------------------------------------

def _make_nsec() -> str:
    """Generate a fresh Nostr nsec for testing."""
    return PrivateKey().nsec


def _make_npub() -> str:
    """Generate a fresh Nostr npub for testing."""
    return PrivateKey().public_key.bech32()


def _make_manager(
    nsec: str | None = None,
    relays: list[str] | None = None,
    **kwargs,
) -> NotificationManager:
    """Create a NotificationManager with test defaults."""
    if nsec is None:
        nsec = _make_nsec()
    if relays is None:
        relays = ["wss://relay.test.com"]
    return NotificationManager(
        operator_nsec=nsec,
        relays=relays,
        service_name="Test MCP",
        **kwargs,
    )


def _make_tranche(
    remaining: int = 1000,
    original: int = 1000,
    expires_at: str | None = None,
) -> Tranche:
    """Create a test Tranche."""
    return Tranche(
        granted_at=datetime.now(timezone.utc).isoformat(),
        original_sats=original,
        remaining_sats=remaining,
        invoice_id="test-inv",
        expires_at=expires_at,
    )


# -- NotificationLevel ordering -----------------------------------------------

class TestNotificationLevel:
    """Tests for NotificationLevel enum ordering."""

    def test_ordering(self):
        assert NotificationLevel.NONE < NotificationLevel.WARNING
        assert NotificationLevel.WARNING < NotificationLevel.CRITICAL
        assert NotificationLevel.EXPIRATION_APPROACHING < NotificationLevel.WARNING

    def test_values(self):
        assert NotificationLevel.NONE == 0
        assert NotificationLevel.EXPIRATION_APPROACHING == 10
        assert NotificationLevel.WARNING == 20
        assert NotificationLevel.CRITICAL == 30


# -- NotificationPreferences --------------------------------------------------

class TestNotificationPreferences:
    """Tests for NotificationPreferences defaults and usage."""

    def test_defaults_are_opt_in(self):
        prefs = NotificationPreferences()
        assert prefs.low_balance_dm is True
        assert prefs.expiration_dm is True

    def test_custom_preferences(self):
        prefs = NotificationPreferences(low_balance_dm=False, expiration_dm=True)
        assert prefs.low_balance_dm is False
        assert prefs.expiration_dm is True


# -- Message templates ---------------------------------------------------------

class TestMessageTemplates:
    """Tests for notification message rendering."""

    def test_warning_message_includes_balance(self):
        msg = _render_warning_message("TestService", 500, 20, 100)
        assert "500" in msg
        assert "20%" in msg
        assert "100" in msg
        assert "TestService" in msg

    def test_warning_message_without_estimate(self):
        msg = _render_warning_message("TestService", 500, 20, None)
        assert "500" in msg
        assert "remaining" not in msg.lower() or "calls remaining" not in msg.lower()

    def test_warning_message_zero_estimate(self):
        msg = _render_warning_message("TestService", 500, 20, 0)
        assert "500" in msg
        # Zero estimate should not show calls remaining line
        assert "Approximate calls remaining" not in msg

    def test_critical_message_includes_interruption_warning(self):
        msg = _render_critical_message("TestService", 50, 5, 10)
        assert "50" in msg
        assert "5%" in msg
        assert "interrupted" in msg.lower()
        assert "TestService" in msg

    def test_critical_message_without_estimate(self):
        msg = _render_critical_message("TestService", 50, 5, None)
        assert "50" in msg
        assert "interrupted" in msg.lower()

    def test_expiration_message_includes_hours(self):
        msg = _render_expiration_message("TestService", 2000, 36.5)
        assert "2,000" in msg
        assert "36 hour" in msg
        assert "demurrage" in msg.lower()
        assert "TestService" in msg

    def test_expiration_message_singular_hour(self):
        msg = _render_expiration_message("TestService", 100, 1.2)
        assert "1 hour" in msg
        # Should not say "1 hours"
        assert "1 hours" not in msg

    def test_expiration_message_very_short_time(self):
        msg = _render_expiration_message("TestService", 100, 0.3)
        # Minimum display is 1 hour
        assert "1 hour" in msg


# -- NotificationManager initialization ---------------------------------------

class TestManagerInit:
    """Tests for NotificationManager initialization and graceful degradation."""

    def test_enabled_with_valid_config(self):
        mgr = _make_manager()
        assert mgr.enabled is True

    def test_disabled_with_empty_relays(self):
        mgr = _make_manager(relays=[])
        assert mgr.enabled is False

    def test_disabled_with_invalid_nsec(self):
        mgr = _make_manager(nsec="not-a-valid-nsec")
        assert mgr.enabled is False

    @patch("tollbooth.nostr_notifications._HAS_PYNOSTR", False)
    def test_disabled_without_pynostr(self):
        mgr = NotificationManager(
            operator_nsec="irrelevant",
            relays=["wss://relay.test.com"],
        )
        assert mgr.enabled is False

    @patch("tollbooth.nostr_notifications._HAS_WEBSOCKET", False)
    def test_disabled_without_websocket(self):
        mgr = NotificationManager(
            operator_nsec=_make_nsec(),
            relays=["wss://relay.test.com"],
        )
        assert mgr.enabled is False

    @patch("tollbooth.nostr_notifications._HAS_NIP44", False)
    def test_disabled_without_nip44(self):
        mgr = NotificationManager(
            operator_nsec=_make_nsec(),
            relays=["wss://relay.test.com"],
        )
        assert mgr.enabled is False


# -- Preferences management ---------------------------------------------------

class TestPreferences:
    """Tests for get/set preferences."""

    def test_default_preferences(self):
        mgr = _make_manager()
        npub = _make_npub()
        prefs = mgr.get_preferences(npub)
        assert prefs.low_balance_dm is True
        assert prefs.expiration_dm is True

    def test_set_and_get_preferences(self):
        mgr = _make_manager()
        npub = _make_npub()
        mgr.set_preferences(npub, NotificationPreferences(
            low_balance_dm=False, expiration_dm=True,
        ))
        prefs = mgr.get_preferences(npub)
        assert prefs.low_balance_dm is False
        assert prefs.expiration_dm is True

    def test_preferences_per_patron(self):
        mgr = _make_manager()
        npub1 = _make_npub()
        npub2 = _make_npub()
        mgr.set_preferences(npub1, NotificationPreferences(low_balance_dm=False))
        prefs1 = mgr.get_preferences(npub1)
        prefs2 = mgr.get_preferences(npub2)
        assert prefs1.low_balance_dm is False
        assert prefs2.low_balance_dm is True  # default


# -- Balance threshold notifications ------------------------------------------

class TestBalanceThresholds:
    """Tests for balance threshold crossing notifications."""

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_warning_at_20_percent(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()
        # 20% of 1000 = 200, balance is 150 (below 20%)
        mgr.check_and_notify(npub, balance=150, last_deposit=1000, tranches=[])
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][1]
        assert "20%" in msg

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_critical_at_5_percent(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()
        # 5% of 1000 = 50, balance is 30 (below 5%)
        mgr.check_and_notify(npub, balance=30, last_deposit=1000, tranches=[])
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][1]
        assert "5%" in msg
        assert "interrupted" in msg.lower()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_no_notification_above_warning(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()
        # 25% of 1000 = 250, balance is 250 (above 20%)
        mgr.check_and_notify(npub, balance=250, last_deposit=1000, tranches=[])
        mock_send.assert_not_called()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_exact_threshold_triggers(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()
        # Exactly 20% -- should trigger
        mgr.check_and_notify(npub, balance=200, last_deposit=1000, tranches=[])
        mock_send.assert_called_once()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_zero_last_deposit_no_crash(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()
        # Zero deposit means no percentage can be computed
        mgr.check_and_notify(npub, balance=100, last_deposit=0, tranches=[])
        mock_send.assert_not_called()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_estimate_calls_in_message(self, mock_send):
        mgr = _make_manager(avg_cost_per_call=10)
        npub = _make_npub()
        # balance=100, avg_cost=10 => ~10 calls remaining
        mgr.check_and_notify(npub, balance=100, last_deposit=1000, tranches=[])
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][1]
        assert "10" in msg


# -- Deduplication -------------------------------------------------------------

class TestDeduplication:
    """Tests for notification deduplication logic."""

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_dedup_same_level_no_resend(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        # First call at warning level
        mgr.check_and_notify(npub, balance=150, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 1

        # Second call still at warning level -- should NOT re-send
        mgr.check_and_notify(npub, balance=140, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 1

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_escalation_from_warning_to_critical(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        # First: warning level
        mgr.check_and_notify(npub, balance=150, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 1

        # Second: critical level -- should send again
        mgr.check_and_notify(npub, balance=30, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 2

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_deposit_resets_dedup(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        # Trigger warning
        mgr.check_and_notify(npub, balance=150, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 1

        # Patron deposits new credits -- reset dedup
        mgr.record_deposit(npub, 5000)

        # New check at warning level after deposit -- should fire again
        mgr.check_and_notify(npub, balance=800, last_deposit=5000, tranches=[])
        assert mock_send.call_count == 2

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_balance_recovery_then_new_drop(self, mock_send):
        """Balance drops to warning, then recovers (no notification),
        then drops again. The second drop should NOT re-fire because
        dedup state persists until record_deposit."""
        mgr = _make_manager()
        npub = _make_npub()

        # Drop to warning
        mgr.check_and_notify(npub, balance=150, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 1

        # Balance goes back up (no notification expected)
        mgr.check_and_notify(npub, balance=300, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 1

        # Drop again -- dedup state still at WARNING, so no re-send
        mgr.check_and_notify(npub, balance=180, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 1

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_different_patrons_independent(self, mock_send):
        mgr = _make_manager()
        npub1 = _make_npub()
        npub2 = _make_npub()

        mgr.check_and_notify(npub1, balance=150, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 1

        # Different patron at same level -- should fire independently
        mgr.check_and_notify(npub2, balance=150, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 2


# -- Tranche expiration notifications -----------------------------------------

class TestExpirationNotifications:
    """Tests for tranche expiration approaching notifications."""

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_expiration_within_window(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        # Tranche expiring in 24 hours (within 48h window)
        exp_time = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        tranche = _make_tranche(remaining=500, expires_at=exp_time)

        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        # Should send expiration DM (balance is at 100%, no low-balance DM)
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][1]
        assert "expiring" in msg.lower()
        assert "500" in msg

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_no_expiration_outside_window(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        # Tranche expiring in 72 hours (outside 48h window)
        exp_time = (datetime.now(timezone.utc) + timedelta(hours=72)).isoformat()
        tranche = _make_tranche(remaining=500, expires_at=exp_time)

        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        mock_send.assert_not_called()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_no_expiration_for_never_expiring_tranche(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        tranche = _make_tranche(remaining=500, expires_at=None)
        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        mock_send.assert_not_called()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_expiration_dedup_24h_cooldown(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        exp_time = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        tranche = _make_tranche(remaining=500, expires_at=exp_time)

        # First call: sends expiration DM
        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        assert mock_send.call_count == 1

        # Second call within 24h: should NOT re-send
        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        assert mock_send.call_count == 1

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_already_expired_tranche_ignored(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        # Already expired tranche
        exp_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        tranche = _make_tranche(remaining=500, expires_at=exp_time)

        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        mock_send.assert_not_called()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_empty_tranche_ignored(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        exp_time = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        tranche = _make_tranche(remaining=0, expires_at=exp_time)

        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        mock_send.assert_not_called()


# -- Relay failure resilience --------------------------------------------------

class TestRelayResilience:
    """Tests that notification failures never block the caller."""

    @patch("tollbooth.nostr_notifications.create_connection")
    def test_relay_failure_does_not_raise(self, mock_ws):
        mock_ws.side_effect = ConnectionError("relay down")
        mgr = _make_manager()
        npub = _make_npub()

        # Should not raise even though relay is down
        mgr.check_and_notify(npub, balance=30, last_deposit=1000, tranches=[])

    @patch.object(NotificationManager, "_publish_to_relays")
    def test_send_nip44_dm_catches_encrypt_error(self, mock_publish):
        mgr = _make_manager()
        npub = _make_npub()

        # Patch encrypt to raise -- should be caught internally
        with patch("tollbooth.nostr_notifications._nip44_encrypt", side_effect=Exception("boom")):
            mgr._send_nip44_dm(npub, "test message")

        mock_publish.assert_not_called()  # Never got to publish

    def test_check_and_notify_catches_all_exceptions(self):
        mgr = _make_manager()
        npub = _make_npub()

        # Patch internal method to raise
        with patch.object(mgr, "_do_check_and_notify", side_effect=RuntimeError("kaboom")):
            # Should not raise
            mgr.check_and_notify(npub, balance=30, last_deposit=1000, tranches=[])

    def test_non_npub_patron_skipped_silently(self):
        mgr = _make_manager()
        # Non-npub patron ID -- should be silently skipped
        mgr.check_and_notify("user_01KGZY", balance=30, last_deposit=1000, tranches=[])

    def test_disabled_manager_is_noop(self):
        mgr = _make_manager(relays=[])
        assert mgr.enabled is False
        npub = _make_npub()
        # Should not raise
        mgr.check_and_notify(npub, balance=30, last_deposit=1000, tranches=[])


# -- Notification preferences respected ----------------------------------------

class TestPreferencesRespected:
    """Tests that opt-out preferences suppress DMs."""

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_low_balance_opt_out(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()
        mgr.set_preferences(npub, NotificationPreferences(
            low_balance_dm=False, expiration_dm=True,
        ))

        # Below warning threshold but opted out
        mgr.check_and_notify(npub, balance=150, last_deposit=1000, tranches=[])
        mock_send.assert_not_called()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_expiration_opt_out(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()
        mgr.set_preferences(npub, NotificationPreferences(
            low_balance_dm=True, expiration_dm=False,
        ))

        exp_time = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        tranche = _make_tranche(remaining=500, expires_at=exp_time)

        # High balance so no low-balance DM, but tranche expiring
        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        mock_send.assert_not_called()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_all_opted_out_suppresses_all(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()
        mgr.set_preferences(npub, NotificationPreferences(
            low_balance_dm=False, expiration_dm=False,
        ))

        exp_time = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        tranche = _make_tranche(remaining=500, expires_at=exp_time)

        # Below critical AND tranche expiring -- but all opted out
        mgr.check_and_notify(npub, balance=10, last_deposit=1000, tranches=[tranche])
        mock_send.assert_not_called()


# -- NIP-44 DM sending --------------------------------------------------------

class TestNip44DmSending:
    """Tests for the NIP-44 encrypted DM construction and sending."""

    @patch("tollbooth.nostr_notifications.create_connection")
    @patch("tollbooth.nostr_notifications._nip44_encrypt", return_value="encrypted_payload")
    def test_send_nip44_dm_constructs_event(self, mock_encrypt, mock_ws):
        mock_conn = MagicMock()
        mock_conn.recv.return_value = '["OK"]'
        mock_ws.return_value = mock_conn

        mgr = _make_manager()
        npub = _make_npub()

        mgr._send_nip44_dm(npub, "Hello patron")

        # Encrypt was called with the message
        mock_encrypt.assert_called_once()
        args = mock_encrypt.call_args[0]
        assert args[0] == "Hello patron"

        # Give daemon thread a moment to start
        time.sleep(0.1)

    @patch("tollbooth.nostr_notifications.create_connection")
    @patch("tollbooth.nostr_notifications._nip44_encrypt", return_value="enc")
    def test_send_to_multiple_relays(self, mock_encrypt, mock_ws):
        mock_conn = MagicMock()
        mock_conn.recv.return_value = '["OK"]'
        mock_ws.return_value = mock_conn

        mgr = _make_manager(relays=["wss://r1.test", "wss://r2.test"])
        npub = _make_npub()

        # Call _publish_to_relays directly (synchronous) for deterministic test
        mgr._send_nip44_dm(npub, "test")
        # Wait for daemon thread
        time.sleep(0.2)

        # Both relays should have been contacted
        assert mock_ws.call_count == 2

    def test_send_nip44_dm_invalid_npub_no_crash(self):
        mgr = _make_manager()
        # Invalid npub should be caught gracefully
        mgr._send_nip44_dm("not-an-npub", "test")

    def test_send_nip44_dm_disabled_manager(self):
        mgr = _make_manager(relays=[])
        npub = _make_npub()
        # Should return without doing anything
        mgr._send_nip44_dm(npub, "test")


# -- record_deposit -----------------------------------------------------------

class TestRecordDeposit:
    """Tests for the record_deposit reset mechanism."""

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_deposit_clears_level(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        # Trigger critical
        mgr.check_and_notify(npub, balance=30, last_deposit=1000, tranches=[])
        assert mock_send.call_count == 1

        # Record deposit
        mgr.record_deposit(npub, 5000)

        # Now at warning level again against new deposit -- should fire
        mgr.check_and_notify(npub, balance=800, last_deposit=5000, tranches=[])
        assert mock_send.call_count == 2

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_deposit_for_unknown_patron(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()
        # Should not crash for previously unseen patron
        mgr.record_deposit(npub, 5000)

        # Now check -- should work normally
        mgr.check_and_notify(npub, balance=800, last_deposit=5000, tranches=[])
        assert mock_send.call_count == 1


# -- Custom threshold configuration -------------------------------------------

class TestCustomThresholds:
    """Tests for non-default threshold configuration."""

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_custom_warning_percent(self, mock_send):
        mgr = _make_manager(warning_pct=30, critical_pct=10)
        npub = _make_npub()

        # 25% of 1000 = below 30% custom warning
        mgr.check_and_notify(npub, balance=250, last_deposit=1000, tranches=[])
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][1]
        assert "30%" in msg

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_custom_critical_percent(self, mock_send):
        mgr = _make_manager(warning_pct=30, critical_pct=10)
        npub = _make_npub()

        # 8% of 1000 = below 10% custom critical
        mgr.check_and_notify(npub, balance=80, last_deposit=1000, tranches=[])
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][1]
        assert "10%" in msg

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_custom_expiration_window(self, mock_send):
        mgr = _make_manager(expiration_window_secs=12 * 3600)  # 12 hours
        npub = _make_npub()

        # Tranche expiring in 10 hours (within 12h custom window)
        exp_time = (datetime.now(timezone.utc) + timedelta(hours=10)).isoformat()
        tranche = _make_tranche(remaining=500, expires_at=exp_time)

        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        mock_send.assert_called_once()

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_custom_expiration_window_excludes_far_tranche(self, mock_send):
        mgr = _make_manager(expiration_window_secs=12 * 3600)  # 12 hours
        npub = _make_npub()

        # Tranche expiring in 24 hours (outside 12h custom window)
        exp_time = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        tranche = _make_tranche(remaining=500, expires_at=exp_time)

        mgr.check_and_notify(npub, balance=500, last_deposit=500, tranches=[tranche])
        mock_send.assert_not_called()


# -- Combined scenarios --------------------------------------------------------

class TestCombinedScenarios:
    """Tests for scenarios with both low-balance and expiration triggers."""

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_low_balance_and_expiration_both_fire(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        exp_time = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        tranche = _make_tranche(remaining=150, expires_at=exp_time)

        # 15% balance (warning) AND tranche expiring
        mgr.check_and_notify(
            npub, balance=150, last_deposit=1000, tranches=[tranche],
        )
        # Should get 2 DMs: one for low balance, one for expiration
        assert mock_send.call_count == 2

    @patch.object(NotificationManager, "_send_nip44_dm")
    def test_multiple_expiring_tranches_one_notification(self, mock_send):
        mgr = _make_manager()
        npub = _make_npub()

        exp1 = (datetime.now(timezone.utc) + timedelta(hours=20)).isoformat()
        exp2 = (datetime.now(timezone.utc) + timedelta(hours=30)).isoformat()
        t1 = _make_tranche(remaining=300, expires_at=exp1)
        t2 = _make_tranche(remaining=200, expires_at=exp2)

        # High balance, two expiring tranches
        mgr.check_and_notify(
            npub, balance=500, last_deposit=500, tranches=[t1, t2],
        )
        # Should get 1 DM covering combined expiring sats
        assert mock_send.call_count == 1
        msg = mock_send.call_args[0][1]
        assert "500" in msg  # 300 + 200 = 500 expiring
