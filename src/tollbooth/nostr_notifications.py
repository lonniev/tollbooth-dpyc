"""Proactive low-balance and expiration Nostr DM notifications.

Extends the Secure Courier relay channel into a persistent operator-to-patron
notification channel. Sends NIP-44 encrypted DMs when balance crosses
thresholds or tranches approach expiration.

    "The diplomatic pouch network doesn't just deliver credentials --
     it also delivers dispatches."

Architecture:

    LedgerCache.debit() ---> NotificationManager.check_and_notify()
                                  |
                                  v
                            NIP-44 encrypted kind 4 DM
                                  |
                                  v
                            Nostr relays (fire-and-forget)

All relay I/O is fire-and-forget in daemon threads. Notification failures
never block the calling path.

Dependencies are optional -- install with ``pip install tollbooth-dpyc[nostr]``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Any

from tollbooth.ledger import Tranche

logger = logging.getLogger(__name__)

# Optional imports -- graceful degradation
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
    from tollbooth.nip44 import encrypt as _nip44_encrypt

    _HAS_NIP44 = True
except ImportError:
    _HAS_NIP44 = False


# ---------------------------------------------------------------------------
# Notification levels (ordered by severity)
# ---------------------------------------------------------------------------

class NotificationLevel(IntEnum):
    """Balance notification severity levels.

    Higher values = more severe. Used for deduplication: a patron
    receives at most one DM per level per downward crossing.
    """

    NONE = 0
    EXPIRATION_APPROACHING = 10
    WARNING = 20  # 20% of last deposit remaining
    CRITICAL = 30  # 5% remaining


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_WARNING_PERCENT = 20  # Fire at 20% of last deposit remaining
DEFAULT_CRITICAL_PERCENT = 5  # Fire at 5% of last deposit remaining
DEFAULT_EXPIRATION_WINDOW_SECS = 48 * 3600  # 48 hours before expiry

# Nostr event kind for encrypted DMs (NIP-04 envelope, NIP-44 content)
_KIND_ENCRYPTED_DM = 4

# Default relay timeout
_RELAY_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Notification preferences
# ---------------------------------------------------------------------------

@dataclass
class NotificationPreferences:
    """Per-patron notification preferences.

    Opt-in by default for Secure Courier onboarded patrons (those who
    have an npub in the vault).
    """

    low_balance_dm: bool = True
    expiration_dm: bool = True


# ---------------------------------------------------------------------------
# Message templates
# ---------------------------------------------------------------------------

def _render_warning_message(
    service_name: str,
    balance: int,
    threshold_pct: int,
    estimate_calls: int | None,
) -> str:
    """Render a low-balance warning DM."""
    parts = [
        f"Hey -- your {service_name} balance just dropped below {threshold_pct}%.",
        "",
        f"Current balance: {balance:,} api-sats",
    ]
    if estimate_calls is not None and estimate_calls > 0:
        parts.append(f"Approximate calls remaining: ~{estimate_calls:,}")
    parts.extend([
        "",
        "Top up: call purchase_credits in your next Claude session.",
    ])
    return "\n".join(parts)


def _render_critical_message(
    service_name: str,
    balance: int,
    threshold_pct: int,
    estimate_calls: int | None,
) -> str:
    """Render a critical low-balance DM."""
    parts = [
        f"Heads up -- your {service_name} balance is critically low "
        f"(below {threshold_pct}%).",
        "",
        f"Current balance: {balance:,} api-sats",
    ]
    if estimate_calls is not None and estimate_calls > 0:
        parts.append(f"Approximate calls remaining: ~{estimate_calls:,}")
    parts.extend([
        "",
        "Service will be interrupted when credits run out.",
        "Top up now: call purchase_credits in your next Claude session.",
    ])
    return "\n".join(parts)


def _render_expiration_message(
    service_name: str,
    expiring_sats: int,
    hours_until: float,
) -> str:
    """Render a tranche expiration approaching DM."""
    hours_int = max(1, int(hours_until))
    parts = [
        f"Friendly notice -- some of your {service_name} credits are "
        f"expiring soon.",
        "",
        f"Credits expiring: {expiring_sats:,} api-sats",
        f"Time remaining: ~{hours_int} hour{'s' if hours_int != 1 else ''}",
        "",
        "Unused credits in expiring tranches will be collected when they reach their lifetime.",
        "Use them or top up before they expire.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dedup tracker
# ---------------------------------------------------------------------------

@dataclass
class _PatronState:
    """In-memory dedup state for a single patron."""

    last_level: NotificationLevel = NotificationLevel.NONE
    last_expiration_notified_at: float = 0.0  # monotonic time
    last_deposit_balance: int = 0  # balance right after last known deposit


# ---------------------------------------------------------------------------
# NotificationManager
# ---------------------------------------------------------------------------

class NotificationManager:
    """Manages proactive low-balance and expiration DM notifications.

    Tracks per-patron dedup state and sends NIP-44 encrypted DMs
    when balance crosses configurable thresholds.

    All relay I/O is fire-and-forget in daemon threads. Notification
    failures are logged but never block or raise.

    Args:
        operator_nsec: Operator's Nostr nsec (bech32).
        relays: List of relay WebSocket URLs.
        service_name: Human-readable service name for message templates.
        warning_pct: Low-balance warning threshold (default 20%).
        critical_pct: Critical balance threshold (default 5%).
        expiration_window_secs: Seconds before tranche expiry to notify
            (default 48 hours).
        avg_cost_per_call: Average cost per tool call for estimating
            remaining calls (default 5 api-sats).
    """

    def __init__(
        self,
        operator_nsec: str,
        relays: list[str],
        service_name: str = "Tollbooth MCP",
        *,
        warning_pct: int = DEFAULT_WARNING_PERCENT,
        critical_pct: int = DEFAULT_CRITICAL_PERCENT,
        expiration_window_secs: int = DEFAULT_EXPIRATION_WINDOW_SECS,
        avg_cost_per_call: int = 5,
    ) -> None:
        self._service_name = service_name
        self._warning_pct = warning_pct
        self._critical_pct = critical_pct
        self._expiration_window_secs = expiration_window_secs
        self._avg_cost_per_call = avg_cost_per_call
        self._relays = [r.strip() for r in relays if r.strip()]

        # Key material
        self._privkey_hex: str = ""
        self._pubkey_hex: str = ""
        self._enabled = True

        if not _HAS_PYNOSTR:
            logger.warning(
                "pynostr not installed -- notifications disabled. "
                "Install with: pip install tollbooth-dpyc[nostr]"
            )
            self._enabled = False
            return

        if not _HAS_WEBSOCKET:
            logger.warning(
                "websocket-client not installed -- notifications disabled. "
                "Install with: pip install tollbooth-dpyc[nostr]"
            )
            self._enabled = False
            return

        if not _HAS_NIP44:
            logger.warning(
                "NIP-44 module not available -- notifications disabled."
            )
            self._enabled = False
            return

        if not self._relays:
            logger.warning("No Nostr relays configured -- notifications disabled.")
            self._enabled = False
            return

        try:
            pk = PrivateKey.from_nsec(operator_nsec)
            self._privkey_hex = pk.hex()
            self._pubkey_hex = pk.public_key.hex()
        except Exception as exc:
            logger.warning("Invalid operator nsec -- notifications disabled: %s", exc)
            self._enabled = False

        # Per-patron dedup state (in-memory)
        self._patron_state: dict[str, _PatronState] = {}
        # Per-patron notification preferences
        self._preferences: dict[str, NotificationPreferences] = {}
        # Lock for thread-safe state access
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Whether the notification manager is active."""
        return self._enabled

    # -- Preferences API -------------------------------------------------------

    def get_preferences(self, patron_npub: str) -> NotificationPreferences:
        """Get notification preferences for a patron.

        Returns default (all enabled) if no preferences have been set.
        """
        with self._lock:
            return self._preferences.get(
                patron_npub, NotificationPreferences(),
            )

    def set_preferences(
        self, patron_npub: str, prefs: NotificationPreferences,
    ) -> None:
        """Set notification preferences for a patron."""
        with self._lock:
            self._preferences[patron_npub] = prefs

    # -- Main entry point ------------------------------------------------------

    def check_and_notify(
        self,
        patron_npub: str,
        balance: int,
        last_deposit: int,
        tranches: list[Tranche],
    ) -> None:
        """Check balance thresholds and send notifications if warranted.

        Called after each tool-call debit. All work is fire-and-forget;
        this method returns immediately and never raises.

        Args:
            patron_npub: Patron's npub (bech32).
            balance: Current balance in api-sats (post-debit).
            last_deposit: Size of the most recent deposit in api-sats.
                Used to compute percentage thresholds.
            tranches: Current list of active tranches for expiration checks.
        """
        if not self._enabled:
            return

        if not patron_npub.startswith("npub1"):
            logger.debug(
                "Notification skipped: patron ID %s is not an npub.",
                patron_npub[:16],
            )
            return

        try:
            self._do_check_and_notify(patron_npub, balance, last_deposit, tranches)
        except Exception as exc:
            logger.debug(
                "Notification check failed (non-fatal): %s", exc,
            )

    def _do_check_and_notify(
        self,
        patron_npub: str,
        balance: int,
        last_deposit: int,
        tranches: list[Tranche],
    ) -> None:
        """Internal check logic (may raise; caller catches)."""
        prefs = self.get_preferences(patron_npub)

        with self._lock:
            state = self._patron_state.setdefault(
                patron_npub, _PatronState(),
            )

        # -- Balance threshold checks ------------------------------------------

        if prefs.low_balance_dm and last_deposit > 0:
            pct = (balance * 100) / last_deposit
            new_level = NotificationLevel.NONE

            if pct <= self._critical_pct:
                new_level = NotificationLevel.CRITICAL
            elif pct <= self._warning_pct:
                new_level = NotificationLevel.WARNING

            with self._lock:
                if new_level > state.last_level:
                    # Downward crossing -- send notification
                    state.last_level = new_level
                    estimate = (
                        balance // self._avg_cost_per_call
                        if self._avg_cost_per_call > 0 else None
                    )
                    self._fire_balance_notification(
                        patron_npub, balance, new_level, estimate,
                    )

        # -- Tranche expiration checks -----------------------------------------

        if prefs.expiration_dm:
            self._check_expiration(patron_npub, tranches)

    def record_deposit(self, patron_npub: str, deposit_amount: int) -> None:
        """Record that a patron made a deposit. Resets dedup state.

        Called when patron deposits new credits so that threshold
        notifications fire fresh on the next downward crossing.

        Args:
            patron_npub: Patron's npub (bech32).
            deposit_amount: Amount of the deposit in api-sats.
        """
        with self._lock:
            state = self._patron_state.setdefault(
                patron_npub, _PatronState(),
            )
            state.last_level = NotificationLevel.NONE
            state.last_deposit_balance = deposit_amount

    # -- Internal helpers ------------------------------------------------------

    def _fire_balance_notification(
        self,
        patron_npub: str,
        balance: int,
        level: NotificationLevel,
        estimate_calls: int | None,
    ) -> None:
        """Compose and send a balance notification DM (fire-and-forget)."""
        if level == NotificationLevel.CRITICAL:
            msg = _render_critical_message(
                self._service_name,
                balance,
                self._critical_pct,
                estimate_calls,
            )
        elif level == NotificationLevel.WARNING:
            msg = _render_warning_message(
                self._service_name,
                balance,
                self._warning_pct,
                estimate_calls,
            )
        else:
            return  # NONE or EXPIRATION_APPROACHING handled separately

        self._send_nip44_dm(patron_npub, msg)

    def _check_expiration(
        self,
        patron_npub: str,
        tranches: list[Tranche],
    ) -> None:
        """Check for tranches expiring within the notification window."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(seconds=self._expiration_window_secs)

        expiring_sats = 0
        earliest_expiry: datetime | None = None

        for t in tranches:
            if t.remaining_sats <= 0 or t.is_expired_at(now):
                continue
            if t.expires_at is None:
                continue
            try:
                exp = datetime.fromisoformat(t.expires_at.replace("Z", "+00:00"))
                if exp <= cutoff:
                    expiring_sats += t.remaining_sats
                    if earliest_expiry is None or exp < earliest_expiry:
                        earliest_expiry = exp
            except (ValueError, TypeError):
                continue

        if expiring_sats <= 0 or earliest_expiry is None:
            return

        # Dedup: only notify once per expiration window
        # Re-notify after the window resets (e.g., new tranches added)
        with self._lock:
            state = self._patron_state.setdefault(
                patron_npub, _PatronState(),
            )
            mono_now = time.monotonic()
            # Don't re-send if we notified within the last 24 hours
            # (skip dedup on first-ever check: sentinel 0.0)
            if state.last_expiration_notified_at > 0 and mono_now - state.last_expiration_notified_at < 86400:
                return
            state.last_expiration_notified_at = mono_now

        hours_until = (earliest_expiry - now).total_seconds() / 3600.0
        msg = _render_expiration_message(
            self._service_name, expiring_sats, hours_until,
        )
        self._send_nip44_dm(patron_npub, msg)

    def _send_nip44_dm(self, recipient_npub: str, message_text: str) -> None:
        """Send a NIP-44 encrypted kind 4 DM in a daemon thread.

        Uses NIP-04 kind 4 envelope with NIP-44v2 content encryption
        for forward secrecy. Fire-and-forget -- failures are logged
        but never block or raise.
        """
        if not self._enabled:
            return

        try:
            recipient_hex = PublicKey.from_npub(recipient_npub).hex()
        except Exception as exc:
            logger.debug("Invalid recipient npub %s: %s", recipient_npub[:20], exc)
            return

        try:
            encrypted = _nip44_encrypt(
                message_text, self._privkey_hex, recipient_hex,
            )

            event = Event(
                kind=_KIND_ENCRYPTED_DM,
                content=encrypted,
                tags=[
                    ["p", recipient_hex],
                    ["encrypted", "nip44"],
                ],
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

            logger.info(
                "Sent NIP-44 notification DM to %s via %d relays",
                recipient_npub[:20], len(self._relays),
            )
        except Exception as exc:
            logger.debug(
                "Failed to send notification DM to %s (non-fatal): %s",
                recipient_npub[:20], exc,
            )

    def _publish_to_relays(self, message: str) -> None:
        """Send event to all configured relays. Catches all exceptions."""
        sslopt: dict[str, Any] = {}
        for relay_url in self._relays:
            try:
                ws = create_connection(
                    relay_url, timeout=_RELAY_TIMEOUT, sslopt=sslopt,
                )
                try:
                    ws.send(message)
                    ws.recv()
                finally:
                    ws.close()
            except Exception as exc:
                logger.debug(
                    "Relay publish %s failed (non-fatal): %s", relay_url, exc,
                )
