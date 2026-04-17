"""Per-user credit ledger for tool-call metering.

Pure data model — no I/O. All api_sats values are integer API credits
(not real Bitcoin satoshis). Real BTC amounts use ``amount_sats`` and
only appear in invoice/BTCPay contexts.

Credits are stored as ordered tranches. Each tranche has an optional
TTL (time-to-live). Debits consume FIFO from the oldest non-expired
tranche. Expired tranches are collected lazily — their remaining
balance moves to ``total_expired_api_sats`` during debits and
serialization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 4


# ---------------------------------------------------------------------------
# ToolUsage
# ---------------------------------------------------------------------------


@dataclass
class ToolUsage:
    """Aggregate usage counter for a single tool."""

    calls: int = 0
    api_sats: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"calls": self.calls, "api_sats": self.api_sats}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolUsage:
        return cls(
            calls=int(data.get("calls", 0)),
            # Migration: accept old "sats" key or new "api_sats"
            api_sats=int(data.get("api_sats", data.get("sats", 0))),
        )


# ---------------------------------------------------------------------------
# InvoiceRecord
# ---------------------------------------------------------------------------


@dataclass
class InvoiceRecord:
    """Append-only record of a single BTCPay invoice."""

    invoice_id: str
    amount_sats: int  # Real BTC satoshis (never rename to api_sats)
    api_sats_credited: int = 0  # Multiplied credits granted
    multiplier: int = 1
    status: str = "Pending"  # Pending | Settled | Expired | Invalid
    created_at: str = ""  # ISO datetime
    settled_at: str | None = None
    btcpay_status: str | None = None  # Raw BTCPay status string

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "amount_sats": self.amount_sats,
            "api_sats_credited": self.api_sats_credited,
            "multiplier": self.multiplier,
            "status": self.status,
            "created_at": self.created_at,
            "settled_at": self.settled_at,
            "btcpay_status": self.btcpay_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvoiceRecord:
        return cls(
            invoice_id=str(data.get("invoice_id", "")),
            amount_sats=int(data.get("amount_sats", 0)),
            api_sats_credited=int(data.get("api_sats_credited", 0)),
            multiplier=int(data.get("multiplier", 1)),
            status=str(data.get("status", "Pending")),
            created_at=str(data.get("created_at", "")),
            settled_at=data.get("settled_at"),
            btcpay_status=data.get("btcpay_status"),
        )


# ---------------------------------------------------------------------------
# Tranche
# ---------------------------------------------------------------------------


@dataclass
class Tranche:
    """A discrete credit allocation with optional expiration.

    Tranches are consumed FIFO — oldest first. Once remaining_sats
    reaches 0, the tranche is inert. Expired tranches have their
    remaining balance swept into total_expired_api_sats during debits.
    """

    granted_at: str  # ISO datetime
    original_sats: int
    remaining_sats: int
    invoice_id: str = ""
    expires_at: str | None = None  # ISO datetime, None = never expires

    def __post_init__(self) -> None:
        # Defensive: ensure numeric fields are int even if deserialized as str
        self.original_sats = int(self.original_sats)
        self.remaining_sats = int(self.remaining_sats)

    def is_expired_at(self, now: datetime) -> bool:
        """Check if this tranche is expired at the given time."""
        if self.expires_at is None:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return now >= exp
        except (ValueError, TypeError):
            return False  # Malformed → treat as never-expiring

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "granted_at": self.granted_at,
            "original_sats": self.original_sats,
            "remaining_sats": self.remaining_sats,
            "invoice_id": self.invoice_id,
        }
        if self.expires_at is not None:
            d["expires_at"] = self.expires_at
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tranche:
        return cls(
            granted_at=str(data.get("granted_at", "")),
            original_sats=int(data.get("original_sats", 0)),
            remaining_sats=int(data.get("remaining_sats", 0)),
            invoice_id=str(data.get("invoice_id", "")),
            expires_at=data.get("expires_at"),
        )


# ---------------------------------------------------------------------------
# UserLedger
# ---------------------------------------------------------------------------


@dataclass
class UserLedger:
    """Per-user credit balance with tranche-based expiration.

    Credits live in ordered tranches. ``balance_api_sats`` is a computed
    property that sums non-expired tranche balances. Debits draw FIFO
    from the oldest non-expired tranche. Rollbacks create compensating
    tranches (never-expiring) rather than modifying existing ones.

    ``from_json()`` returns a fresh ledger on corrupt data (never blocks a user).
    """

    tranches: list[Tranche] = field(default_factory=list)
    total_deposited_api_sats: int = 0
    total_consumed_api_sats: int = 0
    total_expired_api_sats: int = 0
    pending_invoices: list[str] = field(default_factory=list)
    credited_invoices: list[str] = field(default_factory=list)
    last_deposit_at: str | None = None
    daily_log: dict[str, dict[str, ToolUsage]] = field(default_factory=dict)
    history: dict[str, ToolUsage] = field(default_factory=dict)
    invoices: dict[str, InvoiceRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.total_deposited_api_sats = int(self.total_deposited_api_sats)
        self.total_consumed_api_sats = int(self.total_consumed_api_sats)
        self.total_expired_api_sats = int(self.total_expired_api_sats)

    @property
    def balance_api_sats(self) -> int:
        """Sum of remaining_sats in non-expired tranches (read-only, no mutation)."""
        now = datetime.now(timezone.utc)
        return sum(
            t.remaining_sats for t in self.tranches
            if t.remaining_sats > 0 and not t.is_expired_at(now)
        )

    # -- invoice record helpers ------------------------------------------------

    def record_invoice_created(
        self, invoice_id: str, amount_sats: int, multiplier: int, created_at: str,
    ) -> None:
        """Record a newly created invoice (Pending status)."""
        self.invoices[invoice_id] = InvoiceRecord(
            invoice_id=invoice_id,
            amount_sats=amount_sats,
            multiplier=multiplier,
            status="Pending",
            created_at=created_at,
            btcpay_status="New",
        )

    def record_invoice_settled(
        self,
        invoice_id: str,
        api_sats_credited: int,
        settled_at: str,
        btcpay_status: str = "Settled",
    ) -> None:
        """Update an existing invoice record to Settled with credit info."""
        rec = self.invoices.get(invoice_id)
        if rec:
            rec.status = "Settled"
            rec.api_sats_credited = api_sats_credited
            rec.settled_at = settled_at
            rec.btcpay_status = btcpay_status
        else:
            self.invoices[invoice_id] = InvoiceRecord(
                invoice_id=invoice_id,
                amount_sats=0,  # Unknown — wasn't tracked at creation
                api_sats_credited=api_sats_credited,
                multiplier=0,  # Unknown
                status="Settled",
                created_at="",  # Unknown
                settled_at=settled_at,
                btcpay_status=btcpay_status,
            )

    def record_invoice_terminal(
        self, invoice_id: str, status: str, btcpay_status: str,
    ) -> None:
        """Update an existing invoice record to a terminal state (Expired/Invalid)."""
        rec = self.invoices.get(invoice_id)
        if rec:
            rec.status = status
            rec.btcpay_status = btcpay_status

    # -- tranche lifetime ------------------------------------------------------

    def apply_tranche_lifetime(self, tranche_lifetime_seconds: int | None) -> None:
        """Retroactively stamp ``expires_at`` on tranches that lack one.

        When an operator introduces a tranche lifetime, existing tranches
        that were minted without expiration acquire one based on their true
        age: ``granted_at + tranche_lifetime_seconds``.  A tranche older
        than the lifetime is already expired and will be collected on the
        next debit.  This is a one-time mutation per tranche — once
        ``expires_at`` is set, it is persisted and never recalculated.
        """
        if tranche_lifetime_seconds is None:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        for t in self.tranches:
            if t.expires_at is not None or t.remaining_sats <= 0 or not t.granted_at:
                continue
            try:
                granted = datetime.fromisoformat(t.granted_at.replace("Z", "+00:00"))
                t.expires_at = (granted + timedelta(seconds=tranche_lifetime_seconds)).isoformat()
            except (ValueError, TypeError):
                # Malformed granted_at — expire immediately. An unverifiable
                # tranche must not be trusted with a lifetime extension.
                logger.warning(
                    "Tranche with malformed granted_at=%r expired immediately",
                    t.granted_at,
                )
                t.expires_at = now_iso

    # -- mutations ------------------------------------------------------------

    def _collect_expired(self, now: datetime | None = None) -> int:
        """Sweep expired tranches: move their remaining balance to total_expired_api_sats.

        Returns the number of sats newly collected.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        collected = 0
        for t in self.tranches:
            if t.remaining_sats > 0 and t.is_expired_at(now):
                collected += t.remaining_sats
                t.remaining_sats = 0
        self.total_expired_api_sats += collected
        return collected

    def debit(self, tool_name: str, api_sats: int) -> bool:
        """Deduct api_sats via FIFO from oldest non-expired tranche.

        Returns False if insufficient balance (no partial debit).
        """
        if api_sats < 0:
            return False

        now = datetime.now(timezone.utc)
        self._collect_expired(now)

        # Compute available from non-expired tranches
        available = sum(
            t.remaining_sats for t in self.tranches
            if t.remaining_sats > 0 and not t.is_expired_at(now)
        )
        if available < api_sats:
            return False

        # FIFO draw
        remaining_to_debit = api_sats
        for t in self.tranches:
            if remaining_to_debit <= 0:
                break
            if t.remaining_sats <= 0 or t.is_expired_at(now):
                continue
            draw = min(remaining_to_debit, t.remaining_sats)
            t.remaining_sats -= draw
            remaining_to_debit -= draw

        self.total_consumed_api_sats += api_sats

        today = date.today().isoformat()
        day_log = self.daily_log.setdefault(today, {})
        usage = day_log.setdefault(tool_name, ToolUsage())
        usage.calls += 1
        usage.api_sats += api_sats

        agg = self.history.setdefault(tool_name, ToolUsage())
        agg.calls += 1
        agg.api_sats += api_sats

        return True

    def credit_deposit(
        self, api_sats: int, invoice_id: str, ttl_seconds: int | None = None,
    ) -> None:
        """Add credits as a new tranche. ttl_seconds=None means no expiration."""
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds else None
        self.tranches.append(Tranche(
            granted_at=now.isoformat(),
            original_sats=api_sats,
            remaining_sats=api_sats,
            invoice_id=invoice_id,
            expires_at=expires_at,
        ))
        self.total_deposited_api_sats += api_sats
        self.last_deposit_at = date.today().isoformat()
        if invoice_id in self.pending_invoices:
            self.pending_invoices.remove(invoice_id)
        if invoice_id not in self.credited_invoices:
            self.credited_invoices.append(invoice_id)

    def rollback_debit(self, tool_name: str, api_sats: int) -> None:
        """Undo a previous debit by adding sats back to the soonest-expiring tranche."""
        if api_sats <= 0:
            return
        now = datetime.now(timezone.utc)
        # Find the active tranche expiring soonest; fall back to any active tranche
        active = [t for t in self.tranches if t.remaining_sats > 0 and not t.is_expired_at(now)]
        target = None
        if active:
            with_expiry = [t for t in active if t.expires_at is not None]
            target = min(with_expiry, key=lambda t: t.expires_at) if with_expiry else active[0]
        if target:
            target.remaining_sats += api_sats
            target.original_sats += api_sats
        else:
            # No active tranches — create one with 7-day TTL
            self.tranches.append(Tranche(
                granted_at=now.isoformat(),
                original_sats=api_sats,
                remaining_sats=api_sats,
                invoice_id=f"rollback:{tool_name}",
                expires_at=(now + timedelta(days=7)).isoformat(),
            ))
        self.total_consumed_api_sats -= api_sats

        today = date.today().isoformat()
        day_log = self.daily_log.get(today, {})
        usage = day_log.get(tool_name)
        if usage:
            usage.calls = max(0, usage.calls - 1)
            usage.api_sats = max(0, usage.api_sats - api_sats)

        agg = self.history.get(tool_name)
        if agg:
            agg.calls = max(0, agg.calls - 1)
            agg.api_sats = max(0, agg.api_sats - api_sats)

    def rotate_daily_log(self, retention_days: int = 30) -> None:
        """Fold daily entries older than ``retention_days`` into ``history``."""
        cutoff = (date.today() - timedelta(days=retention_days)).isoformat()
        expired_keys = [d for d in self.daily_log if d < cutoff]
        for day_key in expired_keys:
            del self.daily_log[day_key]

    # -- expiration analytics -------------------------------------------------

    def expiring_within(self, seconds: int) -> int:
        """Sum of remaining_sats in tranches expiring within the given window."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(seconds=seconds)
        total = 0
        for t in self.tranches:
            if t.remaining_sats <= 0 or t.is_expired_at(now):
                continue
            if t.expires_at is not None:
                try:
                    exp = datetime.fromisoformat(t.expires_at.replace("Z", "+00:00"))
                    if exp <= cutoff:
                        total += t.remaining_sats
                except (ValueError, TypeError):
                    pass
        return total

    def next_expiration(self) -> str | None:
        """ISO datetime of the earliest tranche expiration, or None."""
        now = datetime.now(timezone.utc)
        earliest: datetime | None = None
        for t in self.tranches:
            if t.remaining_sats <= 0 or t.is_expired_at(now):
                continue
            if t.expires_at is not None:
                try:
                    exp = datetime.fromisoformat(t.expires_at.replace("Z", "+00:00"))
                    if earliest is None or exp < earliest:
                        earliest = exp
                except (ValueError, TypeError):
                    pass
        return earliest.isoformat() if earliest else None

    # -- serialization --------------------------------------------------------

    def to_json(self) -> str:
        """Serialize to JSON (schema v4). Prunes empty/expired tranches."""
        self._collect_expired()
        # Prune fully consumed tranches
        active = [t for t in self.tranches if t.remaining_sats > 0]
        return json.dumps({
            "v": _SCHEMA_VERSION,
            "tranches": [t.to_dict() for t in active],
            "total_deposited_api_sats": self.total_deposited_api_sats,
            "total_consumed_api_sats": self.total_consumed_api_sats,
            "total_expired_api_sats": self.total_expired_api_sats,
            "pending_invoices": self.pending_invoices,
            "credited_invoices": self.credited_invoices,
            "last_deposit_at": self.last_deposit_at,
            "daily_log": {
                day: {tool: u.to_dict() for tool, u in tools.items()}
                for day, tools in self.daily_log.items()
            },
            "history": {
                tool: u.to_dict() for tool, u in self.history.items()
            },
            "invoices": {
                iid: rec.to_dict() for iid, rec in self.invoices.items()
            },
        }, indent=2)

    @classmethod
    def from_json(cls, data: str) -> UserLedger:
        """Deserialize from JSON. v4 only — non-v4 data returns a fresh ledger."""
        try:
            obj = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Ledger data is corrupt; returning fresh ledger.")
            return cls()

        if not isinstance(obj, dict):
            logger.warning("Ledger data is not a dict; returning fresh ledger.")
            return cls()

        version = obj.get("v", 0)
        if version != 4:
            logger.warning(
                "Ledger schema v%s is not supported (expected v4); returning fresh ledger.",
                version,
            )
            return cls()

        # Parse tranches
        tranches: list[Tranche] = []
        for t_data in obj.get("tranches", []):
            if isinstance(t_data, dict):
                tranches.append(Tranche.from_dict(t_data))

        daily_log: dict[str, dict[str, ToolUsage]] = {}
        raw_daily = obj.get("daily_log", {})
        if isinstance(raw_daily, dict):
            for day, tools in raw_daily.items():
                if isinstance(tools, dict):
                    daily_log[day] = {
                        t: ToolUsage.from_dict(u)
                        for t, u in tools.items()
                        if isinstance(u, dict)
                    }

        history: dict[str, ToolUsage] = {}
        raw_history = obj.get("history", {})
        if isinstance(raw_history, dict):
            history = {
                t: ToolUsage.from_dict(u)
                for t, u in raw_history.items()
                if isinstance(u, dict)
            }

        invoices: dict[str, InvoiceRecord] = {}
        raw_invoices = obj.get("invoices", {})
        if isinstance(raw_invoices, dict):
            for iid, rec_data in raw_invoices.items():
                if isinstance(rec_data, dict):
                    invoices[iid] = InvoiceRecord.from_dict(rec_data)

        return cls(
            tranches=tranches,
            total_deposited_api_sats=int(obj.get("total_deposited_api_sats", 0)),
            total_consumed_api_sats=int(obj.get("total_consumed_api_sats", 0)),
            total_expired_api_sats=int(obj.get("total_expired_api_sats", 0)),
            pending_invoices=list(obj.get("pending_invoices", [])),
            credited_invoices=list(obj.get("credited_invoices", [])),
            last_deposit_at=obj.get("last_deposit_at"),
            daily_log=daily_log,
            history=history,
            invoices=invoices,
        )
