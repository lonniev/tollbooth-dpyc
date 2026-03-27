"""Tests for UserLedger model with tranche-based credit expiration."""

import json
from datetime import date, datetime, timedelta, timezone


from tollbooth.ledger import Tranche, ToolUsage, UserLedger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc)
_PAST = (_NOW - timedelta(hours=1)).isoformat()
_FUTURE = (_NOW + timedelta(days=7)).isoformat()
_EXPIRED = (_NOW - timedelta(seconds=1)).isoformat()


def _tranche(
    sats: int,
    remaining: int | None = None,
    expires_at: str | None = None,
    invoice_id: str = "test-init",
) -> Tranche:
    """Create a test tranche with sensible defaults."""
    return Tranche(
        granted_at=_PAST,
        original_sats=sats,
        remaining_sats=remaining if remaining is not None else sats,
        invoice_id=invoice_id,
        expires_at=expires_at,
    )


def _ledger_with_balance(balance: int, **kwargs) -> UserLedger:
    """Create a UserLedger with initial balance as a single non-expiring tranche."""
    ledger = UserLedger(**kwargs)
    if balance > 0:
        ledger.tranches.append(_tranche(balance))
    return ledger


# ---------------------------------------------------------------------------
# ToolUsage
# ---------------------------------------------------------------------------


class TestToolUsage:
    def test_defaults(self) -> None:
        u = ToolUsage()
        assert u.calls == 0
        assert u.api_sats == 0

    def test_to_dict(self) -> None:
        u = ToolUsage(calls=5, api_sats=100)
        assert u.to_dict() == {"calls": 5, "api_sats": 100}

    def test_from_dict(self) -> None:
        u = ToolUsage.from_dict({"calls": 3, "api_sats": 42})
        assert u.calls == 3
        assert u.api_sats == 42

    def test_from_dict_missing_fields(self) -> None:
        u = ToolUsage.from_dict({})
        assert u.calls == 0
        assert u.api_sats == 0

    def test_roundtrip(self) -> None:
        original = ToolUsage(calls=10, api_sats=200)
        restored = ToolUsage.from_dict(original.to_dict())
        assert restored.calls == original.calls
        assert restored.api_sats == original.api_sats


# ---------------------------------------------------------------------------
# Tranche
# ---------------------------------------------------------------------------


class TestTranche:
    def test_not_expired_when_no_expiry(self) -> None:
        t = _tranche(100, expires_at=None)
        assert t.is_expired_at(_NOW) is False

    def test_not_expired_when_future(self) -> None:
        t = _tranche(100, expires_at=_FUTURE)
        assert t.is_expired_at(_NOW) is False

    def test_expired_when_past(self) -> None:
        t = _tranche(100, expires_at=_EXPIRED)
        assert t.is_expired_at(_NOW) is True

    def test_malformed_expires_at_treated_as_never(self) -> None:
        t = _tranche(100, expires_at="not-a-datetime")
        assert t.is_expired_at(_NOW) is False

    def test_to_dict_omits_expires_at_when_none(self) -> None:
        t = _tranche(100, expires_at=None)
        d = t.to_dict()
        assert "expires_at" not in d

    def test_to_dict_includes_expires_at(self) -> None:
        t = _tranche(100, expires_at=_FUTURE)
        d = t.to_dict()
        assert d["expires_at"] == _FUTURE

    def test_roundtrip(self) -> None:
        t = _tranche(100, remaining=75, expires_at=_FUTURE, invoice_id="inv-42")
        restored = Tranche.from_dict(t.to_dict())
        assert restored.original_sats == 100
        assert restored.remaining_sats == 75
        assert restored.expires_at == _FUTURE
        assert restored.invoice_id == "inv-42"


# ---------------------------------------------------------------------------
# UserLedger — balance property
# ---------------------------------------------------------------------------


class TestBalance:
    def test_empty_ledger_zero_balance(self) -> None:
        ledger = UserLedger()
        assert ledger.balance_api_sats == 0

    def test_single_tranche(self) -> None:
        ledger = _ledger_with_balance(100)
        assert ledger.balance_api_sats == 100

    def test_multiple_tranches(self) -> None:
        ledger = UserLedger(tranches=[_tranche(50), _tranche(30)])
        assert ledger.balance_api_sats == 80

    def test_expired_tranche_excluded(self) -> None:
        ledger = UserLedger(tranches=[
            _tranche(100, expires_at=_EXPIRED),
            _tranche(50),
        ])
        assert ledger.balance_api_sats == 50

    def test_empty_tranche_excluded(self) -> None:
        ledger = UserLedger(tranches=[_tranche(100, remaining=0)])
        assert ledger.balance_api_sats == 0


# ---------------------------------------------------------------------------
# UserLedger — debit / credit / rollback
# ---------------------------------------------------------------------------


class TestUserLedger:
    def test_debit_success(self) -> None:
        ledger = _ledger_with_balance(100)
        assert ledger.debit("search", 30) is True
        assert ledger.balance_api_sats == 70
        assert ledger.total_consumed_api_sats == 30

    def test_debit_insufficient_balance(self) -> None:
        ledger = _ledger_with_balance(10)
        assert ledger.debit("search", 20) is False
        assert ledger.balance_api_sats == 10
        assert ledger.total_consumed_api_sats == 0

    def test_debit_exact_balance(self) -> None:
        ledger = _ledger_with_balance(50)
        assert ledger.debit("search", 50) is True
        assert ledger.balance_api_sats == 0

    def test_debit_negative_amount_rejected(self) -> None:
        ledger = _ledger_with_balance(100)
        assert ledger.debit("search", -5) is False
        assert ledger.balance_api_sats == 100

    def test_debit_zero(self) -> None:
        ledger = _ledger_with_balance(100)
        assert ledger.debit("search", 0) is True
        assert ledger.balance_api_sats == 100

    def test_debit_updates_daily_log(self) -> None:
        ledger = _ledger_with_balance(100)
        ledger.debit("search", 10)
        today = date.today().isoformat()
        assert today in ledger.daily_log
        assert ledger.daily_log[today]["search"].calls == 1
        assert ledger.daily_log[today]["search"].api_sats == 10

    def test_debit_updates_history(self) -> None:
        ledger = _ledger_with_balance(100)
        ledger.debit("search", 10)
        ledger.debit("search", 20)
        assert ledger.history["search"].calls == 2
        assert ledger.history["search"].api_sats == 30

    def test_debit_fifo_across_tranches(self) -> None:
        """Debit draws from oldest tranche first, then spills to next."""
        t1 = _tranche(30, invoice_id="first")
        t2 = _tranche(70, invoice_id="second")
        ledger = UserLedger(tranches=[t1, t2])
        assert ledger.debit("search", 50) is True
        assert t1.remaining_sats == 0  # fully consumed
        assert t2.remaining_sats == 50  # 70 - 20
        assert ledger.balance_api_sats == 50

    def test_debit_skips_expired_tranches(self) -> None:
        """Expired tranches are skipped during FIFO draw."""
        t_expired = _tranche(100, expires_at=_EXPIRED, invoice_id="expired")
        t_fresh = _tranche(50, invoice_id="fresh")
        ledger = UserLedger(tranches=[t_expired, t_fresh])
        assert ledger.debit("search", 30) is True
        assert t_expired.remaining_sats == 0  # collected as expired
        assert t_fresh.remaining_sats == 20
        assert ledger.total_expired_api_sats == 100

    def test_debit_insufficient_after_expiration(self) -> None:
        """If only expired tranches exist, debit fails."""
        ledger = UserLedger(tranches=[_tranche(100, expires_at=_EXPIRED)])
        assert ledger.debit("search", 10) is False
        assert ledger.total_expired_api_sats == 100

    def test_credit_deposit(self) -> None:
        ledger = _ledger_with_balance(50, pending_invoices=["inv-1"])
        ledger.credit_deposit(100, "inv-1")
        assert ledger.balance_api_sats == 150
        assert ledger.total_deposited_api_sats == 100
        assert ledger.last_deposit_at == date.today().isoformat()
        assert "inv-1" not in ledger.pending_invoices
        assert len(ledger.tranches) == 2  # original + new

    def test_credit_deposit_with_ttl(self) -> None:
        """credit_deposit with ttl_seconds creates expiring tranche."""
        ledger = UserLedger()
        ledger.credit_deposit(500, "inv-1", ttl_seconds=3600)
        assert len(ledger.tranches) == 1
        t = ledger.tranches[0]
        assert t.original_sats == 500
        assert t.expires_at is not None

    def test_credit_deposit_without_ttl(self) -> None:
        """credit_deposit without ttl_seconds defaults to 7-day expiry."""
        ledger = UserLedger()
        ledger.credit_deposit(500, "inv-1")
        assert ledger.tranches[0].expires_at is not None

    def test_credit_deposit_unknown_invoice(self) -> None:
        ledger = UserLedger(pending_invoices=["inv-1"])
        ledger.credit_deposit(50, "inv-other")
        assert ledger.balance_api_sats == 50
        assert "inv-1" in ledger.pending_invoices

    def test_rollback_debit(self) -> None:
        ledger = _ledger_with_balance(100)
        ledger.debit("search", 30)
        assert ledger.balance_api_sats == 70
        ledger.rollback_debit("search", 30)
        assert ledger.balance_api_sats == 100
        assert ledger.total_consumed_api_sats == 0

    def test_rollback_adds_to_existing_tranche(self) -> None:
        """Rollback adds sats back to an existing tranche instead of creating a new one."""
        ledger = _ledger_with_balance(100)
        ledger.debit("search", 30)
        initial_count = len(ledger.tranches)
        ledger.rollback_debit("search", 30)
        assert len(ledger.tranches) == initial_count  # no new tranche
        assert ledger.tranches[0].remaining_sats == 100
        assert ledger.balance_api_sats == 100

    def test_rollback_clamps_to_zero(self) -> None:
        ledger = _ledger_with_balance(100)
        ledger.debit("search", 10)
        # Rollback more than was debited
        ledger.rollback_debit("search", 20)
        assert ledger.history["search"].calls == 0
        assert ledger.history["search"].api_sats == 0

    def test_seed_via_credit_deposit(self) -> None:
        """Seed balance via credit_deposit with sentinel ID."""
        ledger = UserLedger()
        ledger.credit_deposit(1000, "seed_balance_v1")
        assert ledger.balance_api_sats == 1000
        assert ledger.total_deposited_api_sats == 1000
        assert "seed_balance_v1" in ledger.credited_invoices

    def test_seed_sentinel_prevents_double_credit(self) -> None:
        """Second credit_deposit with same sentinel is a no-op for credited_invoices."""
        ledger = UserLedger()
        ledger.credit_deposit(1000, "seed_balance_v1")
        assert "seed_balance_v1" in ledger.credited_invoices
        assert ledger.credited_invoices.count("seed_balance_v1") == 1

    def test_seed_balance_is_spendable(self) -> None:
        """Seeded balance can be spent via debit()."""
        ledger = UserLedger()
        ledger.credit_deposit(1000, "seed_balance_v1")
        assert ledger.debit("search", 100) is True
        assert ledger.balance_api_sats == 900

    def test_rotate_daily_log(self) -> None:
        ledger = _ledger_with_balance(100)
        # Add an old entry
        ledger.daily_log["2020-01-01"] = {"search": ToolUsage(calls=5, api_sats=50)}
        # Add today's entry
        today = date.today().isoformat()
        ledger.daily_log[today] = {"search": ToolUsage(calls=1, api_sats=10)}
        ledger.rotate_daily_log(retention_days=30)
        assert "2020-01-01" not in ledger.daily_log
        assert today in ledger.daily_log


# ---------------------------------------------------------------------------
# Expiration collection
# ---------------------------------------------------------------------------


class TestCollectExpired:
    def test_collect_expired_sweeps_balance(self) -> None:
        t = _tranche(100, expires_at=_EXPIRED)
        ledger = UserLedger(tranches=[t])
        collected = ledger._collect_expired()
        assert collected == 100
        assert t.remaining_sats == 0
        assert ledger.total_expired_api_sats == 100

    def test_collect_expired_skips_non_expired(self) -> None:
        t = _tranche(100, expires_at=_FUTURE)
        ledger = UserLedger(tranches=[t])
        collected = ledger._collect_expired()
        assert collected == 0
        assert t.remaining_sats == 100
        assert ledger.total_expired_api_sats == 0

    def test_collect_expired_skips_never_expiring(self) -> None:
        t = _tranche(100, expires_at=None)
        ledger = UserLedger(tranches=[t])
        collected = ledger._collect_expired()
        assert collected == 0

    def test_collect_expired_idempotent(self) -> None:
        """Calling twice doesn't double-count."""
        t = _tranche(100, expires_at=_EXPIRED)
        ledger = UserLedger(tranches=[t])
        ledger._collect_expired()
        ledger._collect_expired()
        assert ledger.total_expired_api_sats == 100


# ---------------------------------------------------------------------------
# Expiration analytics
# ---------------------------------------------------------------------------


class TestExpirationAnalytics:
    def test_expiring_within_returns_sum(self) -> None:
        soon = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
        far = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        ledger = UserLedger(tranches=[
            _tranche(100, expires_at=soon),
            _tranche(200, expires_at=far),
            _tranche(50, expires_at=None),
        ])
        assert ledger.expiring_within(86400) == 100  # only the 12h tranche

    def test_expiring_within_empty(self) -> None:
        ledger = UserLedger(tranches=[_tranche(100, expires_at=None)])
        assert ledger.expiring_within(86400) == 0

    def test_next_expiration_returns_earliest(self) -> None:
        soon = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        later = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        ledger = UserLedger(tranches=[
            _tranche(100, expires_at=later),
            _tranche(50, expires_at=soon),
        ])
        result = ledger.next_expiration()
        assert result is not None
        assert result == soon

    def test_next_expiration_none_when_no_expiring(self) -> None:
        ledger = UserLedger(tranches=[_tranche(100, expires_at=None)])
        assert ledger.next_expiration() is None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestLedgerSerialization:
    def test_roundtrip(self) -> None:
        ledger = UserLedger()
        ledger.credit_deposit(500, "inv-1")
        ledger.credit_deposit(500, "inv-2", ttl_seconds=3600)
        ledger.debit("search", 100)
        restored = UserLedger.from_json(ledger.to_json())
        assert restored.balance_api_sats == ledger.balance_api_sats
        assert restored.total_deposited_api_sats == 1000
        assert restored.total_consumed_api_sats == 100
        assert "search" in restored.history
        assert restored.history["search"].calls == 1
        assert len(restored.tranches) == 2

    def test_schema_version(self) -> None:
        ledger = UserLedger()
        obj = json.loads(ledger.to_json())
        assert obj["v"] == 4

    def test_from_json_v3_returns_fresh(self) -> None:
        """v3 ledger data returns a fresh (empty) ledger."""
        restored = UserLedger.from_json('{"v": 3, "balance_api_sats": 500}')
        assert restored.balance_api_sats == 0
        assert restored.tranches == []

    def test_from_json_v1_returns_fresh(self) -> None:
        restored = UserLedger.from_json('{"v": 1}')
        assert restored.balance_api_sats == 0

    def test_from_json_corrupt_data(self) -> None:
        restored = UserLedger.from_json("not json at all")
        assert restored.balance_api_sats == 0

    def test_from_json_none(self) -> None:
        restored = UserLedger.from_json(None)  # type: ignore[arg-type]
        assert restored.balance_api_sats == 0

    def test_from_json_non_dict(self) -> None:
        restored = UserLedger.from_json('"just a string"')
        assert restored.balance_api_sats == 0

    def test_daily_log_survives_roundtrip(self) -> None:
        ledger = _ledger_with_balance(100)
        ledger.debit("search", 10)
        ledger.debit("create", 20)
        restored = UserLedger.from_json(ledger.to_json())
        today = date.today().isoformat()
        assert restored.daily_log[today]["search"].api_sats == 10
        assert restored.daily_log[today]["create"].api_sats == 20

    def test_pending_invoices_survive_roundtrip(self) -> None:
        ledger = UserLedger(pending_invoices=["inv-a", "inv-b"])
        restored = UserLedger.from_json(ledger.to_json())
        assert restored.pending_invoices == ["inv-a", "inv-b"]

    def test_to_json_is_pretty_printed(self) -> None:
        ledger = _ledger_with_balance(100)
        output = ledger.to_json()
        assert "\n" in output
        parsed = json.loads(output)
        assert "tranches" in parsed

    def test_to_json_prunes_empty_tranches(self) -> None:
        """Fully consumed tranches are not serialized."""
        ledger = _ledger_with_balance(100)
        ledger.debit("search", 100)
        obj = json.loads(ledger.to_json())
        assert len(obj["tranches"]) == 0

    def test_to_json_prunes_expired_tranches(self) -> None:
        """Expired tranches are collected and pruned during serialization."""
        ledger = UserLedger(tranches=[_tranche(100, expires_at=_EXPIRED)])
        obj = json.loads(ledger.to_json())
        assert len(obj["tranches"]) == 0
        assert obj["total_expired_api_sats"] == 100

    def test_tranches_survive_roundtrip(self) -> None:
        """Tranche details (including expires_at) survive serialization."""
        ledger = UserLedger()
        ledger.credit_deposit(200, "inv-1", ttl_seconds=7200)
        restored = UserLedger.from_json(ledger.to_json())
        assert len(restored.tranches) == 1
        assert restored.tranches[0].original_sats == 200
        assert restored.tranches[0].expires_at is not None

    def test_total_expired_survives_roundtrip(self) -> None:
        ledger = UserLedger(total_expired_api_sats=42)
        restored = UserLedger.from_json(ledger.to_json())
        assert restored.total_expired_api_sats == 42
