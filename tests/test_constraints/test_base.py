"""Tests for tollbooth.constraints.base — context, price modifier, result."""

from datetime import datetime, timezone

import pytest

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
    PriceModifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(**overrides):
    """Build a ConstraintContext with sensible defaults."""
    return ConstraintContext(
        ledger=overrides.get("ledger", LedgerSnapshot()),
        patron=overrides.get("patron", PatronIdentity()),
        env=overrides.get(
            "env",
            EnvironmentSnapshot(utc_now=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)),
        ),
    )


# ---------------------------------------------------------------------------
# LedgerSnapshot
# ---------------------------------------------------------------------------


class TestLedgerSnapshot:
    def test_defaults(self):
        snap = LedgerSnapshot()
        assert snap.balance_api_sats == 0
        assert snap.total_deposited_api_sats == 0
        assert snap.total_consumed_api_sats == 0
        assert snap.total_expired_api_sats == 0

    def test_frozen(self):
        snap = LedgerSnapshot(balance_api_sats=100)
        with pytest.raises(AttributeError):
            snap.balance_api_sats = 200  # type: ignore[misc]

    def test_custom_values(self):
        snap = LedgerSnapshot(
            balance_api_sats=500,
            total_deposited_api_sats=1000,
            total_consumed_api_sats=400,
            total_expired_api_sats=100,
        )
        assert snap.balance_api_sats == 500
        assert snap.total_deposited_api_sats == 1000


# ---------------------------------------------------------------------------
# PatronIdentity
# ---------------------------------------------------------------------------


class TestPatronIdentity:
    def test_defaults(self):
        p = PatronIdentity()
        assert p.npub == ""
        assert p.membership_tier == "default"
        assert p.join_date is None

    def test_custom(self):
        p = PatronIdentity(npub="npub1abc", membership_tier="gold", join_date="2026-01-01")
        assert p.npub == "npub1abc"
        assert p.membership_tier == "gold"


# ---------------------------------------------------------------------------
# EnvironmentSnapshot
# ---------------------------------------------------------------------------


class TestEnvironmentSnapshot:
    def test_requires_utc_now(self):
        now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
        env = EnvironmentSnapshot(utc_now=now)
        assert env.utc_now == now
        assert env.tool_name == ""
        assert env.invocation_count == 0

    def test_frozen(self):
        env = EnvironmentSnapshot(utc_now=datetime.now(tz=timezone.utc))
        with pytest.raises(AttributeError):
            env.tool_name = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ConstraintContext
# ---------------------------------------------------------------------------


class TestConstraintContext:
    def test_composition(self):
        ctx = _make_context()
        assert ctx.ledger.balance_api_sats == 0
        assert ctx.patron.npub == ""
        assert ctx.env.tool_name == ""

    def test_frozen(self):
        ctx = _make_context()
        with pytest.raises(AttributeError):
            ctx.ledger = LedgerSnapshot()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PriceModifier
# ---------------------------------------------------------------------------


class TestPriceModifier:
    def test_defaults(self):
        pm = PriceModifier()
        assert pm.discount_percent == 0.0
        assert pm.discount_sats == 0
        assert pm.free is False
        assert pm.bonus_multiplier == 1.0

    def test_apply_free(self):
        pm = PriceModifier(free=True)
        assert pm.apply_to(1000) == 0

    def test_apply_discount_percent(self):
        pm = PriceModifier(discount_percent=25.0)
        assert pm.apply_to(1000) == 750

    def test_apply_discount_sats(self):
        pm = PriceModifier(discount_sats=200)
        assert pm.apply_to(1000) == 800

    def test_apply_discount_sats_exceeds_price(self):
        pm = PriceModifier(discount_sats=2000)
        assert pm.apply_to(1000) == 0

    def test_apply_combined_discounts(self):
        pm = PriceModifier(discount_percent=50.0, discount_sats=100)
        # 1000 -> 50% off = 500 -> minus 100 = 400
        assert pm.apply_to(1000) == 400

    def test_apply_no_discount(self):
        pm = PriceModifier()
        assert pm.apply_to(500) == 500

    def test_free_overrides_all(self):
        pm = PriceModifier(free=True, discount_percent=50.0, discount_sats=100)
        assert pm.apply_to(1000) == 0

    def test_to_dict_minimal(self):
        pm = PriceModifier()
        assert pm.to_dict() == {}

    def test_to_dict_full(self):
        pm = PriceModifier(discount_percent=10.0, discount_sats=50, free=True, bonus_multiplier=1.5)
        d = pm.to_dict()
        assert d["discount_percent"] == 10.0
        assert d["discount_sats"] == 50
        assert d["free"] is True
        assert d["bonus_multiplier"] == 1.5

    def test_round_trip(self):
        pm = PriceModifier(discount_percent=20.0, bonus_multiplier=1.25)
        restored = PriceModifier.from_dict(pm.to_dict())
        assert restored.discount_percent == pm.discount_percent
        assert restored.bonus_multiplier == pm.bonus_multiplier

    def test_apply_bonus_multiplier(self):
        # 1.25x bonus means 25% more value per sat → effective cost drops
        pm = PriceModifier(bonus_multiplier=1.25)
        assert pm.apply_to(100) == 80  # 100 / 1.25 = 80

    def test_apply_bonus_floors_to_one(self):
        # Even huge bonus can't make cost < 1 (avoids free-by-accident)
        pm = PriceModifier(bonus_multiplier=100.0)
        assert pm.apply_to(5) == 1

    def test_apply_surge_then_bonus(self):
        # 2x surge, 2x bonus → they cancel out
        pm = PriceModifier(surge_multiplier=2.0, bonus_multiplier=2.0)
        assert pm.apply_to(100) == 100

    def test_apply_all_modifiers(self):
        # surge 2x, bonus 1.25x, 20% discount, 10 sats off
        pm = PriceModifier(
            surge_multiplier=2.0,
            bonus_multiplier=1.25,
            discount_percent=20.0,
            discount_sats=10,
        )
        # 100 * 2.0 = 200 surge
        # 200 / 1.25 = 160 bonus
        # 160 * 0.8 = 128 pct discount
        # 128 - 10 = 118 sats discount
        assert pm.apply_to(100) == 118

    def test_from_dict_defaults(self):
        pm = PriceModifier.from_dict({})
        assert pm.discount_percent == 0.0
        assert pm.free is False


# ---------------------------------------------------------------------------
# ConstraintResult
# ---------------------------------------------------------------------------


class TestConstraintResult:
    def test_defaults(self):
        r = ConstraintResult()
        assert r.allowed is True
        assert r.reason == ""
        assert r.message == ""
        assert r.retry_after is None
        assert r.price_modifier is None
        assert r.metadata == {}

    def test_denied(self):
        r = ConstraintResult(
            allowed=False,
            reason="test_denied",
            message="Not allowed.",
        )
        assert r.allowed is False
        assert r.reason == "test_denied"

    def test_metadata_mutable(self):
        r = ConstraintResult()
        r.metadata["key"] = "value"
        assert r.metadata["key"] == "value"

    def test_with_price_modifier(self):
        pm = PriceModifier(free=True)
        r = ConstraintResult(allowed=True, price_modifier=pm)
        assert r.price_modifier is not None
        assert r.price_modifier.free is True
