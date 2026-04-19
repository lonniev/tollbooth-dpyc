"""Tests for tollbooth.constraints.pricing — all pricing constraints."""

from datetime import datetime, timezone


from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.pricing import (
    BulkBonusConstraint,
    CouponConstraint,
    FreeTrialConstraint,
    HappyHourConstraint,
    LoyaltyDiscountConstraint,
)


def _ctx(
    utc_now=None,
    invocation_count=0,
    total_consumed=0,
    balance=0,
):
    if utc_now is None:
        utc_now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    return ConstraintContext(
        ledger=LedgerSnapshot(
            balance_api_sats=balance,
            total_consumed_api_sats=total_consumed,
        ),
        patron=PatronIdentity(),
        env=EnvironmentSnapshot(
            utc_now=utc_now,
            invocation_count=invocation_count,
        ),
    )


# ---------------------------------------------------------------------------
# CouponConstraint
# ---------------------------------------------------------------------------


class TestCouponConstraint:
    def test_valid_coupon_percent(self):
        c = CouponConstraint(code="SAVE50", discount_percent=50.0)
        result = c.evaluate(_ctx())
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.discount_percent == 50.0

    def test_valid_coupon_sats(self):
        c = CouponConstraint(code="FLAT100", discount_sats=100)
        result = c.evaluate(_ctx())
        assert result.allowed is True
        assert result.price_modifier.discount_sats == 100

    def test_valid_coupon_free(self):
        c = CouponConstraint(code="FREEBIE", free=True)
        result = c.evaluate(_ctx())
        assert result.allowed is True
        assert result.price_modifier.free is True

    def test_expired_coupon(self):
        c = CouponConstraint(
            code="OLD",
            discount_percent=25.0,
            expires_at="2026-01-01T00:00:00Z",
        )
        result = c.evaluate(_ctx())  # March 2026 > Jan 2026
        assert result.allowed is False
        assert result.reason == "coupon_expired"

    def test_future_expiry_ok(self):
        c = CouponConstraint(
            code="FUTURE",
            discount_percent=10.0,
            expires_at="2027-01-01T00:00:00Z",
        )
        result = c.evaluate(_ctx())
        assert result.allowed is True

    def test_global_cap_exhausted(self):
        c = CouponConstraint(
            code="LIMITED",
            discount_percent=30.0,
            max_redemptions=100,
            current_redemptions=100,
        )
        result = c.evaluate(_ctx())
        assert result.allowed is False
        assert result.reason == "coupon_exhausted"

    def test_global_cap_within(self):
        c = CouponConstraint(
            code="LIMITED",
            discount_percent=30.0,
            max_redemptions=100,
            current_redemptions=50,
        )
        result = c.evaluate(_ctx())
        assert result.allowed is True

    def test_patron_cap_exhausted(self):
        c = CouponConstraint(
            code="ONCE",
            discount_percent=20.0,
            max_per_patron=1,
            patron_redemptions=1,
        )
        result = c.evaluate(_ctx())
        assert result.allowed is False
        assert result.reason == "coupon_patron_limit"

    def test_patron_cap_within(self):
        c = CouponConstraint(
            code="TWICE",
            discount_percent=20.0,
            max_per_patron=2,
            patron_redemptions=1,
        )
        result = c.evaluate(_ctx())
        assert result.allowed is True

    def test_metadata_includes_code(self):
        c = CouponConstraint(code="META", discount_percent=5.0)
        result = c.evaluate(_ctx())
        assert result.metadata["coupon_code"] == "META"

    def test_no_timezone_expiry_treated_as_utc(self):
        # Naive datetime in expires_at should be treated as UTC
        c = CouponConstraint(
            code="NAIVE",
            discount_percent=10.0,
            expires_at="2026-01-01T00:00:00",  # no Z
        )
        result = c.evaluate(_ctx())
        assert result.allowed is False


class TestCouponSerialization:
    def test_to_dict(self):
        c = CouponConstraint(
            code="TEST",
            discount_percent=25.0,
            expires_at="2026-12-31",
            max_redemptions=50,
        )
        d = c.to_dict()
        assert d["type"] == "coupon"
        assert d["code"] == "TEST"
        assert d["discount_percent"] == 25.0
        assert d["expires_at"] == "2026-12-31"

    def test_round_trip(self):
        c = CouponConstraint(
            code="RT", discount_sats=200, max_per_patron=3, patron_redemptions=1
        )
        restored = CouponConstraint.from_dict(c.to_dict())
        assert restored.code == "RT"
        assert restored.discount_sats == 200
        assert restored.max_per_patron == 3

    def test_describe(self):
        c = CouponConstraint(code="HALF", discount_percent=50.0, expires_at="2026-12-31")
        desc = c.describe()
        assert "HALF" in desc
        assert "50" in desc


# ---------------------------------------------------------------------------
# FreeTrialConstraint
# ---------------------------------------------------------------------------


class TestFreeTrialConstraint:
    def test_within_trial(self):
        c = FreeTrialConstraint(first_n_free=5)
        result = c.evaluate(_ctx(invocation_count=3))
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.free is True
        assert result.metadata["trial_remaining"] == 2

    def test_first_invocation(self):
        c = FreeTrialConstraint(first_n_free=3)
        result = c.evaluate(_ctx(invocation_count=0))
        assert result.allowed is True
        assert result.price_modifier.free is True
        assert result.metadata["trial_remaining"] == 3

    def test_trial_exhausted(self):
        c = FreeTrialConstraint(first_n_free=3)
        result = c.evaluate(_ctx(invocation_count=3))
        assert result.allowed is True  # Still allowed, just not free
        assert result.price_modifier is None
        assert result.metadata["trial_remaining"] == 0

    def test_well_past_trial(self):
        c = FreeTrialConstraint(first_n_free=3)
        result = c.evaluate(_ctx(invocation_count=100))
        assert result.allowed is True
        assert result.price_modifier is None


class TestFreeTrialSerialization:
    def test_to_dict(self):
        c = FreeTrialConstraint(first_n_free=10)
        d = c.to_dict()
        assert d["type"] == "free_trial"
        assert d["first_n_free"] == 10

    def test_round_trip(self):
        c = FreeTrialConstraint(first_n_free=7)
        restored = FreeTrialConstraint.from_dict(c.to_dict())
        assert restored.first_n_free == 7

    def test_describe(self):
        c = FreeTrialConstraint(first_n_free=5)
        assert "5" in c.describe()


# ---------------------------------------------------------------------------
# LoyaltyDiscountConstraint
# ---------------------------------------------------------------------------


class TestLoyaltyDiscountConstraint:
    def test_qualified(self):
        c = LoyaltyDiscountConstraint(
            threshold_consumed_api_sats=1000,
            discount_percent=15.0,
        )
        result = c.evaluate(_ctx(total_consumed=1500))
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.discount_percent == 15.0
        assert result.metadata["loyalty_qualified"] is True

    def test_at_threshold(self):
        c = LoyaltyDiscountConstraint(
            threshold_consumed_api_sats=1000,
            discount_percent=10.0,
        )
        result = c.evaluate(_ctx(total_consumed=1000))
        assert result.price_modifier is not None

    def test_not_qualified(self):
        c = LoyaltyDiscountConstraint(
            threshold_consumed_api_sats=1000,
            discount_percent=10.0,
        )
        result = c.evaluate(_ctx(total_consumed=500))
        assert result.allowed is True
        assert result.price_modifier is None
        assert result.metadata["loyalty_qualified"] is False
        assert result.metadata["loyalty_remaining"] == 500


class TestLoyaltySerialization:
    def test_round_trip(self):
        c = LoyaltyDiscountConstraint(
            threshold_consumed_api_sats=5000,
            discount_percent=20.0,
        )
        restored = LoyaltyDiscountConstraint.from_dict(c.to_dict())
        assert restored.threshold_consumed_api_sats == 5000
        assert restored.discount_percent == 20.0


# ---------------------------------------------------------------------------
# BulkBonusConstraint
# ---------------------------------------------------------------------------


class TestBulkBonusConstraint:
    def test_best_tier_matched(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
            {"min_consumed": 500, "bonus_multiplier": 1.25},
            {"min_consumed": 1000, "bonus_multiplier": 1.5},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=750))
        assert result.price_modifier is not None
        assert result.price_modifier.bonus_multiplier == 1.25

    def test_top_tier(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
            {"min_consumed": 1000, "bonus_multiplier": 1.5},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=5000))
        assert result.price_modifier.bonus_multiplier == 1.5

    def test_no_tier_matched(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=50))
        assert result.price_modifier is None

    def test_at_tier_boundary(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=100))
        assert result.price_modifier.bonus_multiplier == 1.1

    def test_single_tier(self):
        tiers = [{"min_consumed": 0, "bonus_multiplier": 1.05}]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=0))
        assert result.price_modifier.bonus_multiplier == 1.05


class TestBulkBonusSerialization:
    def test_round_trip(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
            {"min_consumed": 500, "bonus_multiplier": 1.25},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        d = c.to_dict()
        assert d["type"] == "bulk_bonus"
        # Should be sorted by min_consumed ascending
        assert d["tiers"][0]["min_consumed"] == 100
        assert d["tiers"][1]["min_consumed"] == 500

        restored = BulkBonusConstraint.from_dict(d)
        result = restored.evaluate(_ctx(total_consumed=200))
        assert result.price_modifier.bonus_multiplier == 1.1


# ---------------------------------------------------------------------------
# HappyHourConstraint
# ---------------------------------------------------------------------------


class TestHappyHourConstraint:
    def test_during_happy_hour(self):
        c = HappyHourConstraint(
            in_effect="11:00", until="14:00",
            percent_off=50.0,
        )
        result = c.evaluate(_ctx())  # 12:00 UTC
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.discount_percent == 50.0
        assert result.metadata["happy_hour_active"] is True

    def test_outside_happy_hour(self):
        c = HappyHourConstraint(
            in_effect="14:00", until="16:00",
            percent_off=50.0,
        )
        result = c.evaluate(_ctx())  # 12:00 UTC
        assert result.allowed is True  # Still allowed
        assert result.price_modifier is None
        assert result.metadata["happy_hour_active"] is False

    def test_free_happy_hour(self):
        c = HappyHourConstraint(
            in_effect="11:00", until="14:00",
            free=True,
        )
        result = c.evaluate(_ctx())
        assert result.price_modifier.free is True

    def test_with_timezone(self):
        # 12:00 UTC = 07:00 US/Eastern (EST)
        c = HappyHourConstraint(
            in_effect="08:00", until="17:00",
            timezone="US/Eastern",
            percent_off=25.0,
        )
        result = c.evaluate(_ctx())  # 07:00 Eastern -> outside
        assert result.metadata["happy_hour_active"] is False

    def test_with_days_of_week(self):
        # 2026-03-01 is Sunday (weekday=6)
        c = HappyHourConstraint(
            in_effect="00:00", until="23:59",
            days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri only
            percent_off=30.0,
        )
        result = c.evaluate(_ctx())  # Sunday
        assert result.allowed is True  # Still allowed, just no discount
        assert result.metadata["happy_hour_active"] is False


class TestHappyHourSerialization:
    def test_to_dict(self):
        c = HappyHourConstraint(
            in_effect="11:00", until="14:00",
            timezone="US/Eastern",
            percent_off=25.0,
        )
        d = c.to_dict()
        assert d["type"] == "happy_hour"
        assert d["in_effect"] == "11:00"
        assert d["until"] == "14:00"
        assert d["percent_off"] == 25.0

    def test_round_trip(self):
        c = HappyHourConstraint(
            in_effect="18:00", until="22:00",
            timezone="Europe/Berlin",
            days_of_week=[4, 5],
            max_discount=50,
        )
        restored = HappyHourConstraint.from_dict(c.to_dict())
        assert restored.in_effect == "18:00"
        assert restored.until == "22:00"
        assert restored.timezone == "Europe/Berlin"
        assert restored.days_of_week == [4, 5]
        assert restored.max_discount == 50

    def test_describe(self):
        c = HappyHourConstraint(in_effect="17:00", until="19:00", percent_off=40.0)
        desc = c.describe()
        assert "17:00-19:00" in desc
        assert "40" in desc
