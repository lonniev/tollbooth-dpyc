"""Tests for tollbooth.constraints.temporal — TemporalWindowConstraint."""

from datetime import UTC, datetime

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.temporal import TemporalWindowConstraint


def _ctx(year=2026, month=3, day=2, hour=12, minute=0):
    """Build a context with a specific UTC time.  2026-03-02 is a Monday."""
    utc_now = datetime(year, month, day, hour, minute, tzinfo=UTC)
    return ConstraintContext(
        ledger=LedgerSnapshot(),
        patron=PatronIdentity(),
        env=EnvironmentSnapshot(utc_now=utc_now),
    )


class TestWindowMatching:
    def test_inside_window(self):
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00")
        result = c.evaluate(_ctx(hour=12))
        assert result.allowed is True

    def test_outside_window_before(self):
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00")
        result = c.evaluate(_ctx(hour=6))
        assert result.allowed is False
        assert result.reason == "outside_window"
        assert result.retry_after is not None

    def test_outside_window_after(self):
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00")
        result = c.evaluate(_ctx(hour=22))
        assert result.allowed is False

    def test_at_window_start_boundary(self):
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00")
        result = c.evaluate(_ctx(hour=8, minute=0))
        assert result.allowed is True

    def test_at_window_end_boundary(self):
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00")
        result = c.evaluate(_ctx(hour=21, minute=0))
        # end is exclusive
        assert result.allowed is False


class TestMidnightWrap:
    def test_overnight_window_during(self):
        c = TemporalWindowConstraint(schedule_start="22:00", schedule_end="06:00")
        result = c.evaluate(_ctx(hour=23))
        assert result.allowed is True

    def test_overnight_window_early_morning(self):
        c = TemporalWindowConstraint(schedule_start="22:00", schedule_end="06:00")
        result = c.evaluate(_ctx(hour=3))
        assert result.allowed is True

    def test_overnight_window_outside(self):
        c = TemporalWindowConstraint(schedule_start="22:00", schedule_end="06:00")
        result = c.evaluate(_ctx(hour=12))
        assert result.allowed is False


class TestTimezoneHandling:
    def test_us_eastern_conversion(self):
        # 17:00 UTC = 12:00 US/Eastern (EST, UTC-5)
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00", timezone="US/Eastern")
        result = c.evaluate(_ctx(hour=17))  # 12:00 Eastern
        assert result.allowed is True

    def test_us_eastern_outside(self):
        # 04:00 UTC = 23:00 US/Eastern (EST, UTC-5)
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00", timezone="US/Eastern")
        result = c.evaluate(_ctx(hour=4))  # 23:00 Eastern
        assert result.allowed is False


class TestDaysOfWeek:
    def test_allowed_day(self):
        # 2026-03-02 is Monday (weekday=0)
        c = TemporalWindowConstraint(
            schedule_start="00:00", schedule_end="23:59",
            days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri
        )
        result = c.evaluate(_ctx(day=2, hour=12))  # Monday
        assert result.allowed is True

    def test_blocked_day(self):
        # 2026-03-07 is Saturday (weekday=5)
        c = TemporalWindowConstraint(
            schedule_start="00:00", schedule_end="23:59",
            days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri only
        )
        result = c.evaluate(_ctx(day=7, hour=12))  # Saturday
        assert result.allowed is False
        assert result.reason == "outside_window"

    def test_retry_after_skips_to_next_valid_day(self):
        # Saturday -> should retry_after on Monday
        c = TemporalWindowConstraint(
            schedule_start="08:00", schedule_end="17:00",
            days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri
        )
        result = c.evaluate(_ctx(day=7, hour=12))  # Saturday
        assert result.retry_after is not None
        # Should be on Monday (day=9)
        retry_local = result.retry_after.astimezone(UTC)
        assert retry_local.weekday() == 0  # Monday


class TestRetryAfter:
    def test_retry_after_is_utc(self):
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00")
        result = c.evaluate(_ctx(hour=22))
        assert result.retry_after is not None
        assert result.retry_after.tzinfo is not None

    def test_retry_after_is_next_window_start(self):
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00")
        result = c.evaluate(_ctx(hour=22))
        assert result.retry_after is not None
        # Should be 08:00 on the next day
        assert result.retry_after.hour == 8
        assert result.retry_after.minute == 0


class TestSerialization:
    def test_to_dict(self):
        c = TemporalWindowConstraint(
            schedule_start="08:00", schedule_end="21:00",
            timezone="US/Eastern",
            days_of_week=[0, 1, 2],
        )
        d = c.to_dict()
        assert d["type"] == "temporal_window"
        assert d["schedule_start"] == "08:00"
        assert d["schedule_end"] == "21:00"
        assert d["timezone"] == "US/Eastern"
        assert d["days_of_week"] == [0, 1, 2]

    def test_to_dict_no_days(self):
        c = TemporalWindowConstraint(schedule_start="08:00", schedule_end="21:00")
        d = c.to_dict()
        assert "days_of_week" not in d

    def test_round_trip(self):
        c = TemporalWindowConstraint(
            schedule_start="09:00", schedule_end="17:00",
            timezone="Europe/London",
            days_of_week=[0, 4],
        )
        restored = TemporalWindowConstraint.from_dict(c.to_dict())
        assert restored.schedule_start == c.schedule_start
        assert restored.schedule_end == c.schedule_end
        assert restored.timezone == c.timezone
        assert restored.days_of_week == c.days_of_week

    def test_describe(self):
        c = TemporalWindowConstraint(
            schedule_start="08:00", schedule_end="21:00",
            timezone="UTC",
            days_of_week=[0, 4],
        )
        desc = c.describe()
        assert "08:00-21:00" in desc
        assert "Mon" in desc
        assert "Fri" in desc
