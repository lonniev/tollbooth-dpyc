"""Tests for tollbooth.constraints.periodic — PeriodicRefreshConstraint + ISO parser."""

from datetime import UTC, datetime, timedelta

import pytest

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.periodic import (
    PeriodicRefreshConstraint,
    parse_iso_duration,
)


def _ctx(utc_now=None):
    if utc_now is None:
        utc_now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    return ConstraintContext(
        ledger=LedgerSnapshot(),
        patron=PatronIdentity(),
        env=EnvironmentSnapshot(utc_now=utc_now),
    )


# ---------------------------------------------------------------------------
# parse_iso_duration
# ---------------------------------------------------------------------------


class TestParseIsoDuration:
    def test_hours(self):
        assert parse_iso_duration("PT5H") == timedelta(hours=5)

    def test_minutes(self):
        assert parse_iso_duration("PT30M") == timedelta(minutes=30)

    def test_seconds(self):
        assert parse_iso_duration("PT90S") == timedelta(seconds=90)

    def test_days(self):
        assert parse_iso_duration("P7D") == timedelta(days=7)

    def test_combined(self):
        assert parse_iso_duration("P1DT2H30M") == timedelta(days=1, hours=2, minutes=30)

    def test_years(self):
        assert parse_iso_duration("P1Y") == timedelta(days=365)

    def test_months(self):
        assert parse_iso_duration("P2M") == timedelta(days=60)

    def test_full_format(self):
        result = parse_iso_duration("P1Y2M3DT4H5M6S")
        expected = timedelta(days=365 + 60 + 3, hours=4, minutes=5, seconds=6)
        assert result == expected

    def test_lowercase(self):
        assert parse_iso_duration("pt1h") == timedelta(hours=1)

    def test_fractional_seconds(self):
        assert parse_iso_duration("PT1.5S") == timedelta(seconds=1.5)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid ISO 8601 duration"):
            parse_iso_duration("5 hours")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            parse_iso_duration("")

    def test_just_p(self):
        # "P" alone is technically valid (zero duration)
        result = parse_iso_duration("P")
        assert result == timedelta(0)

    def test_just_pt(self):
        result = parse_iso_duration("PT")
        assert result == timedelta(0)


# ---------------------------------------------------------------------------
# PeriodicRefreshConstraint
# ---------------------------------------------------------------------------


class TestPeriodicRefresh:
    def test_within_window_budget(self):
        c = PeriodicRefreshConstraint(
            max_per_window=10,
            window_duration="PT1H",
            current_window_count=5,
            window_start="2026-03-01T11:30:00Z",
        )
        result = c.evaluate(_ctx())
        assert result.allowed is True
        assert result.metadata["remaining_in_window"] == 5

    def test_window_exhausted(self):
        c = PeriodicRefreshConstraint(
            max_per_window=10,
            window_duration="PT1H",
            current_window_count=10,
            window_start="2026-03-01T11:30:00Z",
        )
        result = c.evaluate(_ctx())
        assert result.allowed is False
        assert result.reason == "rate_limited"
        assert result.retry_after is not None

    def test_window_expired_resets_count(self):
        # Window started 2 hours ago, duration is 1 hour -> window expired
        c = PeriodicRefreshConstraint(
            max_per_window=3,
            window_duration="PT1H",
            current_window_count=3,
            window_start="2026-03-01T10:00:00Z",
        )
        result = c.evaluate(_ctx())  # 12:00 UTC
        assert result.allowed is True
        assert result.metadata["remaining_in_window"] == 3

    def test_retry_after_points_to_window_end(self):
        c = PeriodicRefreshConstraint(
            max_per_window=5,
            window_duration="PT2H",
            current_window_count=5,
            window_start="2026-03-01T11:00:00Z",
        )
        result = c.evaluate(_ctx())  # 12:00 UTC
        assert result.retry_after is not None
        # Window end = 11:00 + 2H = 13:00 UTC
        expected = datetime(2026, 3, 1, 13, 0, tzinfo=UTC)
        assert result.retry_after == expected

    def test_no_window_start(self):
        c = PeriodicRefreshConstraint(
            max_per_window=5,
            window_duration="PT1H",
            current_window_count=3,
        )
        result = c.evaluate(_ctx())
        assert result.allowed is True
        assert result.metadata["remaining_in_window"] == 2

    def test_no_window_start_exhausted(self):
        c = PeriodicRefreshConstraint(
            max_per_window=5,
            window_duration="PT1H",
            current_window_count=5,
        )
        result = c.evaluate(_ctx())
        assert result.allowed is False


class TestPeriodicSerialization:
    def test_to_dict(self):
        c = PeriodicRefreshConstraint(
            max_per_window=10,
            window_duration="PT5H",
            current_window_count=3,
            window_start="2026-03-01T10:00:00Z",
        )
        d = c.to_dict()
        assert d["type"] == "periodic_refresh"
        assert d["max_per_window"] == 10
        assert d["window_duration"] == "PT5H"
        assert d["window_start"] == "2026-03-01T10:00:00Z"

    def test_to_dict_no_window_start(self):
        c = PeriodicRefreshConstraint(max_per_window=5, window_duration="PT1H")
        d = c.to_dict()
        assert "window_start" not in d

    def test_round_trip(self):
        c = PeriodicRefreshConstraint(
            max_per_window=20,
            window_duration="P1D",
            current_window_count=7,
            window_start="2026-03-01T00:00:00Z",
        )
        restored = PeriodicRefreshConstraint.from_dict(c.to_dict())
        assert restored.max_per_window == 20
        assert restored.window_duration == "P1D"
        assert restored.current_window_count == 7
        assert restored.window_start == "2026-03-01T00:00:00Z"

    def test_describe(self):
        c = PeriodicRefreshConstraint(max_per_window=10, window_duration="PT5H")
        desc = c.describe()
        assert "10" in desc
        assert "PT5H" in desc
