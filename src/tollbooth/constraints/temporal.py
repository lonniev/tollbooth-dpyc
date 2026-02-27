"""Temporal window constraint — restrict tool access by time of day / day of week."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ToolConstraint,
)


class TemporalWindowConstraint(ToolConstraint):
    """Allow invocations only during a configurable time window.

    Parameters
    ----------
    schedule:
        ``"HH:MM-HH:MM"`` (24-hour) range.  The window wraps around midnight
        when end < start (e.g. ``"22:00-06:00"``).
    timezone:
        IANA timezone name (e.g. ``"US/Eastern"``).  Defaults to ``"UTC"``.
    days_of_week:
        Optional list of allowed weekday numbers, 0 = Monday .. 6 = Sunday.
        ``None`` means every day.
    """

    def __init__(
        self,
        schedule: str,
        timezone: str = "UTC",
        days_of_week: list[int] | None = None,
    ) -> None:
        self.schedule = schedule
        self.timezone = timezone
        self.days_of_week = days_of_week
        # pre-parse schedule
        start_s, end_s = schedule.split("-")
        sh, sm = (int(x) for x in start_s.strip().split(":"))
        eh, em = (int(x) for x in end_s.strip().split(":"))
        self._start_minutes = sh * 60 + sm
        self._end_minutes = eh * 60 + em
        self._tz = ZoneInfo(timezone)

    # ---- ToolConstraint interface ----

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        local_dt = context.env.utc_now.astimezone(self._tz)
        now_minutes = local_dt.hour * 60 + local_dt.minute

        # Check day-of-week (Monday=0)
        if self.days_of_week is not None and local_dt.weekday() not in self.days_of_week:
            return ConstraintResult(
                allowed=False,
                reason="outside_window",
                message=f"Tool not available on {local_dt.strftime('%A')}.",
                retry_after=self._next_open(local_dt),
            )

        # Check time window
        if self._in_window(now_minutes):
            return ConstraintResult(allowed=True)

        return ConstraintResult(
            allowed=False,
            reason="outside_window",
            message=f"Tool available {self.schedule} {self.timezone}.",
            retry_after=self._next_open(local_dt),
        )

    def describe(self) -> str:
        parts = [f"Available {self.schedule} {self.timezone}"]
        if self.days_of_week is not None:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            names = [day_names[d] for d in sorted(self.days_of_week)]
            parts.append(f"on {', '.join(names)}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "temporal_window",
            "schedule": self.schedule,
            "timezone": self.timezone,
        }
        if self.days_of_week is not None:
            d["days_of_week"] = self.days_of_week
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalWindowConstraint:
        return cls(
            schedule=data["schedule"],
            timezone=data.get("timezone", "UTC"),
            days_of_week=data.get("days_of_week"),
        )

    # ---- helpers ----

    def _in_window(self, now_minutes: int) -> bool:
        if self._start_minutes <= self._end_minutes:
            # Normal range, e.g. 08:00-21:00
            return self._start_minutes <= now_minutes < self._end_minutes
        else:
            # Wraps midnight, e.g. 22:00-06:00
            return now_minutes >= self._start_minutes or now_minutes < self._end_minutes

    def _next_open(self, local_dt: datetime) -> datetime:
        """Compute the next datetime when the window opens."""
        # Start by trying today's window start
        candidate = local_dt.replace(
            hour=self._start_minutes // 60,
            minute=self._start_minutes % 60,
            second=0,
            microsecond=0,
        )
        # If the window hasn't opened yet today, candidate is today
        # If it already passed, try tomorrow
        if candidate <= local_dt:
            candidate += timedelta(days=1)

        # If days_of_week is set, advance to the next valid day
        if self.days_of_week is not None:
            for _ in range(8):  # at most 7 days forward
                if candidate.weekday() in self.days_of_week:
                    break
                candidate += timedelta(days=1)

        # Convert back to UTC for the result
        return candidate.astimezone(timezone.utc)
