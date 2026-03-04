"""Periodic refresh constraint — rate-limit invocations per rolling window."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ToolConstraint,
)

# ---------------------------------------------------------------------------
# ISO-8601 duration parser (stdlib-only, no pendulum)
# ---------------------------------------------------------------------------

_ISO_DURATION_RE = re.compile(
    r"^P"
    r"(?:(\d+)Y)?"
    r"(?:(\d+)M)?"
    r"(?:(\d+)D)?"
    r"(?:T"
    r"(?:(\d+)H)?"
    r"(?:(\d+)M)?"
    r"(?:(\d+(?:\.\d+)?)S)?"
    r")?$",
    re.IGNORECASE,
)


def parse_iso_duration(s: str) -> timedelta:
    """Parse a subset of ISO-8601 durations to :class:`timedelta`.

    Supports ``P[nY][nM][nD][T[nH][nM][nS]]``.
    Year = 365 days, Month = 30 days (approximations).
    """
    m = _ISO_DURATION_RE.match(s)
    if not m:
        raise ValueError(f"Invalid ISO 8601 duration: {s!r}")

    years = int(m.group(1) or 0)
    months = int(m.group(2) or 0)
    days = int(m.group(3) or 0)
    hours = int(m.group(4) or 0)
    minutes = int(m.group(5) or 0)
    seconds = float(m.group(6) or 0)

    total_days = years * 365 + months * 30 + days
    return timedelta(
        days=total_days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
    )


# ---------------------------------------------------------------------------
# PeriodicRefreshConstraint
# ---------------------------------------------------------------------------


class PeriodicRefreshConstraint(ToolConstraint):
    """Limit invocations to *max_per_window* within each rolling window.

    Parameters
    ----------
    max_per_window:
        How many invocations are allowed per window.
    window_duration:
        ISO-8601 duration string, e.g. ``"PT5H"`` for 5 hours.
    current_window_count:
        Number of invocations already consumed in the current window.
    window_start:
        ISO-8601 datetime of current window start.  If omitted the
        constraint treats *current_window_count* as authoritative.
    """

    def __init__(
        self,
        max_per_window: int,
        window_duration: str,
        current_window_count: int = 0,
        window_start: str | None = None,
    ) -> None:
        self.max_per_window = max_per_window
        self.window_duration = window_duration
        self.current_window_count = current_window_count
        self.window_start = window_start
        self._delta = parse_iso_duration(window_duration)

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        now = context.env.utc_now

        # Determine effective window start / count
        if self.window_start is not None:
            ws = datetime.fromisoformat(self.window_start.replace("Z", "+00:00"))
            if ws.tzinfo is None:
                ws = ws.replace(tzinfo=timezone.utc)
            window_end = ws + self._delta
            if now >= window_end:
                # Window expired — new window, count resets
                count = 0
            else:
                count = self.current_window_count
        else:
            ws = now  # synthetic "current" window
            window_end = ws + self._delta
            count = self.current_window_count

        remaining = max(0, self.max_per_window - count)

        if remaining <= 0:
            return ConstraintResult(
                allowed=False,
                reason="rate_limited",
                message=(
                    f"Rate limit reached ({self.max_per_window} per "
                    f"{self.window_duration}). Try again later."
                ),
                retry_after=window_end.astimezone(timezone.utc),
                metadata={"remaining_in_window": 0},
            )

        return ConstraintResult(
            allowed=True,
            metadata={"remaining_in_window": remaining},
        )

    def describe(self) -> str:
        return (
            f"Rate-limited to {self.max_per_window} invocations "
            f"per {self.window_duration}."
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "periodic_refresh",
            "max_per_window": self.max_per_window,
            "window_duration": self.window_duration,
            "current_window_count": self.current_window_count,
        }
        if self.window_start is not None:
            d["window_start"] = self.window_start
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PeriodicRefreshConstraint:
        return cls(
            max_per_window=int(data["max_per_window"]),
            window_duration=str(data["window_duration"]),
            current_window_count=int(data.get("current_window_count", 0)),
            window_start=data.get("window_start"),
        )
