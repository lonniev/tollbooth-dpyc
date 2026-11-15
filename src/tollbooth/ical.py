"""iCalendar output — RFC 5545, written by hand and on purpose.

Calendar clients are unforgiving. A missing CRLF, an unfolded long line, an
unescaped comma in a summary: any of them turns a working subscription into a
silent parse failure, and the client rarely says why. So this module is small,
strict, and tested against the parts of the spec that actually bite.

Three shapes:

* **VEVENT, all-day** — something that falls on a day, not at a time.
  All-day means DTSTART;VALUE=DATE with DTEND the following day, the half-open
  convention every client expects and the one most often wrong by a day.
* **VEVENT, timed** — something that takes a slot on a day, in floating local time.
* **VTODO** — a task. Apple Reminders, Google Tasks and Thunderbird read these
  from a subscribed calendar, which is what makes a to-do arrive as a reminder
  rather than as another appointment.

An operator supplies its own PRODID and UID domain; nothing here names one.

No dependency: `icalendar` would be a wheel to carry, a version to pin and a
transitive tree to audit, for perhaps eighty lines of string building.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

# RFC 5545 §3.1: lines are folded at 75 OCTETS, and continuation lines begin
# with a single space. Octets, not characters — a multi-byte glyph split down
# the middle produces mojibake in some clients and a parse error in others.
FOLD_AT = 73


def escape(text: str) -> str:
    """Escape a value per RFC 5545 §3.3.11.

    Backslash first, or the escapes introduced afterwards get escaped again.
    """
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace(";", r"\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
    )


def fold(line: str) -> list[str]:
    """Fold one content line to the octet limit, splitting on rune boundaries."""
    raw = line.encode("utf-8")
    if len(raw) <= FOLD_AT:
        return [line]

    out: list[str] = []
    buf = ""
    size = 0
    limit = FOLD_AT
    for ch in line:
        w = len(ch.encode("utf-8"))
        if size + w > limit:
            out.append(buf)
            buf = ch
            size = w
            limit = FOLD_AT - 1     # continuation lines carry a leading space
        else:
            buf += ch
            size += w
    if buf:
        out.append(buf)
    return [out[0]] + [" " + p for p in out[1:]]


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _day(d: date) -> str:
    return d.strftime("%Y%m%d")


def vevent(
    uid: str,
    summary: str,
    day: date,
    *,
    description: str = "",
    stamp: datetime,
    categories: str | None = None,
    url: str | None = None,
) -> list[str]:
    """One all-day event.

    DTEND is the day AFTER — RFC 5545 all-day ranges are half-open, and
    omitting the increment renders a zero-length event that some clients hide
    entirely while others show on the wrong day.
    """
    lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{_stamp(stamp)}",
        f"DTSTART;VALUE=DATE:{_day(day)}",
        f"DTEND;VALUE=DATE:{_day(day + timedelta(days=1))}",
        f"SUMMARY:{escape(summary)}",
        "TRANSP:TRANSPARENT",   # an all-day note does not make anyone busy
    ]
    if description:
        lines.append(f"DESCRIPTION:{escape(description)}")
    if categories:
        lines.append(f"CATEGORIES:{escape(categories)}")
    if url:
        lines.append(f"URL:{url}")
    lines.append("END:VEVENT")
    return lines


def vtodo(
    uid: str,
    summary: str,
    due: date | None,
    *,
    description: str = "",
    stamp: datetime,
    completed: bool = False,
    categories: str | None = None,
) -> list[str]:
    """One task, for clients that surface VTODO as a reminder."""
    lines = [
        "BEGIN:VTODO",
        f"UID:{uid}",
        f"DTSTAMP:{_stamp(stamp)}",
        f"SUMMARY:{escape(summary)}",
    ]
    if due:
        lines.append(f"DUE;VALUE=DATE:{_day(due)}")
    if description:
        lines.append(f"DESCRIPTION:{escape(description)}")
    if categories:
        lines.append(f"CATEGORIES:{escape(categories)}")
    lines.append(f"STATUS:{'COMPLETED' if completed else 'NEEDS-ACTION'}")
    if completed:
        lines.append(f"COMPLETED:{_stamp(stamp)}")
        lines.append("PERCENT-COMPLETE:100")
    lines.append("END:VTODO")
    return lines


def calendar(
    name: str,
    entries: list[list[str]],
    *,
    prodid: str,
    description: str = "",
    refresh_hours: int = 12,
) -> str:
    """Wrap components into a complete calendar.

    ``prodid`` names the product that made it, like ``-//Good Earth//Farm climate//EN``.

    X-WR-CALNAME and the Apple/Google refresh hints are non-standard but are
    what actually make a subscription show the right name and poll at a sane
    rate. Without them clients invent both.
    """
    head = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{prodid}",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{escape(name)}",
        f"REFRESH-INTERVAL;VALUE=DURATION:PT{refresh_hours}H",
        f"X-PUBLISHED-TTL:PT{refresh_hours}H",
    ]
    if description:
        head.append(f"X-WR-CALDESC:{escape(description)}")

    body: list[str] = []
    for e in entries:
        body.extend(e)

    folded: list[str] = []
    for line in [*head, *body, "END:VCALENDAR"]:
        folded.extend(fold(line))
    # RFC 5545 §3.1: CRLF, and a trailing one.
    return "\r\n".join(folded) + "\r\n"


def uid_for(token: str, kind: str, key: str, *, domain: str) -> str:
    """A stable UID.

    Stability is the whole game for a subscription: a client matches on UID,
    so a UID that changes between refreshes duplicates every event instead of
    updating it. Derived from the feed token and the event's own identity, so
    the same event keeps its UID across recomputations. ``domain`` is the
    site that serves the feed.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in f"{kind}-{key}")[:120]
    return f"{safe}.{token}@{domain}"


def as_date(v: Any) -> date | None:
    """Coerce an ISO date, tolerating a datetime or a None."""
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str) and v:
        try:
            return date.fromisoformat(v[:10])
        except ValueError:
            return None
    return None


def timed_event(
    uid: str,
    summary: str,
    day: date,
    start: str,
    end: str,
    *,
    description: str = "",
    stamp: datetime,
    categories: str | None = None,
) -> list[str]:
    """One event that takes a slot on a day, rather than a whole day.

    It shows up as a block in the day rather than as a banner across it.

    Written as LOCAL time with no Z and no TZID: the hours a person types are
    the hours where they are, and stamping them UTC would move every task
    by the offset. A floating time is what RFC 5545 has for exactly this.
    """
    d = _day(day)
    return [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{_stamp(stamp)}",
        f"DTSTART:{d}T{_clock(start)}",
        f"DTEND:{d}T{_clock(end)}",
        f"SUMMARY:{escape(summary)}",
        *([f"DESCRIPTION:{escape(description)}"] if description else []),
        *([f"CATEGORIES:{escape(categories)}"] if categories else []),
        "END:VEVENT",
    ]


def _clock(hhmm: str) -> str:
    """"09:00" as iCalendar's HHMMSS. Anything unparseable becomes midnight."""
    parts = (hhmm or "").split(":")
    try:
        h = max(0, min(23, int(parts[0])))
        m = max(0, min(59, int(parts[1]))) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return "000000"
    return f"{h:02d}{m:02d}00"
