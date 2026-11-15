"""iCalendar output — the parts of RFC 5545 that actually break clients.

A calendar that fails to parse fails silently: the client shows an empty
subscription and says nothing about why. So these check the mechanics rather
than the prose.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tollbooth import ical

PRODID = "-//Example//Feeds//EN"

STAMP = datetime(2026, 8, 31, 12, 0, 0, tzinfo=UTC)


# ── Escaping ─────────────────────────────────────────────────────────────


def test_commas_and_semicolons_are_escaped():
    """Unescaped, these split one property into several and corrupt the event."""
    assert ical.escape("Dahlia; Cafe, first bloom") == r"Dahlia\; Cafe\, first bloom"


def test_backslashes_are_escaped_first():
    """Escaping them last would re-escape the escapes just introduced."""
    assert ical.escape("a\\b,c") == "a\\\\b\\,c"


def test_newlines_become_literal_n():
    assert ical.escape("one\ntwo") == "one\\ntwo"
    assert ical.escape("one\r\ntwo") == "one\\ntwo"


# ── Folding ──────────────────────────────────────────────────────────────


def test_short_lines_are_not_folded():
    assert ical.fold("SUMMARY:short") == ["SUMMARY:short"]


def test_long_lines_fold_with_a_leading_space():
    got = ical.fold("SUMMARY:" + "x" * 200)
    assert len(got) > 1
    assert all(p.startswith(" ") for p in got[1:])


def test_folding_counts_OCTETS_not_characters():
    """A multi-byte glyph split down the middle is mojibake or a parse error."""
    line = "SUMMARY:" + "é" * 80          # 2 bytes each
    for part in ical.fold(line):
        assert len(part.encode("utf-8")) <= 75


def test_folding_never_splits_a_rune():
    for part in ical.fold("SUMMARY:" + "🐝" * 40):
        part.encode("utf-8").decode("utf-8")   # would raise on a split rune


# ── All-day events ───────────────────────────────────────────────────────


def test_an_all_day_event_uses_VALUE_DATE():
    out = "\n".join(ical.vevent("u@x", "Bloom", date(2026, 7, 31), stamp=STAMP))
    assert "DTSTART;VALUE=DATE:20260731" in out


def test_DTEND_is_the_NEXT_day():
    """RFC 5545 all-day ranges are half-open. Omitting the increment renders a
    zero-length event that some clients hide and others show a day early."""
    out = "\n".join(ical.vevent("u@x", "Bloom", date(2026, 7, 31), stamp=STAMP))
    assert "DTEND;VALUE=DATE:20260801" in out


def test_an_event_is_transparent_so_it_does_not_book_the_day():
    out = "\n".join(ical.vevent("u@x", "Bloom", date(2026, 7, 31), stamp=STAMP))
    assert "TRANSP:TRANSPARENT" in out


def test_event_summaries_are_escaped():
    out = "\n".join(ical.vevent("u@x", "Dahlia, Cafe au Lait", date(2026, 7, 31), stamp=STAMP))
    assert "SUMMARY:Dahlia\\, Cafe au Lait" in out


# ── Tasks ────────────────────────────────────────────────────────────────


def test_a_todo_is_a_VTODO_not_an_event():
    out = "\n".join(ical.vtodo("u@x", "Cover the beds", date(2026, 9, 28), stamp=STAMP))
    assert out.startswith("BEGIN:VTODO") and "BEGIN:VEVENT" not in out


def test_an_open_todo_needs_action():
    out = "\n".join(ical.vtodo("u@x", "Scout", None, stamp=STAMP))
    assert "STATUS:NEEDS-ACTION" in out
    assert "DUE" not in out


def test_a_done_todo_is_completed_and_full():
    out = "\n".join(ical.vtodo("u@x", "Scout", date(2026, 9, 1), stamp=STAMP, completed=True))
    assert "STATUS:COMPLETED" in out and "PERCENT-COMPLETE:100" in out


def test_a_todo_no_longer_carries_a_priority():
    """PRIORITY went with the field. RFC 5545's 1-9 scale was never something
    a person set meaningfully, and the checkbox that fed it is gone."""
    assert "PRIORITY" not in "\n".join(ical.vtodo("u@x", "T", None, stamp=STAMP))


# ── A task that takes a slot ─────────────────────────────────────────────


def test_a_timed_event_carries_both_ends_on_the_due_day():
    out = "\n".join(ical.timed_event(
        "u@x", "Chip branches", date(2026, 9, 5), "09:00", "11:30", stamp=STAMP))
    assert "DTSTART:20260905T090000" in out
    assert "DTEND:20260905T113000" in out


def test_a_timed_event_is_floating_local_time():
    """No Z and no TZID on purpose.

    The hours a person types are the hours where they are. Stamping them
    UTC would slide every task by the offset — five hours in Vermont, which
    is enough to move a morning job into the previous evening.
    """
    out = "\n".join(ical.timed_event(
        "u@x", "T", date(2026, 9, 5), "09:00", "10:00", stamp=STAMP))
    assert "DTSTART:20260905T090000" in out and "DTSTART;TZID" not in out
    assert not any(l.startswith("DTSTART") and l.endswith("Z") for l in out.split("\n"))


@pytest.mark.parametrize(("given", "want"), [
    ("09:00", "090000"), ("9:5", "090500"), ("23:59", "235900"),
    ("25:00", "230000"), ("", "000000"), ("nonsense", "000000"),
])
def test_clock_parsing_never_produces_an_invalid_time(given, want):
    """A malformed time must not emit an unparseable calendar.

    One bad DTSTART can make a client reject the WHOLE feed, so the fallback
    is midnight rather than an exception or a blank.
    """
    out = "\n".join(ical.timed_event("u@x", "T", date(2026, 9, 5), given, "12:00", stamp=STAMP))
    assert f"DTSTART:20260905T{want}" in out


# ── The calendar wrapper ─────────────────────────────────────────────────


def test_a_calendar_is_wellformed_and_CRLF_terminated():
    ics = ical.calendar("Chores", [ical.vevent("u@x", "B", date(2026, 7, 31), stamp=STAMP)], prodid=PRODID)
    assert ics.startswith("BEGIN:VCALENDAR\r\n")
    assert ics.endswith("END:VCALENDAR\r\n")
    assert "\r\n" in ics and "\n\n" not in ics


def test_every_line_ends_CRLF_not_LF():
    ics = ical.calendar("X", [ical.vtodo("u@x", "T", None, stamp=STAMP)], prodid=PRODID)
    for line in ics.split("\r\n")[:-1]:
        assert "\n" not in line


def test_the_calendar_carries_a_name_clients_will_show():
    ics = ical.calendar("Chores — Sam", [], prodid=PRODID)
    assert "X-WR-CALNAME:Chores — Sam" in ics


def test_a_refresh_hint_is_published():
    ics = ical.calendar("X", [], refresh_hours=6, prodid=PRODID)
    assert "REFRESH-INTERVAL;VALUE=DURATION:PT6H" in ics
    assert "X-PUBLISHED-TTL:PT6H" in ics


def test_an_empty_calendar_is_still_valid():
    ics = ical.calendar("X", [], prodid=PRODID)
    assert "BEGIN:VCALENDAR" in ics and "END:VCALENDAR" in ics


# ── UIDs ─────────────────────────────────────────────────────────────────


def test_a_uid_is_stable_across_recomputations():
    """A UID that changes duplicates every event instead of updating it — the
    single most common way a subscribed calendar goes wrong."""
    a = ical.uid_for("tok", "crop", "Dahlia-2026-05-24", domain="feeds.example")
    b = ical.uid_for("tok", "crop", "Dahlia-2026-05-24", domain="feeds.example")
    assert a == b


def test_uids_differ_between_events_and_between_feeds():
    assert ical.uid_for("t", "crop", "A", domain="feeds.example") != ical.uid_for("t", "crop", "B", domain="feeds.example")
    assert ical.uid_for("t1", "crop", "A", domain="feeds.example") != ical.uid_for("t2", "crop", "A", domain="feeds.example")


def test_a_uid_is_safe_for_the_wire():
    uid = ical.uid_for("tok", "pest", "Cabbage maggot · second flight, 50%", domain="feeds.example")
    assert " " not in uid and "," not in uid and ";" not in uid


# ── Date coercion ────────────────────────────────────────────────────────


@pytest.mark.parametrize(("given", "want"), [
    ("2026-07-31", date(2026, 7, 31)),
    ("2026-07-31T12:00:00", date(2026, 7, 31)),
    (date(2026, 7, 31), date(2026, 7, 31)),
    (datetime(2026, 7, 31, 9, tzinfo=UTC), date(2026, 7, 31)),
])
def test_dates_are_coerced(given, want):
    assert ical.as_date(given) == want


@pytest.mark.parametrize("bad", [None, "", "not a date", 42])
def test_bad_dates_coerce_to_none_rather_than_today(bad):
    assert ical.as_date(bad) is None


def test_the_calendar_names_the_product_that_made_it():
    assert "PRODID:-//Example//Feeds//EN\r\n" in ical.calendar("X", [], prodid=PRODID)


def test_a_uid_ends_in_the_serving_domain():
    assert ical.uid_for("tok", "chore", "abc", domain="chipin.example").endswith(".tok@chipin.example")
