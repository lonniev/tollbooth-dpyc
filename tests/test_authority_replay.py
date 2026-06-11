"""Tests for tollbooth.authority.replay.ReplayTracker (audit M2.4).

The anti-replay tracker is defence-in-depth against certificate replay within
the TTL window. Was ~31% covered.
"""

from __future__ import annotations

import tollbooth.authority.replay as replay_mod
from tollbooth.authority.replay import ReplayTracker


def test_new_jti_accepted_replay_rejected():
    t = ReplayTracker()
    assert t.check_and_record("jti-1") is True   # first sighting accepted
    assert t.check_and_record("jti-1") is False  # replay rejected
    assert t.check_and_record("jti-2") is True   # different jti accepted
    assert t.size == 2


def test_expired_entries_pruned(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(replay_mod.time, "monotonic", lambda: clock[0])

    t = ReplayTracker(ttl_seconds=600)
    assert t.check_and_record("old") is True      # recorded at t=1000
    assert t.size == 1

    clock[0] = 1700.0  # 700s later — past the 600s TTL
    assert t.check_and_record("fresh") is True     # pruning drops "old"
    assert t.size == 1                              # only "fresh" remains

    # "old" was pruned, so it is accepted again (no longer considered a replay)
    assert t.check_and_record("old") is True


def test_unexpired_entry_still_blocks(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(replay_mod.time, "monotonic", lambda: clock[0])

    t = ReplayTracker(ttl_seconds=600)
    t.check_and_record("x")
    clock[0] = 1300.0  # 300s later — within the 600s TTL
    assert t.check_and_record("x") is False        # still a replay
