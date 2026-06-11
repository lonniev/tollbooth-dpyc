"""Tests for tollbooth.authority.onboarding (audit M2.4).

The Authority onboarding state machine tracks a single claim → approval flow
with TTL expiry. Was ~47% covered.
"""

from __future__ import annotations

import time

import pytest

from tollbooth.authority.onboarding import (
    ONBOARDING_TEMPLATES,
    OnboardingChallenge,
    OnboardingState,
)


def test_templates_present():
    assert set(ONBOARDING_TEMPLATES) == {"authority_claim", "authority_approval"}
    assert ONBOARDING_TEMPLATES["authority_claim"].fields["claim"].required


def test_challenge_expired_property():
    fresh = OnboardingChallenge("cand", "claim", ttl_seconds=600)
    assert fresh.expired is False
    old = OnboardingChallenge("cand", "claim", created_at=time.time() - 700, ttl_seconds=600)
    assert old.expired is True


def test_start_claim_then_already_active_raises():
    state = OnboardingState()
    ch = state.start_claim("npub_cand")
    assert ch.phase == "claim" and ch.candidate_npub == "npub_cand"
    assert state.get() is ch
    with pytest.raises(ValueError, match="already in progress"):
        state.start_claim("npub_other")


def test_promote_to_approval_sets_phase_and_parent():
    state = OnboardingState()
    state.start_claim("npub_cand")
    before = state.get().created_at
    promoted = state.promote_to_approval("npub_parent")
    assert promoted.phase == "approval"
    assert promoted.parent_npub == "npub_parent"
    assert promoted.created_at >= before  # TTL window refreshed


def test_promote_without_active_raises():
    with pytest.raises(ValueError, match="No active onboarding"):
        OnboardingState().promote_to_approval("npub_parent")


def test_promote_wrong_phase_raises():
    state = OnboardingState()
    state.start_claim("npub_cand")
    state.promote_to_approval("npub_parent")  # now in approval
    with pytest.raises(ValueError, match="expected 'claim'"):
        state.promote_to_approval("npub_parent")  # can't promote approval again


def test_complete_clears_state():
    state = OnboardingState()
    state.start_claim("npub_cand")
    state.complete()
    assert state.get() is None
    # a fresh onboarding can start after completion
    assert state.start_claim("npub_next").candidate_npub == "npub_next"


def test_expired_active_is_pruned_and_allows_new_claim():
    state = OnboardingState(ttl_seconds=600)
    ch = state.start_claim("npub_cand")
    ch.created_at = time.time() - 700  # force expiry
    assert state.get() is None  # pruned on access
    # expired claim no longer blocks a new one
    assert state.start_claim("npub_fresh").candidate_npub == "npub_fresh"
