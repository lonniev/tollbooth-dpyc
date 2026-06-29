"""Tests for the structured async-job failure situation (curated, frontend-facing)."""

import json

from tollbooth.async_situation import (
    AsyncJobSituation,
    situation_response_from_row,
)


def test_to_response_carries_curated_fields():
    sit = AsyncJobSituation(
        error_code="operator_llm_unfunded",
        message="This service's AI provider is temporarily unavailable. No fare was charged.",
        next_steps="Please try again shortly.",
        transient=False,
    )
    resp = sit.to_response()
    assert resp["status"] == "error"
    assert resp["error_code"] == "operator_llm_unfunded"
    assert resp["error"] == sit.message
    assert resp["next_steps"] == "Please try again shortly."
    assert resp["transient"] is False
    assert resp["refunded"] is True


def test_row_roundtrip_preserves_situation():
    sit = AsyncJobSituation(
        error_code="upstream_rate_limited", message="Rate limited; try again.",
        next_steps="Wait a minute.", transient=True,
    )
    rebuilt = situation_response_from_row(sit.to_row())
    assert rebuilt["error_code"] == "upstream_rate_limited"
    assert rebuilt["error"] == "Rate limited; try again."
    assert rebuilt["transient"] is True
    assert rebuilt["refunded"] is True


def test_plain_string_row_is_treated_as_safe_message():
    # back-compat: a row written by the generic refund path (never a raw
    # exception) is surfaced as a plain message, not parsed as a situation.
    resp = situation_response_from_row("Job execution failed.")
    assert resp["status"] == "error"
    assert resp["error"] == "Job execution failed."
    assert resp["refunded"] is True
    assert "error_code" not in resp  # plain path carries no machine code


def test_to_row_is_json_and_tagged():
    sit = AsyncJobSituation(error_code="x", message="y")
    data = json.loads(sit.to_row())
    assert data["kind"] == "async_job_situation"
    assert data["error_code"] == "x"
