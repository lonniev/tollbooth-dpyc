"""Route construction, key hygiene, and provider-failure curation."""

from __future__ import annotations

import pytest

from tollbooth.constants import ErrorCode
from tollbooth.llm_route import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL_READER,
    DEFAULT_MODEL_WRITER,
    ENV_ENDPOINT,
    ENV_MODEL_READER,
    ENV_MODEL_WRITER,
    LlmRoute,
    build_messages_request,
    clamp_timeout,
    classify_llm_failure,
    error_message,
    llm_failure_situation,
    model_for,
    resolve_route,
    web_fetch_tool,
    web_search_tool,
)

# --------------------------------------------------------------------------
# The account boundary: per model client, never per process
# --------------------------------------------------------------------------

def test_two_clients_hold_two_accounts_in_one_process():
    """The load-bearing guarantee of this module.

    Resource sharing between components is allocation, not design. If a refactor ever
    turns the route back into module state read from the environment, this fails.
    """
    compose = resolve_route(api_key="key-compose", tier="writer")
    refine = resolve_route(api_key="key-refine", tier="reader")

    assert compose.api_key == "key-compose"
    assert refine.api_key == "key-refine"
    assert compose.model != refine.model


def test_route_never_renders_its_key():
    """A route rides in job specs, log lines, and tracebacks. The dataclass default
    repr would print the operator's provider key into every one of them."""
    route = resolve_route(api_key="sk-or-v1-secretvalue", tier="reader")

    assert "secretvalue" not in repr(route)
    assert "secretvalue" not in str(route)
    assert "secretvalue" not in f"{route}"
    assert "***" in repr(route)


def test_route_requires_an_explicit_account():
    with pytest.raises(ValueError, match="api_key"):
        resolve_route(api_key="", tier="reader")


# --------------------------------------------------------------------------
# Tier resolution
# --------------------------------------------------------------------------

def test_tiers_resolve_to_fleet_defaults():
    assert model_for("writer") == DEFAULT_MODEL_WRITER
    assert model_for("reader") == DEFAULT_MODEL_READER


def test_provider_slug_passes_through_untouched():
    """So an operator can pin a model without waiting on an SDK release."""
    assert model_for("anthropic/claude-sonnet-4.6") == "anthropic/claude-sonnet-4.6"


def test_unknown_tier_raises_rather_than_guessing():
    """A silent fallback to some default model is a bill nobody chose."""
    with pytest.raises(ValueError, match="unknown LLM tier"):
        model_for("supercomputer")


def test_environment_supplies_defaults(monkeypatch):
    monkeypatch.setenv(ENV_ENDPOINT, "https://example.test/v1/messages")
    monkeypatch.setenv(ENV_MODEL_WRITER, "vendor/writer-x")
    monkeypatch.setenv(ENV_MODEL_READER, "vendor/reader-x")

    assert model_for("writer") == "vendor/writer-x"
    assert resolve_route(api_key="k", tier="reader") == LlmRoute(
        endpoint="https://example.test/v1/messages", model="vendor/reader-x", api_key="k",
    )


def test_explicit_arguments_beat_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_ENDPOINT, "https://ignored.test/v1/messages")
    route = resolve_route(
        api_key="k", tier="reader",
        endpoint="https://chosen.test/v1/messages", model="vendor/chosen",
    )
    assert route.endpoint == "https://chosen.test/v1/messages"
    assert route.model == "vendor/chosen"


def test_default_endpoint_is_a_model_router():
    assert resolve_route(api_key="k").endpoint == DEFAULT_ENDPOINT


# --------------------------------------------------------------------------
# Request envelope
# --------------------------------------------------------------------------

def test_request_envelope_is_the_long_runner_shape():
    route = resolve_route(api_key="k", tier="writer")
    req = build_messages_request(
        route, system="be brief", user="hello", max_tokens=99, timeout_seconds=45,
    )

    assert req["method"] == "POST"
    assert req["url"] == route.endpoint
    assert req["timeout"] == 45.0
    assert req["headers"]["x-api-key"] == "k"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["json"]["model"] == route.model
    assert req["json"]["max_tokens"] == 99
    assert req["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in req["json"]


def test_tools_ride_along_when_declared():
    route = resolve_route(api_key="k")
    tools = [web_search_tool(3), web_fetch_tool(3, ["example.com"])]
    req = build_messages_request(route, system="s", user="u", max_tokens=10, tools=tools)

    assert req["json"]["tools"][0]["type"] == "web_search_20260209"
    assert req["json"]["tools"][1]["allowed_domains"] == ["example.com"]


def test_web_fetch_is_unrestricted_without_allowed_domains():
    assert "allowed_domains" not in web_fetch_tool(2)


@pytest.mark.parametrize(("given", "expected"), [
    (None, 210.0), (0, 210.0), (-5, 210.0), (10, 30.0), (60, 60.0), (5000, 900.0),
])
def test_timeout_is_clamped(given, expected):
    assert clamp_timeout(given) == expected


@pytest.mark.parametrize(("given", "maximum", "expected"), [
    # The caller's ceiling governs ABOVE the built-in fallback. This is the case the
    # module used to make impossible: an operator configured a 1800s block budget and
    # got 900s anyway, with nothing anywhere saying it had been cut in half.
    (1800, 1800, 1800.0),
    (5000, 1800, 1800.0),
    # ...and BELOW it, so a caller can also be stricter than the fallback.
    (600, 300, 300.0),
    # The floor still wins over an absurdly small ceiling: a sub-30s LLM request is a
    # misconfiguration, not an instruction.
    (600, 5, 30.0),
    # A missing or meaningless ceiling falls back rather than disabling the guard.
    (5000, None, 900.0),
    (5000, 0, 900.0),
])
def test_caller_supplied_maximum_governs(given, maximum, expected):
    assert clamp_timeout(given, maximum=maximum) == expected


# --------------------------------------------------------------------------
# Provider-failure classification
# --------------------------------------------------------------------------

def test_error_message_reads_the_shape_both_providers_use():
    assert error_message({"error": {"message": "nope", "code": 400}}) == "nope"
    assert error_message({"error": "flat"}) == ""
    assert error_message(None) == ""


@pytest.mark.parametrize(("status", "message"), [
    # Anthropic reports an empty account as a 400, not a 402.
    (400, "Your credit balance is too low to access the Anthropic API"),
    (400, "please go to Plans & Billing to upgrade"),
    # OpenRouter reports it as a 402 whose wording shares no needle with Anthropic's.
    # Matching only Anthropic — as all three operators did — silently lost this case.
    (402, "Insufficient credits. Add more using https://openrouter.ai/credits"),
    (402, "This request requires more credits than are available"),
    # A bare 402 from a metered LLM provider means exactly one thing.
    (402, ""),
])
def test_an_empty_account_is_recognised_whoever_says_it(status, message):
    assert classify_llm_failure(status=status, message=message) == ErrorCode.LLM_PROVIDER_UNFUNDED


@pytest.mark.parametrize("status", [401, 403])
def test_bad_credentials(status):
    assert classify_llm_failure(status=status, message="User not found.") == ErrorCode.LLM_PROVIDER_AUTH


def test_a_retired_model_slug_is_not_an_auth_problem():
    """The signature of a marketplace renaming a model under a running deployment."""
    code = classify_llm_failure(status=400, message="x-ai/grok-9 is not a valid model ID")
    assert code == ErrorCode.LLM_MODEL_UNKNOWN


def test_rate_limiting():
    assert classify_llm_failure(status=429, message="") == ErrorCode.UPSTREAM_RATE_LIMITED


def test_message_alone_classifies_when_no_status_survives():
    """The audit path holds only an exception string from a client library."""
    assert classify_llm_failure(message="credit balance too low") == ErrorCode.LLM_PROVIDER_UNFUNDED
    assert classify_llm_failure(message="Insufficient credits") == ErrorCode.LLM_PROVIDER_UNFUNDED
    assert classify_llm_failure(message="429 rate limit exceeded") == ErrorCode.UPSTREAM_RATE_LIMITED


def test_nothing_definite_classifies_as_nothing():
    assert classify_llm_failure(status=500, message="internal error") == ""


# --------------------------------------------------------------------------
# Situations
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("status", "message", "code"), [
    (402, "Insufficient credits", ErrorCode.LLM_PROVIDER_UNFUNDED),
    (401, "User not found.", ErrorCode.LLM_PROVIDER_AUTH),
    (400, "not a valid model ID", ErrorCode.LLM_MODEL_UNKNOWN),
])
def test_operator_problems_are_never_reported_as_retryable(status, message, code):
    """A scheduler must stop hammering an endpoint that cannot recover on its own."""
    situation = llm_failure_situation(status=status, message=message)
    assert situation.error_code == code
    assert situation.transient is False


def test_a_busy_provider_is_retryable():
    assert llm_failure_situation(status=429).transient is True


def test_situation_reads_the_body_when_given_no_message():
    situation = llm_failure_situation(
        status=402, body={"error": {"message": "Insufficient credits", "code": 402}},
    )
    assert situation.error_code == ErrorCode.LLM_PROVIDER_UNFUNDED


def test_unrecognised_failures_take_the_callers_fallback():
    situation = llm_failure_situation(
        status=500, message="boom",
        fallback_code="dynamic_block_unresolved",
        fallback_message="The dynamic block couldn't be resolved right now.",
    )
    assert situation.error_code == "dynamic_block_unresolved"
    assert situation.transient is True


def test_no_situation_leaks_the_raw_upstream_body():
    """The patron gets curated copy; the raw body stays operator-side."""
    raw = "Your credit balance is too low — account acct_12345"
    assert raw not in llm_failure_situation(status=400, message=raw).message
