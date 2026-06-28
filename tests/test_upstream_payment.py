"""Tests for tollbooth.upstream_payment — bare HTTP 402 subscription situations.

These cover the generic handler that turns an upstream "renew your
subscription" 402 into a structured, human-facing situation — distinct from a
machine-payable x402 protocol challenge (handled by X402Client). No optional
deps required.
"""

from __future__ import annotations

from typing import Any

from tollbooth.constants import ErrorCode
from tollbooth.upstream_payment import (
    classify_upstream_payment,
    is_x402_payment_challenge,
    upstream_payment_situation,
)


class _FakeResponse:
    """Duck-typed stand-in for httpx.Response (status_code, headers, json())."""

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
        body: Any = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


# ---------------------------------------------------------------------------
# is_x402_payment_challenge
# ---------------------------------------------------------------------------


def test_x402_challenge_detected_case_insensitive_dict() -> None:
    assert is_x402_payment_challenge({"Payment-Required": "eyJ4..."}) is True
    assert is_x402_payment_challenge({"payment-required": "eyJ4..."}) is True


def test_x402_challenge_absent() -> None:
    assert is_x402_payment_challenge({"content-type": "application/json"}) is False
    assert is_x402_payment_challenge({}) is False
    assert is_x402_payment_challenge(None) is False


def test_x402_challenge_from_pair_iterable() -> None:
    assert is_x402_payment_challenge([("PAYMENT-REQUIRED", "x")]) is True
    assert is_x402_payment_challenge([("x-other", "y")]) is False


# ---------------------------------------------------------------------------
# upstream_payment_situation
# ---------------------------------------------------------------------------


def test_situation_shape_and_code() -> None:
    s = upstream_payment_situation(service="X (Twitter) API")
    assert s["success"] is False
    assert s["error_code"] == ErrorCode.UPSTREAM_SUBSCRIPTION_REQUIRED
    assert s["status_code"] == 402
    assert s["transient"] is False
    assert s["service"] == "X (Twitter) API"
    assert isinstance(s["next_steps"], list) and s["next_steps"]
    # No renewal URL or detail supplied → keys omitted, not None.
    assert "renew_url" not in s
    assert "detail" not in s


def test_situation_weaves_renew_url() -> None:
    url = "https://developer.x.com/en/portal/dashboard"
    s = upstream_payment_situation(service="X (Twitter) API", renew_url=url)
    assert s["renew_url"] == url
    assert url in s["error"]
    assert any(url in step for step in s["next_steps"])


def test_situation_audience_phrasing() -> None:
    patron = upstream_payment_situation(service="X API", audience="patron")
    operator = upstream_payment_situation(service="X API", audience="operator")
    assert "your subscription" in patron["error"]
    assert "the operator's subscription" in operator["error"]


def test_situation_carries_detail() -> None:
    s = upstream_payment_situation(service="X API", detail="plan lapsed")
    assert s["detail"] == "plan lapsed"


# ---------------------------------------------------------------------------
# classify_upstream_payment
# ---------------------------------------------------------------------------


def test_classify_non_402_returns_none() -> None:
    assert classify_upstream_payment(_FakeResponse(201), service="X API") is None
    assert classify_upstream_payment(_FakeResponse(429), service="X API") is None


def test_classify_bare_402_returns_situation() -> None:
    resp = _FakeResponse(402, headers={"content-type": "application/json"})
    s = classify_upstream_payment(
        resp, service="X (Twitter) API", audience="patron",
    )
    assert s is not None
    assert s["error_code"] == ErrorCode.UPSTREAM_SUBSCRIPTION_REQUIRED
    assert s["transient"] is False


def test_classify_x402_challenge_returns_none() -> None:
    # A 402 advertising on-chain payment terms is machine-payable — not a
    # human subscription situation. Route it to X402Client, not here.
    resp = _FakeResponse(402, headers={"payment-required": "eyJ4NDAyVmVyc2lvbiI6Mn0="})
    assert classify_upstream_payment(resp, service="X API") is None


def test_classify_extracts_detail_from_json_body() -> None:
    resp = _FakeResponse(402, headers={}, body={"detail": "Usage cap reached"})
    s = classify_upstream_payment(resp, service="X API")
    assert s is not None
    assert s["detail"] == "Usage cap reached"


def test_classify_survives_non_json_body() -> None:
    resp = _FakeResponse(402, headers={})  # json() raises
    s = classify_upstream_payment(resp, service="X API")
    assert s is not None
    assert "detail" not in s


def test_exported_from_package_root() -> None:
    import tollbooth

    assert tollbooth.classify_upstream_payment is classify_upstream_payment
    assert tollbooth.upstream_payment_situation is upstream_payment_situation
    assert tollbooth.is_x402_payment_challenge is is_x402_payment_challenge
