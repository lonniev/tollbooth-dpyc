"""The instructions tell a raw-HTTP agent how to connect, in one read."""

from __future__ import annotations

import pytest
from fastmcp import FastMCP

from tollbooth.runtime import OperatorRuntime, register_standard_tools


def _instructions(monkeypatch, **rt_kwargs) -> str:
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    mcp = FastMCP("connecting-test", instructions="Good Earth.")
    rt = OperatorRuntime(nsec_env_var="__UNUSED__", **rt_kwargs)
    register_standard_tools(mcp, "goodearth", rt)
    return mcp.instructions or ""


def test_a_holder_service_says_how_to_connect(monkeypatch):
    text = _instructions(monkeypatch, holder_auth_url="https://op.example")
    assert "## Connecting" in text
    # Every path an agent had to find by trial.
    for fact in (
        "Bearer anonymous",
        "Authorization: Nostr",
        "Bearer nostr.",
        "/register",
        "PKCE S256",
        "Accept: application/json",
        "/token",
    ):
        assert fact in text, fact
    # The tools by the names an agent calls, prefix included.
    assert "goodearth_request_npub_proof(patron_npub)" in text
    assert "goodearth_receive_npub_proof(patron_npub, nonce)" in text
    assert "goodearth_held_identities" in text
    # After the operator's own words, before the troubleshooting.
    assert text.index("Good Earth.") < text.index("## Connecting") < text.index("## Troubleshooting")


def test_a_service_without_holders_says_nothing_it_does_not_do(monkeypatch):
    text = _instructions(monkeypatch)
    assert "## Connecting" not in text
    assert "## Troubleshooting" in text


@pytest.mark.parametrize("stale", ["~1 hour"])
def test_the_proof_lifetime_is_the_codes(monkeypatch, stale):
    text = _instructions(monkeypatch)
    assert stale not in text
    assert "2 hours by default, at most 30 days" in text
