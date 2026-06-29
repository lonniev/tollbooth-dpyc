"""Tests for tollbooth.constraints.patron_proof — PatronProofConstraint (M3.6)."""

from datetime import datetime, timezone
from unittest.mock import patch

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.patron_proof import PatronProofConstraint

_NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)


def _ctx(dpop_token="", npub="npub1patron", tool_name="expensive_tool"):
    return ConstraintContext(
        ledger=LedgerSnapshot(),
        patron=PatronIdentity(npub=npub),
        env=EnvironmentSnapshot(utc_now=_NOW, tool_name=tool_name),
        dpop_token=dpop_token,
    )


def test_missing_proof_denied():
    r = PatronProofConstraint().evaluate(_ctx(dpop_token=""))
    assert r.allowed is False
    assert r.reason == "patron_proof_required"


def test_valid_proof_allowed_and_passes_npub_tool_window():
    c = PatronProofConstraint(window_seconds=300)
    with patch("tollbooth.identity_proof.verify_proof", return_value=True) as vp:
        r = c.evaluate(_ctx(dpop_token="signed", npub="npub1x", tool_name="tool_a"))
    assert r.allowed is True
    assert r.reason == "patron_proof"
    args, kwargs = vp.call_args
    assert args[0] == "signed"
    assert args[1] == "npub1x"
    assert args[2] == "tool_a"
    assert kwargs.get("window_seconds") == 300


def test_invalid_proof_denied():
    c = PatronProofConstraint()
    with patch("tollbooth.identity_proof.verify_proof", return_value=False):
        r = c.evaluate(_ctx(dpop_token="badsig"))
    assert r.allowed is False
    assert r.reason == "patron_proof_invalid"


def test_to_dict_from_dict_roundtrip():
    d = PatronProofConstraint(window_seconds=240).to_dict()
    assert d == {"type": "patron_proof", "window_seconds": 240}
    assert PatronProofConstraint.from_dict(d).window_seconds == 240


def test_from_dict_defaults_window_to_120():
    assert PatronProofConstraint.from_dict({"type": "patron_proof"}).window_seconds == 120


def test_schema_exposes_window_param():
    s = PatronProofConstraint.schema()
    assert s.type == "patron_proof"
    assert "window_seconds" in {p.name for p in s.params}


def test_describe_mentions_window():
    assert "120" in PatronProofConstraint(window_seconds=120).describe()
