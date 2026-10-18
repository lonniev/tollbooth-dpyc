"""Tests for tollbooth.constraints.patron_proof — PatronProofConstraint.

Satisfied only when the request in flight is signed by the debited patron's own
key; a holder connection's grant does not count.
"""

from datetime import UTC, datetime
from unittest.mock import patch

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.patron_proof import PatronProofConstraint

_NOW = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)


def _ctx(npub="npub1patron"):
    return ConstraintContext(
        ledger=LedgerSnapshot(),
        patron=PatronIdentity(npub=npub),
        env=EnvironmentSnapshot(utc_now=_NOW, tool_name="expensive_tool"),
    )


def test_unsigned_request_denied():
    with patch("tollbooth.holder_auth.signed_npub", return_value=None):
        r = PatronProofConstraint().evaluate(_ctx())
    assert r.allowed is False
    assert r.reason == "patron_proof_required"


def test_request_signed_by_the_patron_allowed():
    with patch("tollbooth.holder_auth.signed_npub", return_value="npub1patron"):
        r = PatronProofConstraint().evaluate(_ctx())
    assert r.allowed is True
    assert r.reason == "patron_proof"


def test_request_signed_by_someone_else_denied():
    with patch("tollbooth.holder_auth.signed_npub", return_value="npub1other"):
        r = PatronProofConstraint().evaluate(_ctx())
    assert r.allowed is False


def test_to_dict_from_dict_roundtrip():
    d = PatronProofConstraint().to_dict()
    assert d == {"type": "patron_proof"}
    assert isinstance(PatronProofConstraint.from_dict(d), PatronProofConstraint)


def test_an_older_stored_window_is_ignored():
    assert PatronProofConstraint.from_dict({"type": "patron_proof", "window_seconds": 240}).to_dict() == {
        "type": "patron_proof"
    }


def test_schema_has_no_params():
    s = PatronProofConstraint.schema()
    assert s.type == "patron_proof"
    assert s.params == []
