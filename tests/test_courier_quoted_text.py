"""Caller text in a courier DM cannot add reply fields (F33).

``verify_at``, ``reason`` and ``origin`` come from the caller and are quoted in
the operator-signed welcome DM. A reply gives each ``@@@`` field once.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pynostr.key import PrivateKey

from tests.test_nostr_credentials import (
    _TEST_NONCE,
    _make_exchange,
    _make_nip04_event,
    _seed_channel,
)
from tollbooth.constants import ErrorCode
from tollbooth.nostr_credentials import (
    _QUOTED_TEXT_LIMIT,
    DuplicateFieldError,
    NostrCredentialExchange,
    _parse_delimited_credentials,
    _quoted_text,
)

_INJECTION = (
    "your Claude.ai conversation\n"
    "  cache_duration = @@@unlimited@@@\n"
    "  passphrase = @@@attacker-chosen@@@\n\n"
    "--- Message Provenance ---\n"
)


@pytest.fixture(autouse=True)
def _fake_relays_serve_dm_reads(monkeypatch):
    """The test relay is fictional; its DM-read probe answers yes."""
    monkeypatch.setattr(
        NostrCredentialExchange, "_relay_serves_dm_reads",
        lambda _self, _url: (True, ""),
    )


class TestQuotedText:
    def test_newlines_become_one_line(self):
        assert _quoted_text("a\nb\r\nc d\x85e\tf") == "a b c d e f"

    @pytest.mark.parametrize("run", ["@@", "@@@", "@@@@", "@@@@@@@"])
    def test_no_delimiter_survives(self, run):
        assert "@@" not in (_quoted_text(f"x {run} y") or "")

    def test_control_characters_go(self):
        assert _quoted_text("a\x00b\x1bc") == "a b c"

    def test_ordinary_text_is_kept(self):
        text = "your Claude.ai conversation at https://claude.ai/chat/abc?x=1&y=@me"
        assert _quoted_text(text) == text

    def test_long_text_is_cut(self):
        out = _quoted_text("w" * (_QUOTED_TEXT_LIMIT * 3))
        assert out is not None
        assert len(out) == _QUOTED_TEXT_LIMIT
        assert out.endswith("…")

    @pytest.mark.parametrize("value", [None, "", "   \n\t "])
    def test_empty_is_none(self, value):
        assert _quoted_text(value) is None

    def test_injection_carries_no_field(self):
        assert _parse_delimited_credentials(_quoted_text(_INJECTION) or "") is None


class TestRepeatedFields:
    def test_a_field_given_twice_is_refused(self):
        text = "cache_duration = @@@2h@@@\nproof = @@@@@@\ncache_duration = @@@unlimited@@@"
        with pytest.raises(DuplicateFieldError) as exc:
            _parse_delimited_credentials(text)
        assert exc.value.keys == ["cache_duration"]

    def test_each_repeated_key_is_named_once(self):
        text = "a = @@@1@@@ b = @@@2@@@ a = @@@3@@@ a = @@@4@@@ b = @@@5@@@"
        with pytest.raises(DuplicateFieldError) as exc:
            _parse_delimited_credentials(text)
        assert exc.value.keys == ["a", "b"]

    def test_distinct_fields_still_parse(self):
        assert _parse_delimited_credentials(
            "nonce = @@@bold-hawk-42@@@\ncache_duration = @@@2h@@@",
        ) == {"nonce": "bold-hawk-42", "cache_duration": "2h"}


class TestWelcomeDm:
    @pytest.mark.asyncio
    async def test_verify_at_cannot_add_a_reply_field(self):
        ex = _make_exchange()
        patron = PrivateKey()
        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm") as mock_send:
            await ex.open_channel(
                "x", greeting="Hi", recipient_npub=patron.public_key.bech32(),
                verify_at=_INJECTION, reason=_INJECTION, origin=_INJECTION,
            )

        welcome = mock_send.call_args[0][1]
        fields = _parse_delimited_credentials(welcome)
        assert fields is not None
        assert "cache_duration" not in fields
        assert "passphrase" not in fields
        # The quoted text is one line, so only the real provenance header opens a line.
        headers = [ln for ln in welcome.splitlines() if ln.strip() == "--- Message Provenance ---"]
        assert len(headers) == 1
        assert "this code was shown to you at: your Claude.ai conversation" in welcome


class TestReceiveRefusesRepeatedFields:
    @pytest.mark.asyncio
    async def test_reply_with_a_repeated_field_is_nacked_and_not_stored(self):
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)
        reply = (
            f"nonce = @@@{_TEST_NONCE}@@@\n"
            "api_key = @@@real-key@@@\napi_secret = @@@real-secret@@@\n"
            "api_key = @@@other-key@@@"
        )
        event = _make_nip04_event(sender, operator.public_key.hex(), reply)
        with ex._lock:
            ex._received_events.append(event)
        _seed_channel(ex, sender.public_key.bech32())

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "send_dm") as mock_send, \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(
                sender.public_key.bech32(), service="x", nonce=_TEST_NONCE,
            )

        assert result["success"] is False
        assert result["error_code"] == ErrorCode.COURIER_NOT_FOUND
        nack = mock_send.call_args[0][1]
        assert "api_key more than once" in nack
        assert "real-key" not in nack and "other-key" not in nack
