"""A courier request carries its form as JSON Schema in a private rumor tag.

The form is built from the operator's CredentialTemplate. A form client
replies with a JSON object; a plain-text client replies with @@@ lines. The
golden fixtures under ``tests/fixtures/courier_form/`` are the contract a
client such as Pricing Studio decodes, so a change to the schema shape breaks
a test on both sides.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth import nip44
from tollbooth.credential_templates import (
    COURIER_FORM_DIALECT,
    CredentialTemplate,
    FieldSpec,
    courier_form_schema,
)
from tollbooth.nostr_credentials import (
    DuplicateFieldError,
    NostrCredentialExchange,
    _parse_reply,
    _rumor_id,
)

FIXTURES = Path(__file__).parent / "fixtures" / "courier_form"
NONCE = "bold-hawk-42"
RELAY = "wss://relay.example"

CREDENTIALS = CredentialTemplate(
    service="schwab-operator",
    version=1,
    fields={
        "app_key": FieldSpec(
            required=True, sensitive=True, title="App key",
            description="From developer.schwab.com, your app's page",
        ),
        "secret": FieldSpec(required=True, sensitive=True, title="Secret"),
        "account_label": FieldSpec(required=False, sensitive=False),
    },
    description="Schwab operator credentials",
)

APPROVAL = CredentialTemplate(
    service="npub_ownership",
    version=1,
    fields={
        "confirm": FieldSpec(
            required=True, sensitive=False,
            description="Any text — your signed DM is the proof",
        ),
        "cache_duration": FieldSpec(
            required=False, sensitive=False, default="2h", title="Remember for",
            description="How long to cache this proof (e.g. 2h, 1d, 3 days, unlimited)",
            examples=("2h", "1d", "1w", "30d"),
        ),
    },
    description="Npub ownership — the signed DM itself proves you own this npub",
    kind="approval",
)


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("template", "fixture"),
    [(CREDENTIALS, "credentials.json"), (APPROVAL, "approval.json")],
)
def test_form_matches_golden_fixture(template, fixture):
    form = courier_form_schema(template, nonce=NONCE, rendezvous_relay=RELAY)
    assert form == json.loads((FIXTURES / fixture).read_text())


def test_sensitive_field_is_write_only_password():
    form = courier_form_schema(CREDENTIALS, nonce=NONCE, rendezvous_relay=RELAY)
    app_key = form["properties"]["app_key"]
    assert app_key["writeOnly"] is True
    assert app_key["format"] == "password"
    assert "writeOnly" not in form["properties"]["account_label"]


def test_control_values_are_read_only_constants_and_required():
    form = courier_form_schema(CREDENTIALS, nonce=NONCE, rendezvous_relay=RELAY)
    assert form["properties"]["nonce"] == {"type": "string", "const": NONCE, "readOnly": True}
    assert form["properties"]["rendezvous_relay"]["const"] == RELAY
    assert form["required"] == ["app_key", "secret", "nonce", "rendezvous_relay"]


def test_form_names_dialect_delimiter_and_kind():
    form = courier_form_schema(APPROVAL, nonce=NONCE, rendezvous_relay=RELAY)
    assert form["$schema"] == COURIER_FORM_DIALECT
    assert form["x-delimiter"] == "@@@"
    assert form["x-courier-kind"] == "approval"
    assert form["properties"]["cache_duration"]["examples"] == ["2h", "1d", "1w", "30d"]


def test_title_falls_back_to_field_name():
    form = courier_form_schema(CREDENTIALS, nonce=NONCE, rendezvous_relay=RELAY)
    assert form["properties"]["account_label"]["title"] == "account_label"


def test_unknown_kind_is_refused():
    bad = CredentialTemplate(service="s", version=1, fields={}, kind="survey")
    with pytest.raises(ValueError, match="unknown courier kind"):
        courier_form_schema(bad, nonce=NONCE, rendezvous_relay=RELAY)


# ---------------------------------------------------------------------------
# The reply
# ---------------------------------------------------------------------------


def test_json_reply_keeps_newlines_and_quotes():
    pem = '-----BEGIN KEY-----\nab"c\n-----END KEY-----'
    reply = json.dumps({"app_key": pem, "nonce": NONCE})
    assert _parse_reply(reply) == {"app_key": pem, "nonce": NONCE}


def test_json_reply_with_a_repeated_key_is_refused():
    with pytest.raises(DuplicateFieldError):
        _parse_reply('{"secret": "a", "secret": "b"}')


@pytest.mark.parametrize(
    "reply", ['{"secret": 5}', '{"secret": null}', "{}", '{"a": ["b"]}'],
)
def test_json_reply_without_string_values_holds_no_fields(reply):
    assert _parse_reply(reply) is None


def test_delimited_reply_still_reads():
    reply = "secret = @@@s3cr3t@@@\nnonce = @@@bold-hawk-42@@@"
    assert _parse_reply(reply) == {"secret": "s3cr3t", "nonce": NONCE}


def test_text_that_only_looks_like_json_falls_back_to_delimited():
    assert _parse_reply("{ oops\nsecret = @@@s@@@") == {"secret": "s"}


# ---------------------------------------------------------------------------
# The request on the wire
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_relays_serve_dm_reads(monkeypatch):
    """Fictional relay hosts: answer the DM-read probe yes."""
    monkeypatch.setattr(
        NostrCredentialExchange, "_relay_serves_dm_reads",
        lambda _self, _url: (True, ""),
    )


@pytest.mark.asyncio
async def test_open_channel_sends_the_form_in_a_rumor_tag():
    op, patron = PrivateKey(), PrivateKey()
    ex = NostrCredentialExchange(
        nsec=op.nsec, relays=[RELAY], templates={"schwab-operator": CREDENTIALS},
    )
    with patch.object(ex, "_start_subscription"), \
         patch.object(ex, "send_dm") as mock_send:
        result = await ex.open_channel(
            "schwab-operator", greeting="Hi", recipient_npub=patron.public_key.bech32(),
        )

    tags = mock_send.call_args.kwargs["rumor_tags"]
    forms = [json.loads(t[1]) for t in tags if t[0] == "form"]
    assert len(forms) == 1
    assert forms[0]["properties"]["nonce"]["const"] == result["nonce"]
    assert forms[0]["properties"]["rendezvous_relay"]["const"] == RELAY
    # The text fallback still names the delimiter-wrapped fields.
    assert "app_key = @@@" in mock_send.call_args.args[1]


def test_gift_wrapped_rumor_carries_its_tags_and_a_nip01_id():
    op, patron = PrivateKey(), PrivateKey()
    ex = NostrCredentialExchange(nsec=op.nsec, relays=[RELAY], templates={})
    message = ex._build_gift_wrap(
        patron.public_key.hex(), "hello", [["form", "{}"]],
    )
    wrap = json.loads(message)[1]
    seal = json.loads(nip44.decrypt(wrap["content"], patron.hex(), wrap["pubkey"]))
    rumor = json.loads(nip44.decrypt(seal["content"], patron.hex(), seal["pubkey"]))
    assert rumor["tags"] == [["p", patron.public_key.hex()], ["form", "{}"]]
    reference = Event(
        kind=rumor["kind"], content=rumor["content"], tags=rumor["tags"],
        pubkey=rumor["pubkey"], created_at=rumor["created_at"],
    )
    reference.compute_id()
    assert rumor["id"] == reference.id == _rumor_id(rumor)
