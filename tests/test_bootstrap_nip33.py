"""Bootstrap config is a NIP-33 replaceable event, and send/receive agree.

Guards the kind-4 → kind-30078 (NIP-78 parameterized-replaceable) cutover:
the Authority publishes a replaceable event scoped by a per-operator `d` tag,
and the operator reads that same event back and decrypts it with only its nsec.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from pynostr.key import PrivateKey

from tollbooth.bootstrap_relay import (
    BOOTSTRAP_CONFIG_TAG,
    receive_bootstrap_config,
    send_bootstrap_config,
)

OP_PK = PrivateKey()
OP_NSEC = OP_PK.bech32()
OP_NPUB = OP_PK.public_key.bech32()
OP_HEX = OP_PK.public_key.hex()

AUTH_PK = PrivateKey()
AUTH_NSEC = AUTH_PK.bech32()
AUTH_HEX = AUTH_PK.public_key.hex()

CONFIG = {"neon_database_url": "postgresql://x", "schema": "op_test"}


def _publish_and_capture() -> dict:
    """Run send_bootstrap_config against a mock relay; return the EVENT dict."""
    captured: dict = {}

    def _send(msg: str) -> None:
        arr = json.loads(msg)
        if arr[0] == "EVENT":
            captured["event"] = arr[1]

    ws = MagicMock()
    ws.recv.return_value = json.dumps(["OK", "id", True, ""])
    ws.send.side_effect = _send
    with patch("websocket.create_connection", return_value=ws):
        ok = send_bootstrap_config(
            authority_nsec=AUTH_NSEC, operator_npub=OP_NPUB,
            config=CONFIG, relays=["wss://relay.test"],
        )
    assert ok is True
    return captured["event"]


def test_published_event_is_replaceable_with_scoped_d_tag() -> None:
    ev = _publish_and_capture()
    assert ev["kind"] == 30078
    tags = {t[0]: t[1] for t in ev["tags"] if len(t) >= 2}
    assert tags["d"] == f"{BOOTSTRAP_CONFIG_TAG}:{OP_HEX}"
    assert tags["p"] == OP_HEX


def test_operator_reads_back_and_decrypts_with_only_its_nsec() -> None:
    ev = _publish_and_capture()

    recv_ws = MagicMock()
    recv_ws.recv.side_effect = [
        json.dumps(["EVENT", "sub", ev]),
        json.dumps(["EOSE", "sub"]),
    ]
    with patch("websocket.create_connection", return_value=recv_ws):
        config, author, diag = receive_bootstrap_config(
            operator_nsec=OP_NSEC,
            relays=["wss://relay.test"],
        )
    assert config == CONFIG, diag
    # The Authority npub is DISCOVERED from the event's author — the operator
    # supplied only its own nsec.
    assert author == AUTH_HEX


def test_receive_filter_targets_kind_30078_and_d_tag_without_authors() -> None:
    """The REQ filter selects the replaceable kind + the operator's own scoped
    `d` tag, with NO `authors` clause (the operator finds its own config without
    knowing who signed it) and no `since` window."""
    sent: list = []
    recv_ws = MagicMock()
    recv_ws.recv.side_effect = [json.dumps(["EOSE", "sub"])]
    recv_ws.send.side_effect = lambda m: sent.append(json.loads(m))
    with patch("websocket.create_connection", return_value=recv_ws):
        receive_bootstrap_config(
            operator_nsec=OP_NSEC,
            relays=["wss://relay.test"],
        )
    req = next(m for m in sent if m[0] == "REQ")
    filt = req[2]
    assert filt["kinds"] == [30078]
    assert "authors" not in filt  # no pre-known Authority required
    assert filt["#d"] == [f"{BOOTSTRAP_CONFIG_TAG}:{OP_HEX}"]
    assert "since" not in filt


def test_expected_authority_filter_rejects_a_spoofed_event() -> None:
    """With a trusted Authority known, an event bearing our `d` tag from any
    OTHER author is ignored — the spoof guard for the author-agnostic query."""
    ev = _publish_and_capture()  # legitimately authored by AUTH_PK

    recv_ws = MagicMock()
    recv_ws.recv.side_effect = [
        json.dumps(["EVENT", "sub", ev]),
        json.dumps(["EOSE", "sub"]),
    ]
    imposter_hex = PrivateKey().public_key.hex()
    with patch("websocket.create_connection", return_value=recv_ws):
        config, author, diag = receive_bootstrap_config(
            operator_nsec=OP_NSEC,
            relays=["wss://relay.test"],
            expected_authority_hex=imposter_hex,
        )
    assert config is None, diag  # real event dropped: author != expected
    assert author is None


# ---------------------------------------------------------------------------
# Relays are read together; newest revision still wins within the settle window
# ---------------------------------------------------------------------------

def _event_for(config: dict, at: int) -> dict:
    """Publish ``config`` stamped at epoch ``at``; return the EVENT dict."""
    captured: dict = {}

    def _send(msg: str) -> None:
        arr = json.loads(msg)
        if arr[0] == "EVENT":
            captured["event"] = arr[1]

    ws = MagicMock()
    ws.recv.return_value = json.dumps(["OK", "id", True, ""])
    ws.send.side_effect = _send
    with patch("websocket.create_connection", return_value=ws), patch("time.time", return_value=at):
        assert send_bootstrap_config(
            authority_nsec=AUTH_NSEC, operator_npub=OP_NPUB,
            config=config, relays=["wss://relay.test"],
        )
    return captured["event"]


def _serving(event: dict | None, *, delay: float = 0.0, refuse: bool = False):
    """A per-relay create_connection: optionally slow, optionally refusing."""
    import time as _time

    def _connect(*_a, **_k):
        if refuse:
            raise ConnectionRefusedError("refused")
        ws = MagicMock()
        frames = ([json.dumps(["EVENT", "s", event])] if event else []) + [json.dumps(["EOSE", "s"])]

        def _recv():
            if delay and not ws._slept:
                ws._slept = True
                _time.sleep(delay)
            return frames.pop(0)

        ws._slept = False
        ws.recv.side_effect = _recv
        return ws

    return _connect


def _by_relay(table: dict):
    def _connect(url, *a, **k):
        return table[url](url, *a, **k)
    return _connect


def test_slower_relay_holding_the_newer_revision_wins_within_the_settle_window() -> None:
    stale = _event_for({"neon_database_url": "postgresql://rotated-away"}, at=1_700_000_000)
    fresh = _event_for({"neon_database_url": "postgresql://current"}, at=1_700_000_100)
    table = {
        "wss://quick-but-stale": _serving(stale),
        "wss://slower-but-fresh": _serving(fresh, delay=0.3),
    }
    with patch("websocket.create_connection", side_effect=_by_relay(table)):
        config, author, diag = receive_bootstrap_config(
            operator_nsec=OP_NSEC, relays=list(table),
        )
    assert config == {"neon_database_url": "postgresql://current"}
    assert author == AUTH_HEX
    assert diag == "relays=2, events=2"


def test_refusing_relay_is_an_error_but_a_slow_one_is_only_slow(monkeypatch) -> None:
    import tollbooth.bootstrap_relay as br

    monkeypatch.setattr(br, "_READ_BUDGET_SECONDS", 0.4)
    monkeypatch.setattr(br, "_SETTLE_SECONDS", 0.1)
    fresh = _event_for(CONFIG, at=1_700_000_000)
    table = {
        "wss://serves": _serving(fresh),
        "wss://refuses": _serving(None, refuse=True),
        "wss://asleep": _serving(fresh, delay=3.0),
    }
    with patch("websocket.create_connection", side_effect=_by_relay(table)):
        config, _author, diag = receive_bootstrap_config(
            operator_nsec=OP_NSEC, relays=list(table),
        )
    assert config == CONFIG
    assert diag == "relays=3, events=1, slow=1, errors=[wss://refuses: refused]"
    assert "wss://asleep" not in diag.split("errors=[", 1)[1]
