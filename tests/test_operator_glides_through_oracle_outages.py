"""An operator comes up while the Oracle is down, on what it recorded earlier."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.event import Event
from pynostr.key import PrivateKey

from tollbooth import operator_state
from tollbooth.bootstrap import BootstrapClient, _spreads
from tollbooth.bootstrap_relay import send_bootstrap_config
from tollbooth.nip04 import encrypt
from tollbooth.operator_state import (
    OPERATOR_STATE_TAG,
    read_operator_state,
    record_operator_state,
)
from tollbooth.oracle_client import OracleClient, OracleClientError
from tollbooth.relay_registry import RelayRegistry, RelayRegistryError, fetch_published_relays

pytestmark = pytest.mark.records_operator_state

OP = PrivateKey()
AUTH = PrivateKey()
RELAY = "wss://relay.test"
NEON = "postgresql://from-the-authority"


class Relays:
    """A relay that stores events and honors only kinds and the d tag, as a hostile one might."""

    def __init__(self):
        self.events: list[dict] = []

    def connect(self, _url, timeout=None):
        ws = MagicMock()
        replies: list[str] = []

        def send(raw):
            msg = json.loads(raw)
            if msg[0] == "EVENT":
                self.events.append(msg[1])
                replies.append(json.dumps(["OK", msg[1]["id"], True, ""]))
            elif msg[0] == "REQ":
                wanted = msg[2]["#d"]
                for event in self.events:
                    if any(t[0] == "d" and t[1] in wanted for t in event["tags"]):
                        replies.append(json.dumps(["EVENT", msg[1], event]))
                replies.append(json.dumps(["EOSE", msg[1]]))

        ws.send.side_effect = send
        ws.recv.side_effect = lambda: replies.pop(0)
        return ws


@pytest.fixture
def relays():
    fake = Relays()
    with patch("websocket.create_connection", side_effect=fake.connect):
        yield fake


def _oracle_up(authority=None):
    oracle = MagicMock()
    oracle.get_relays = AsyncMock(return_value=[RELAY])
    oracle.resolve_authority_for = AsyncMock(
        return_value={"npub": AUTH.public_key.bech32()} if authority is None else authority or None
    )
    oracle.session.return_value.__aenter__.return_value = oracle
    oracle.session.return_value.__aexit__ = AsyncMock(return_value=False)
    return oracle


def _oracle_down():
    oracle = MagicMock()
    oracle.session.return_value.__aenter__.side_effect = OracleClientError("refused")
    oracle.session.return_value.__aexit__ = AsyncMock(return_value=False)
    return oracle


async def _bootstrap(oracle, *, published=(RELAY,)):
    fetch = MagicMock(return_value=list(published))
    with patch("tollbooth.oracle_client.default_oracle_client", return_value=oracle), \
         patch("tollbooth.relay_registry.fetch_published_relays", fetch):
        result = await BootstrapClient(nsec_hex=OP.bech32()).bootstrap(retry_backoff=(0,))
        await asyncio.gather(*_spreads)
    return result, fetch


def _authority_publishes():
    send_bootstrap_config(
        authority_nsec=AUTH.bech32(),
        operator_npub=OP.public_key.bech32(),
        config={"neon_database_url": NEON},
        relays=[RELAY],
    )


def _state_event(author: PrivateKey, body: dict) -> dict:
    """A record under the operator's d tag, encrypted to the operator."""
    event = Event(
        pubkey=author.public_key.hex(),
        kind=30078,
        content=encrypt(
            private_key_hex=author.hex(),
            public_key_hex=OP.public_key.hex(),
            plaintext=json.dumps({"type": OPERATOR_STATE_TAG, **body}),
        ),
        created_at=int(time.time()) + 100,
        tags=[["d", f"{OPERATOR_STATE_TAG}:{OP.public_key.hex()}"]],
    )
    event.sign(author.hex())
    return event.to_dict()


@pytest.mark.asyncio
async def test_the_whole_glide_oracle_up_then_down(relays):
    _authority_publishes()

    result, fetch = await _bootstrap(_oracle_up())
    assert result.success and result.neon_database_url == NEON
    fetch.assert_not_called()  # a healthy Oracle means no GitHub read

    result, fetch = await _bootstrap(_oracle_down())
    fetch.assert_called_once()
    assert result.success, result.error
    assert result.neon_database_url == NEON
    assert result.authority_npub == AUTH.public_key.bech32()


@pytest.mark.asyncio
async def test_the_record_holds_no_database_credential(relays):
    _authority_publishes()
    await _bootstrap(_oracle_up())
    state = read_operator_state(OP.bech32(), [RELAY])
    assert state is not None
    assert state.authority_npub == AUTH.public_key.bech32()
    body = operator_state._latest_body(OP.bech32(), [RELAY])
    assert "postgresql" not in json.dumps(body)


@pytest.mark.asyncio
async def test_with_no_record_a_silent_oracle_means_wait(relays):
    _authority_publishes()
    result, _ = await _bootstrap(_oracle_down())
    assert not result.success and result.transient
    assert result.neon_database_url is None


@pytest.mark.asyncio
async def test_a_strangers_record_and_config_do_not_supply_a_database(relays):
    stranger = PrivateKey()
    relays.events.append(_state_event(stranger, {
        "authority_npub": stranger.public_key.bech32(),
        "relays": [RELAY],
        "confirmed_at": int(time.time()),
    }))
    send_bootstrap_config(
        authority_nsec=stranger.bech32(),
        operator_npub=OP.public_key.bech32(),
        config={"neon_database_url": "postgresql://evil"},
        relays=[RELAY],
    )
    result, _ = await _bootstrap(_oracle_down())
    assert not result.success and result.neon_database_url is None


def test_a_record_claiming_the_operator_with_a_bad_signature_is_ignored(relays):
    forged = _state_event(OP, {
        "authority_npub": PrivateKey().public_key.bech32(),
        "relays": [RELAY],
        "confirmed_at": int(time.time()),
    })
    relays.events.append(dict(forged, sig="00" * 64))
    assert read_operator_state(OP.bech32(), [RELAY]) is None


def test_a_forged_newer_record_does_not_hide_the_real_one(relays):
    record_operator_state(OP.bech32(), AUTH.public_key.bech32(), [RELAY])
    relays.events.append(_state_event(PrivateKey(), {"withdrawn_at": 1}))
    state = read_operator_state(OP.bech32(), [RELAY])
    assert state is not None and state.authority_npub == AUTH.public_key.bech32()


def test_a_record_too_old_is_not_trusted(relays):
    old = time.time() - operator_state.MAX_GLIDE_SECONDS - 60
    with patch("tollbooth.operator_state.time.time", return_value=old):
        record_operator_state(OP.bech32(), AUTH.public_key.bech32(), [RELAY])
    assert read_operator_state(OP.bech32(), [RELAY]) is None


@pytest.mark.asyncio
async def test_the_oracle_saying_no_withdraws_the_record(relays):
    _authority_publishes()
    await _bootstrap(_oracle_up())
    await asyncio.sleep(1.1)  # created_at has one-second resolution

    result, _ = await _bootstrap(_oracle_up(authority={}))
    assert not result.success and result.transient

    result, _ = await _bootstrap(_oracle_down())
    assert not result.success, "a withdrawn record must not be glided on"


def test_an_unchanged_fresh_record_is_not_rewritten(relays):
    record_operator_state(OP.bech32(), AUTH.public_key.bech32(), [RELAY])
    count = len(relays.events)
    record_operator_state(OP.bech32(), AUTH.public_key.bech32(), [RELAY])
    assert len(relays.events) == count


@pytest.mark.asyncio
async def test_oracle_and_published_list_both_down_is_transient(relays):
    fetch = MagicMock(side_effect=RuntimeError("429"))
    with patch("tollbooth.oracle_client.default_oracle_client", return_value=_oracle_down()), \
         patch("tollbooth.relay_registry.fetch_published_relays", fetch):
        result = await BootstrapClient(nsec_hex=OP.bech32()).bootstrap(retry_backoff=(0,))
    assert not result.success and result.transient


@pytest.mark.asyncio
async def test_an_oracle_that_cannot_read_its_registry_is_silent_not_a_no():
    client = OracleClient("https://oracle.test/mcp")
    client.call_tool = AsyncMock(return_value={"success": False, "error": "Could not read the registry: 503"})
    with pytest.raises(OracleClientError):
        await client.resolve_authority_for("npub1op")
    client.call_tool = AsyncMock(return_value={"success": False, "error": "No legitimate Authority"})
    assert await client.resolve_authority_for("npub1op") is None


@pytest.mark.asyncio
async def test_silence_about_relays_alone_reads_the_list_and_keeps_the_named_authority(relays):
    _authority_publishes()
    oracle = _oracle_up()
    oracle.get_relays = AsyncMock(side_effect=RuntimeError("oracle down"))
    result, fetch = await _bootstrap(oracle)
    fetch.assert_called_once()
    assert result.success and result.neon_database_url == NEON


def _published(payload):
    response = MagicMock()
    response.json.return_value = payload
    return response


def test_the_published_list_keeps_primary_first_and_only_wss():
    payload = {"relays": [{"url": "wss://b"}, {"url": "wss://a", "primary": True}, {"url": "https://x"}, "wss://c"]}
    with patch("tollbooth.relay_registry.httpx.get", return_value=_published(payload)):
        assert fetch_published_relays() == ["wss://a", "wss://b", "wss://c"]


def test_the_sync_registry_falls_back_to_the_published_list_on_a_cold_cache():
    reg = RelayRegistry()
    with patch.object(reg, "_do_fetch", side_effect=RuntimeError("oracle down")), \
         patch("tollbooth.relay_registry.fetch_published_relays", return_value=["wss://a"]) as fetch:
        assert reg.relays() == ["wss://a"]
        assert reg.relays() == ["wss://a"]
    fetch.assert_called_once()

    with patch.object(RelayRegistry, "_do_fetch", side_effect=RuntimeError("oracle down")), \
         patch("tollbooth.relay_registry.fetch_published_relays", side_effect=RuntimeError("429")), \
         pytest.raises(RelayRegistryError):
        RelayRegistry().relays()
