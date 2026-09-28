"""Audit P1 / M1.2 — relay drains must not block the async event loop.

`open_channel` and the `receive` drain do synchronous websocket relay I/O
(connect + recv until EOSE, bounded by a per-relay timeout). Running that inline
froze every other coroutine on the serverless event loop for up to the timeout
per relay. Both now hop to a worker thread via ``asyncio.to_thread``.

These tests prove the loop stays responsive: a fast ticker coroutine resolves
*during* a slow relay drain, not after it. On the pre-fix inline code the ticker
could not run until the blocking drain finished, so the timing assertion fails.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.nostr_credentials import NostrCredentialExchange

_DRAIN_SECONDS = 0.25
_TICK_AFTER = 0.05
# Generous margin: a free loop lets the tick land right after _TICK_AFTER; a
# blocked loop can't service it until the drain ends (~_DRAIN_SECONDS).
_BLOCKED_THRESHOLD = 0.18


def _template() -> dict[str, CredentialTemplate]:
    return {
        "x": CredentialTemplate(
            service="x",
            version=1,
            fields={
                "api_key": FieldSpec(required=True, sensitive=True),
                "api_secret": FieldSpec(required=True, sensitive=True),
            },
            description="Test X API credentials",
        ),
    }


def _exchange() -> NostrCredentialExchange:
    return NostrCredentialExchange(
        nsec=PrivateKey().nsec,
        relays=["wss://relay.test.com"],
        templates=_template(),
    )


def _slow_connect(*_args, **_kwargs):
    """A mock websocket whose first recv() blocks, then returns EOSE."""
    ws = MagicMock()

    def _recv():
        time.sleep(_DRAIN_SECONDS)
        return json.dumps(["EOSE", "sub"])

    ws.recv.side_effect = _recv
    return ws


async def _ticker(progressed: list[float]) -> None:
    await asyncio.sleep(_TICK_AFTER)
    progressed.append(time.monotonic())


@pytest.mark.asyncio
async def test_open_channel_does_not_block_event_loop() -> None:
    ex = _exchange()
    progressed: list[float] = []

    with patch("tollbooth.nostr_credentials.create_connection", side_effect=_slow_connect):
        start = time.monotonic()
        result, _ = await asyncio.gather(
            ex.open_channel("x", greeting="hi"),
            _ticker(progressed),
        )

    assert result["success"] is True
    assert progressed, "ticker coroutine never ran"
    # The tick landed while open_channel's relay drain was still in flight.
    assert progressed[0] - start < _BLOCKED_THRESHOLD, (
        f"event loop appears blocked: tick at "
        f"{progressed[0] - start:.3f}s (drain is {_DRAIN_SECONDS}s)"
    )


@pytest.mark.asyncio
async def test_receive_drain_does_not_block_event_loop() -> None:
    ex = _exchange()
    sender = PrivateKey()
    sender_bech32 = sender.public_key.bech32()
    # Seed a resolvable pinned channel so receive() reaches the relay drain.
    ex._pending_dpop_tokens[(sender_bech32, "x")] = ("bold-hawk-42", time.time() + 600)
    ex._pinned_relays[(sender_bech32, "x")] = "wss://relay.test.com"

    progressed: list[float] = []

    with patch("tollbooth.nostr_credentials.create_connection", side_effect=_slow_connect):
        start = time.monotonic()
        result, _ = await asyncio.gather(
            ex.receive(sender_bech32, service="x", dpop_token="bold-hawk-42"),
            _ticker(progressed),
        )

    # receive may not find a matching DM (empty relay) — that's fine; we only
    # assert the loop stayed responsive during its drain.
    assert isinstance(result, dict)
    assert progressed, "ticker coroutine never ran"
    assert progressed[0] - start < _BLOCKED_THRESHOLD, (
        f"event loop appears blocked: tick at "
        f"{progressed[0] - start:.3f}s (drain is {_DRAIN_SECONDS}s)"
    )


# ---------------------------------------------------------------------------
# The two cold-start sweeps: bootstrap config read and the courier's relay probe
# ---------------------------------------------------------------------------

def _slow_connect_serving(event: dict):
    """A mock websocket whose first recv() blocks, then serves EVENT + EOSE."""

    def _connect(*_args, **_kwargs):
        ws = MagicMock()
        frames = [json.dumps(["EVENT", "s", event]), json.dumps(["EOSE", "s"])]

        def _recv():
            if len(frames) == 2:
                time.sleep(_DRAIN_SECONDS)
            return frames.pop(0)

        ws.recv.side_effect = _recv
        return ws

    return _connect


@pytest.mark.asyncio
async def test_bootstrap_relay_read_does_not_block_event_loop() -> None:
    from unittest.mock import AsyncMock

    from tollbooth.bootstrap import BootstrapClient
    from tollbooth.bootstrap_relay import send_bootstrap_config

    operator, authority = PrivateKey(), PrivateKey()
    captured: dict = {}

    def _capture(msg: str) -> None:
        arr = json.loads(msg)
        if arr[0] == "EVENT":
            captured["event"] = arr[1]

    publisher = MagicMock()
    publisher.recv.return_value = json.dumps(["OK", "id", True, ""])
    publisher.send.side_effect = _capture
    with patch("websocket.create_connection", return_value=publisher):
        assert send_bootstrap_config(
            authority_nsec=authority.bech32(), operator_npub=operator.public_key.bech32(),
            config={"neon_database_url": "postgresql://x"}, relays=["wss://relay.test"],
        )
    event = captured["event"]

    oracle = AsyncMock()
    oracle.resolve_authority_for = AsyncMock(return_value=None)
    oracle.session = MagicMock()  # a session yields a client bound to one connection
    oracle.session.return_value.__aenter__.return_value = oracle
    client = BootstrapClient(nsec_hex=operator.bech32(), relays=["wss://relay.test"])
    progressed: list[float] = []

    with patch("websocket.create_connection", side_effect=_slow_connect_serving(event)), \
         patch("tollbooth.oracle_client.default_oracle_client", return_value=oracle):
        start = time.monotonic()
        result, _ = await asyncio.gather(client.bootstrap(), _ticker(progressed))

    assert result.success is True
    assert progressed, "ticker coroutine never ran"
    assert progressed[0] - start < _BLOCKED_THRESHOLD, (
        f"event loop appears blocked: tick at "
        f"{progressed[0] - start:.3f}s (relay read is {_DRAIN_SECONDS}s)"
    )


@pytest.mark.asyncio
async def test_relay_liveness_probe_does_not_block_event_loop() -> None:
    from tollbooth.nostr_diagnostics import resolve_relays

    def _slow_probe(*_args, **_kwargs):
        time.sleep(_DRAIN_SECONDS)
        return MagicMock()

    relays = ["wss://primary.test", "wss://second.test"]
    progressed: list[float] = []

    with patch("tollbooth.relay_registry.get_relays", return_value=relays), \
         patch("tollbooth.nostr_diagnostics.create_connection", side_effect=_slow_probe):
        start = time.monotonic()
        live, _ = await asyncio.gather(
            asyncio.to_thread(resolve_relays, timeout=2), _ticker(progressed),
        )

    assert live == relays  # registry order, both live
    assert progressed, "ticker coroutine never ran"
    assert progressed[0] - start < _BLOCKED_THRESHOLD, (
        f"event loop appears blocked: tick at "
        f"{progressed[0] - start:.3f}s (probe is {_DRAIN_SECONDS}s)"
    )
