"""An operator whose config sits on too few relays spreads it — without waiting.

Six operators' configs lived on nos.lol alone (2026-09-28), and the Authority's
refresh is a background task Horizon's freeze rarely lets finish. The operator
already holds the signed event after bootstrap, so it re-broadcasts it; the
bootstrap answer must never wait on that.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.bootstrap import BootstrapClient
from tollbooth.bootstrap_relay import ConfigRead, PublishResult

RELAYS = ["wss://nos.lol", "wss://a", "wss://b", "wss://c"]
EVENT = {"id": "e" * 64, "pubkey": "a" * 64, "kind": 30078, "sig": "s" * 128}
CONFIG = {"neon_database_url": "postgresql://x"}


def _oracle():
    o = MagicMock()
    o.get_relays = AsyncMock(return_value=RELAYS)
    o.resolve_authority_for = AsyncMock(return_value=None)
    o.session.return_value.__aenter__.return_value = o
    return o


async def _bootstrap(read: ConfigRead, broadcast):
    with patch("tollbooth.oracle_client.default_oracle_client", return_value=_oracle()), \
         patch("tollbooth.bootstrap_relay.receive_bootstrap_config", return_value=read), \
         patch("tollbooth.bootstrap_relay.broadcast_signed_event", side_effect=broadcast) as sent:
        result = await BootstrapClient(nsec_hex="1" * 64).bootstrap()
        return result, sent


@pytest.mark.asyncio
async def test_a_thin_config_is_spread_to_the_relays_that_lack_it_after_bootstrap_answers():
    release = threading.Event()
    reached: list = []

    def slow_broadcast(event, relays):
        reached.append((event, relays))
        release.wait(3)
        return PublishResult(len(relays), 0)

    read = ConfigRead(CONFIG, "a" * 64, "d", EVENT, ("wss://nos.lol",))
    result, _ = await asyncio.wait_for(_bootstrap(read, slow_broadcast), timeout=1)
    assert result.success, "bootstrap answered while the spread was still blocked"

    from tollbooth import bootstrap as b

    for _ in range(50):
        if reached:
            break
        await asyncio.sleep(0.01)
    assert reached == [(EVENT, ["wss://a", "wss://b", "wss://c"])]
    release.set()
    await asyncio.gather(*b._spreads)


@pytest.mark.asyncio
async def test_a_well_held_config_is_left_alone():
    read = ConfigRead(CONFIG, "a" * 64, "d", EVENT, ("wss://nos.lol", "wss://a", "wss://b"))
    result, sent = await _bootstrap(read, lambda e, r: PublishResult(0, 0))
    await asyncio.sleep(0)
    assert result.success
    sent.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_spread_never_touches_the_bootstrap_result():
    def broken(event, relays):
        raise ConnectionError("every relay refused")

    read = ConfigRead(CONFIG, "a" * 64, "d", EVENT, ("wss://nos.lol",))
    result, _ = await _bootstrap(read, broken)
    from tollbooth import bootstrap as b

    await asyncio.gather(*b._spreads)
    assert result.success
