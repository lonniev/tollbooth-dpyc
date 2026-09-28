"""The bootstrap starts when a client connects, and the first call joins it.

A cold process wakes on the MCP handshake. These tests pin the three things
that make an early start safe: the handshake triggers exactly one warm-up and
is never delayed by it; the first real call waits on that same attempt rather
than opening a second vault; and a warm-up that fails leaves nothing behind,
so the next connection or call simply tries again.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tollbooth.runtime import OperatorRuntime
from tollbooth.warm_start import WarmOnInitialize


def _runtime(open_vault) -> OperatorRuntime:
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    rt._operator_npub = "npub1test"
    rt._open_vault = open_vault  # type: ignore[method-assign]
    return rt


@pytest.mark.asyncio
async def test_concurrent_callers_open_one_vault():
    opened = 0

    async def open_vault():
        nonlocal opened
        opened += 1
        await asyncio.sleep(0.05)
        return MagicMock(name="vault")

    rt = _runtime(open_vault)
    vaults = await asyncio.gather(*(rt.vault() for _ in range(5)))
    assert opened == 1, "one bootstrap, however many callers arrive together"
    assert all(v is vaults[0] for v in vaults)


@pytest.mark.asyncio
async def test_the_handshake_is_not_delayed_and_the_first_call_joins_the_warm_up():
    started = asyncio.Event()
    release = asyncio.Event()
    opened = 0

    async def open_vault():
        nonlocal opened
        opened += 1
        started.set()
        await release.wait()
        return MagicMock(name="vault")

    rt = _runtime(open_vault)

    async def resolver():
        await rt.vault()
        return MagicMock(_ensure_fresh=AsyncMock())

    rt.pricing_resolver = resolver  # type: ignore[method-assign]

    handshake = AsyncMock(return_value="initialized")
    result = await asyncio.wait_for(WarmOnInitialize(rt).on_initialize(object(), handshake), 0.5)
    assert result == "initialized", "the handshake returns at once, warm-up or not"
    await asyncio.wait_for(started.wait(), 0.5)

    first_call = asyncio.create_task(rt.vault())
    await asyncio.sleep(0.02)
    assert not first_call.done(), "the first call waits on the warm-up in flight"
    release.set()
    await first_call
    assert opened == 1, "the first call did not open a second vault"


@pytest.mark.asyncio
async def test_repeated_handshakes_start_one_warm_up():
    rt = _runtime(AsyncMock(return_value=MagicMock()))
    calls = 0

    async def resolver():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return MagicMock(_ensure_fresh=AsyncMock())

    rt.pricing_resolver = resolver  # type: ignore[method-assign]
    for _ in range(4):
        rt.warm()
    await rt._warm_task
    assert calls == 1


@pytest.mark.asyncio
async def test_a_failed_warm_up_is_forgotten_so_the_next_attempt_retries():
    attempts = 0

    async def open_vault():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("Bootstrap failed: relays unreachable")
        return MagicMock(name="vault")

    rt = _runtime(open_vault)

    async def resolver():
        await rt.vault()
        return MagicMock(_ensure_fresh=AsyncMock())

    rt.pricing_resolver = resolver  # type: ignore[method-assign]
    rt.warm()
    await asyncio.sleep(0.01)
    assert rt._warm_task is None and rt._vault is None, "nothing cached from a failure"
    assert await rt.vault() is not None, "the next caller bootstraps afresh"
    assert attempts == 2


@pytest.mark.asyncio
async def test_a_warm_process_does_not_warm_again():
    rt = _runtime(AsyncMock(return_value=MagicMock()))
    rt._vault = MagicMock()
    rt.warm()
    assert rt._warm_task is None
