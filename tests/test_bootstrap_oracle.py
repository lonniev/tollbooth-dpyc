"""Operator bootstrap is GitHub-free: relays + Authority come from the Oracle.

Guards the cutover where the operator stopped reading the dpyc-community
registry on GitHub. Cold start now needs only the nsec plus MCP calls to the
Oracle (relays, and best-effort Authority resolution); the config itself is read
from Nostr by the operator's own d-tag.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.bootstrap import BootstrapClient

OP = PrivateKey()
AUTH = PrivateKey()


def _oracle(*, relays=None, authority=None, relays_exc=None):
    oracle = MagicMock()
    oracle.get_relays = (
        AsyncMock(side_effect=relays_exc)
        if relays_exc is not None
        else AsyncMock(return_value=relays or ["wss://r.test"])
    )
    oracle.resolve_authority_for = AsyncMock(return_value=authority)
    return oracle


@pytest.mark.asyncio
async def test_bootstrap_is_github_free_and_pins_expected_authority():
    """Oracle resolves the Authority → its hex is passed as the spoof-guard."""
    client = BootstrapClient(nsec_hex=OP.bech32())
    oracle = _oracle(
        relays=["wss://r.test"],
        authority={"npub": AUTH.public_key.bech32(), "url": "https://a/mcp", "name": "auth"},
    )
    with patch("tollbooth.oracle_client.default_oracle_client", return_value=oracle), patch(
        "tollbooth.bootstrap_relay.receive_bootstrap_config",
        return_value=({"neon_database_url": "postgresql://x"}, AUTH.public_key.hex(), "diag"),
    ) as rbc:
        result = await client.bootstrap()

    assert result.success
    assert result.neon_database_url == "postgresql://x"
    assert result.authority_npub == AUTH.public_key.bech32()
    kwargs = rbc.call_args.kwargs
    assert kwargs["relays"] == ["wss://r.test"]  # relays came from the Oracle
    assert kwargs["expected_authority_hex"] == AUTH.public_key.hex()


@pytest.mark.asyncio
async def test_bootstrap_discovers_authority_from_event_when_oracle_cannot_resolve():
    client = BootstrapClient(nsec_hex=OP.bech32())
    oracle = _oracle(relays=["wss://r.test"], authority=None)  # Oracle can't resolve
    with patch("tollbooth.oracle_client.default_oracle_client", return_value=oracle), patch(
        "tollbooth.bootstrap_relay.receive_bootstrap_config",
        return_value=({"neon_database_url": "postgresql://y"}, AUTH.public_key.hex(), "diag"),
    ) as rbc:
        result = await client.bootstrap()

    assert result.success
    # Authority discovered from the event's author, not from a pre-resolve.
    assert result.authority_npub == AUTH.public_key.bech32()
    assert rbc.call_args.kwargs["expected_authority_hex"] is None


@pytest.mark.asyncio
async def test_bootstrap_fails_clearly_when_oracle_unreachable_for_relays():
    client = BootstrapClient(nsec_hex=OP.bech32())
    oracle = _oracle(relays_exc=RuntimeError("oracle down"))
    with patch("tollbooth.oracle_client.default_oracle_client", return_value=oracle):
        result = await client.bootstrap()

    assert not result.success
    assert "Oracle" in (result.error or "")


@pytest.mark.asyncio
async def test_bootstrap_never_calls_the_github_registry():
    """The whole point: no registry.resolve_authority_service on the boot path."""
    client = BootstrapClient(nsec_hex=OP.bech32())
    oracle = _oracle(authority={"npub": AUTH.public_key.bech32(), "url": "u", "name": "n"})
    with patch("tollbooth.oracle_client.default_oracle_client", return_value=oracle), patch(
        "tollbooth.bootstrap_relay.receive_bootstrap_config",
        return_value=({"neon_database_url": "postgresql://z"}, AUTH.public_key.hex(), "d"),
    ), patch("tollbooth.registry.resolve_authority_service") as reg:
        result = await client.bootstrap()

    assert result.success
    reg.assert_not_called()
