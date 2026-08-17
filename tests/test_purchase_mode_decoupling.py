"""Tests for the vault_source / purchase_mode decoupling and registry-derived
certify-up mode (``purchase_mode="auto"``).

Background: ``purchase_mode`` historically conflated two axes — where the Neon
URL comes from (env vs Authority bootstrap) and whether a purchase order is
certified up to a parent Authority. A sub-Authority (e.g. NewEngland under
NorthAmerica) needs env-vault *and* certified, a combination the conflated flag
could not express. These tests pin the split behavior.
"""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from tollbooth.runtime import OperatorRuntime

# NOTE: the registry-topology rule (certified vs direct) now lives in the Oracle
# (dpyc-oracle CommunityRegistry.purchase_mode / resolve_service). Operators are
# nsec-only and never read GitHub, so these tests exercise the runtime's
# consumption of the Oracle answer, not the rule itself.


# --------------------------------------------------------------------------
# _effective_purchase_mode — auto resolution via the Oracle, caching, no fallback
# --------------------------------------------------------------------------

def _rt(purchase_mode: str, vault_source: str = "authority") -> OperatorRuntime:
    rt = OperatorRuntime(tool_registry={}, purchase_mode=purchase_mode,
                         vault_source=vault_source)
    rt._operator_npub = "npub1subauth"  # bypass nsec derivation
    return rt


def _oracle_service(mode):
    """Patch target for default_oracle_client → client.resolve_service→{purchase_mode}."""
    client = MagicMock()
    if isinstance(mode, Exception):
        client.resolve_service = AsyncMock(side_effect=mode)
    else:
        client.resolve_service = AsyncMock(
            return_value=None if mode is None else {"purchase_mode": mode}
        )
    factory = MagicMock(return_value=client)
    factory.client = client
    return factory


@pytest.mark.asyncio
async def test_effective_mode_explicit_passthrough_skips_oracle():
    factory = _oracle_service(AssertionError("must not hit the Oracle"))
    with patch("tollbooth.oracle_client.default_oracle_client", factory):
        assert await _rt("certified")._effective_purchase_mode() == "certified"
        assert await _rt("direct")._effective_purchase_mode() == "direct"
    factory.client.resolve_service.assert_not_called()


@pytest.mark.asyncio
async def test_effective_mode_auto_resolves_once_and_caches():
    rt = _rt("auto")
    factory = _oracle_service("certified")
    with patch("tollbooth.oracle_client.default_oracle_client", factory):
        assert await rt._effective_purchase_mode() == "certified"
        assert await rt._effective_purchase_mode() == "certified"
    factory.client.resolve_service.assert_called_once()  # second call served from cache


@pytest.mark.asyncio
async def test_effective_mode_auto_raises_when_oracle_cannot_resolve():
    """No silent 'direct' fallback: an unresolvable mode raises and is not cached."""
    rt = _rt("auto")
    factory = _oracle_service(None)  # Oracle returns no service
    with patch("tollbooth.oracle_client.default_oracle_client", factory):
        with pytest.raises(RuntimeError, match="purchase_mode"):
            await rt._effective_purchase_mode()
        with pytest.raises(RuntimeError, match="purchase_mode"):
            await rt._effective_purchase_mode()
    assert factory.client.resolve_service.call_count == 2  # not cached
    assert rt._resolved_purchase_mode is None


# --------------------------------------------------------------------------
# vault() — keys off vault_source, independent of purchase_mode
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vault_env_source_reads_neon_env(monkeypatch):
    monkeypatch.setenv("NEON_DATABASE_URL", "postgres://from-env")
    # Self-provisioning actors now encrypt at rest with their own nsec, so the
    # runtime must have a signing key available.
    monkeypatch.setenv(
        "TOLLBOOTH_NOSTR_OPERATOR_NSEC",
        "nsec1vl029mgpspedva04g90vltkh6fvh240zqtv9k0t9af8935ke9laqsnlfe5",
    )
    rt = OperatorRuntime(tool_registry={}, purchase_mode="certified", vault_source="env")
    with patch("tollbooth.vaults.NeonVault") as NV:
        NV.return_value.ensure_schema = AsyncMock()
        await rt.vault()
    # certified + env (the NewEngland combo) self-provisions from env AND
    # encrypts with the actor's own nsec (no more plaintext ledger column).
    NV.assert_called_once_with(
        database_url="postgres://from-env", encryption_nsec_hex=ANY
    )


@pytest.mark.asyncio
async def test_vault_env_source_missing_url_raises(monkeypatch):
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    rt = OperatorRuntime(tool_registry={}, vault_source="env")
    with pytest.raises(ValueError, match="vault_source='env'"):
        await rt.vault()


@pytest.mark.asyncio
async def test_vault_authority_source_ignores_purchase_mode(monkeypatch):
    """purchase_mode='direct' no longer forces env vault — vault_source decides."""
    monkeypatch.setenv("NEON_DATABASE_URL", "postgres://should-not-be-used")
    rt = OperatorRuntime(tool_registry={}, purchase_mode="direct", vault_source="authority")
    boot = MagicMock(success=True, neon_database_url="postgres://bootstrapped",
                     encryption_nsec_hex="ff", error=None)
    with patch("tollbooth.bootstrap.ensure_bootstrapped", AsyncMock(return_value=boot)), \
         patch("tollbooth.vaults.NeonVault") as NV:
        NV.return_value.ensure_schema = AsyncMock()
        await rt.vault()
    NV.assert_called_once_with(database_url="postgres://bootstrapped",
                               encryption_nsec_hex="ff")
