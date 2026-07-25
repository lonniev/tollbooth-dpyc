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

from tollbooth.registry import DPYCRegistry, resolve_purchase_mode
from tollbooth.runtime import OperatorRuntime

# Prime → {penultimate authority → sub-authority}, plus an operator under the
# penultimate authority. Mirrors the live Prime → NorthAmerica → NewEngland chain.
CHAIN_MEMBERS = [
    {"npub": "npub1prime", "role": "prime_authority", "status": "active",
     "upstream_authority_npub": None},
    {"npub": "npub1penult", "role": "authority", "status": "active",
     "upstream_authority_npub": "npub1prime"},
    {"npub": "npub1subauth", "role": "authority", "status": "active",
     "upstream_authority_npub": "npub1penult"},
    {"npub": "npub1operator", "role": "operator", "status": "active",
     "upstream_authority_npub": "npub1penult"},
]


# --------------------------------------------------------------------------
# resolve_purchase_mode — the registry rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "npub,expected",
    [
        ("npub1prime", "direct"),       # root: nothing above to certify against
        ("npub1penult", "direct"),      # parent is Prime (runs no certify MCP) → self-fund
        ("npub1subauth", "certified"),  # parent is a real Authority → certify up
        ("npub1operator", "certified"), # operator under a real Authority → certify up
    ],
)
@pytest.mark.asyncio
async def test_resolve_purchase_mode_rule(npub, expected):
    with patch.object(DPYCRegistry, "_fetch", AsyncMock(return_value=CHAIN_MEMBERS)):
        assert await resolve_purchase_mode(npub) == expected


@pytest.mark.asyncio
async def test_resolve_purchase_mode_unknown_npub_raises():
    with patch.object(DPYCRegistry, "_fetch", AsyncMock(return_value=CHAIN_MEMBERS)):  # noqa: SIM117
        with pytest.raises(ValueError, match="not in the dpyc-community registry"):
            await resolve_purchase_mode("npub1ghost")


@pytest.mark.asyncio
async def test_resolve_purchase_mode_unknown_parent_is_direct():
    """A dangling parent reference fails toward trust-root (no upstream cert)."""
    orphan = [{"npub": "npub1orphan", "role": "authority", "status": "active",
               "upstream_authority_npub": "npub1missing"}]
    with patch.object(DPYCRegistry, "_fetch", AsyncMock(return_value=orphan)):
        assert await resolve_purchase_mode("npub1orphan") == "direct"


# --------------------------------------------------------------------------
# _effective_purchase_mode — auto resolution, caching, fail-safe
# --------------------------------------------------------------------------

def _rt(purchase_mode: str, vault_source: str = "authority") -> OperatorRuntime:
    rt = OperatorRuntime(tool_registry={}, purchase_mode=purchase_mode,
                         vault_source=vault_source)
    rt._operator_npub = "npub1subauth"  # bypass nsec derivation
    return rt


@pytest.mark.asyncio
async def test_effective_mode_explicit_passthrough_skips_registry():
    with patch("tollbooth.registry.resolve_purchase_mode",
               AsyncMock(side_effect=AssertionError("must not hit registry"))):
        assert await _rt("certified")._effective_purchase_mode() == "certified"
        assert await _rt("direct")._effective_purchase_mode() == "direct"


@pytest.mark.asyncio
async def test_effective_mode_auto_resolves_once_and_caches():
    rt = _rt("auto")
    m = AsyncMock(return_value="certified")
    with patch("tollbooth.registry.resolve_purchase_mode", m):
        assert await rt._effective_purchase_mode() == "certified"
        assert await rt._effective_purchase_mode() == "certified"
    m.assert_called_once()  # second call served from cache


@pytest.mark.asyncio
async def test_effective_mode_auto_failsafe_direct_and_not_cached():
    """Registry outage falls back to direct for this call only — retried next time."""
    rt = _rt("auto")
    m = AsyncMock(side_effect=RuntimeError("registry unreachable"))
    with patch("tollbooth.registry.resolve_purchase_mode", m):
        assert await rt._effective_purchase_mode() == "direct"
        assert await rt._effective_purchase_mode() == "direct"
    assert m.call_count == 2  # fallback not cached
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
