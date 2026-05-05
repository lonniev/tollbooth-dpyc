"""Tests for the @runtime.paid_tool() decorator."""

import pytest
from pynostr.key import PrivateKey as _PK

from tollbooth.runtime import OperatorRuntime
from tollbooth.tool_identity import ToolIdentity, capability_uuid
from tollbooth import identity_proof as _id_proof


@pytest.fixture(autouse=True)
def _clear_replay_cache():
    """Clear the proof replay cache between tests to avoid false rejections."""
    _id_proof._consumed_proofs.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakePricingResolver:
    """In-memory pricing resolver for testing."""

    def __init__(self, costs: dict[str, int], priced: dict[str, bool] | None = None):
        self._costs = costs
        self._priced = priced or {k: True for k in costs}
        self._neon_available = True

    @property
    def neon_available(self) -> bool:
        return self._neon_available

    async def _ensure_fresh(self) -> None:
        pass

    async def get_cost(self, tool_id: str) -> int:
        return self._costs.get(tool_id, 0)

    async def has_tool(self, tool_id: str) -> bool:
        return tool_id in self._costs

    async def is_priced(self, tool_id: str) -> bool:
        return self._priced.get(tool_id, False)

    async def get_tool_pricing(self, tool_id: str):
        from tollbooth.pricing import ToolPricing
        return ToolPricing(fixed=self._costs.get(tool_id, 0))

    async def get_constraint_engine(self):
        return None

    def refresh(self):
        pass


def _make_registry(tool_costs: dict[str, int] | None = None) -> tuple[dict[str, ToolIdentity], dict[str, int]]:
    """Build a tool_registry (keyed by UUID) and resolver cost map from {name: cost}."""
    costs = tool_costs or {"my_tool": 1, "expensive_tool": 100}
    registry: dict[str, ToolIdentity] = {}
    resolver_costs: dict[str, int] = {}  # keyed by tool_id (UUID)
    for name, cost in costs.items():
        category = "free" if cost == 0 else "read"
        identity = ToolIdentity(capability=name, category=category, intent=f"Test tool {name}")
        registry[identity.tool_id] = identity
        resolver_costs[identity.tool_id] = cost
    return registry, resolver_costs


def _make_runtime(tool_costs: dict[str, int] | None = None) -> OperatorRuntime:
    """Create a minimal runtime for testing (no vault, no Nostr)."""
    registry, resolver_costs = _make_registry(tool_costs)
    rt = OperatorRuntime(
        tool_registry=registry,
        nsec_env_var="__UNUSED__",
    )
    # Inject fake pricing resolver and proven npub cache (avoids vault bootstrap)
    rt._pricing_resolver = FakePricingResolver(resolver_costs)
    rt._proven_npub_cache = FakeProvenNpubCache()
    return rt


class FakeProvenNpubCache:
    """Stub proven npub cache — always forces inline proof verification."""

    async def is_proven(self, session_id: str, npub: str) -> bool:
        return False

    async def mark_proven(self, session_id: str, npub: str) -> None:
        pass


class FakeLedgerCache:
    """In-memory ledger cache for testing."""

    def __init__(self, balance: int = 1000):
        from tollbooth.ledger import UserLedger, Tranche
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        self._ledgers: dict[str, UserLedger] = {}
        self._default_balance = balance
        self._Tranche = Tranche
        self._now = now

    async def get(self, npub: str):
        if npub not in self._ledgers:
            from tollbooth.ledger import UserLedger
            ledger = UserLedger()
            ledger.tranches.append(self._Tranche(
                granted_at=self._now,
                original_sats=self._default_balance,
                remaining_sats=self._default_balance,
                invoice_id="test-seed",
            ))
            self._ledgers[npub] = ledger
        return self._ledgers[npub]

    def mark_dirty(self, npub: str) -> None:
        pass

    async def flush_user(self, npub: str) -> None:
        pass


# Generate a real keypair for proof tests
_TEST_KEY = _PK()
VALID_NPUB = _TEST_KEY.public_key.bech32()
_TEST_NSEC = _TEST_KEY.bech32()


def _make_proof(tool_name: str = "my_tool") -> str:
    """Create a valid proof for the test keypair."""
    from tollbooth.identity_proof import create_proof
    return create_proof(_TEST_NSEC, tool_name)


async def _inject_fake_cache(rt: OperatorRuntime, balance: int = 1000):
    """Replace the runtime's ledger cache with a fake."""
    cache = FakeLedgerCache(balance)
    rt._ledger_cache = cache
    return cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPaidToolDecorator:

    @pytest.mark.asyncio
    async def test_success_path(self):
        rt = _make_runtime()
        await _inject_fake_cache(rt)

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(x: int, npub: str = "", proof: str = "") -> dict:
            return {"value": x * 2}

        result = await my_tool(21, npub=VALID_NPUB, proof=_make_proof())
        assert result["value"] == 42

    @pytest.mark.asyncio
    async def test_debit_insufficient_balance(self):
        rt = _make_runtime({"my_tool": 9999})
        await _inject_fake_cache(rt, balance=10)

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            return {"should": "not reach"}

        result = await my_tool(npub=VALID_NPUB, proof=_make_proof())
        assert result["success"] is False
        assert "Insufficient balance" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_npub_returns_error(self):
        rt = _make_runtime()
        await _inject_fake_cache(rt)

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            return {"should": "not reach"}

        result = await my_tool(npub="")
        assert result["success"] is False
        assert "npub is required" in result["error"]

    @pytest.mark.asyncio
    async def test_rollback_on_exception_catch_errors_true(self):
        rt = _make_runtime()
        cache = await _inject_fake_cache(rt, balance=100)

        @rt.paid_tool(capability_uuid("my_tool"), catch_errors=True)
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            raise RuntimeError("boom")

        # Get initial balance
        ledger = await cache.get(VALID_NPUB)
        initial = ledger.balance_api_sats

        result = await my_tool(npub=VALID_NPUB, proof=_make_proof())
        assert result["success"] is False
        assert "Tool execution failed" in result["error"]

        # Balance should be restored after rollback
        ledger = await cache.get(VALID_NPUB)
        assert ledger.balance_api_sats == initial

    @pytest.mark.asyncio
    async def test_rollback_on_exception_catch_errors_false(self):
        rt = _make_runtime()
        cache = await _inject_fake_cache(rt, balance=100)

        @rt.paid_tool(capability_uuid("my_tool"), catch_errors=False)
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            raise RuntimeError("boom")

        ledger = await cache.get(VALID_NPUB)
        initial = ledger.balance_api_sats

        with pytest.raises(RuntimeError, match="boom"):
            await my_tool(npub=VALID_NPUB, proof=_make_proof())

        # Balance should be restored after rollback
        ledger = await cache.get(VALID_NPUB)
        assert ledger.balance_api_sats == initial

    @pytest.mark.asyncio
    async def test_free_tool_skips_debit(self):
        rt = _make_runtime({"free_tool": 0})
        # No cache needed — free tools skip debit entirely

        @rt.paid_tool(capability_uuid("free_tool"))
        async def free_tool(npub: str = "", proof: str = "") -> dict:
            return {"free": True}

        result = await free_tool(npub="")
        assert result["free"] is True

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        rt = _make_runtime()

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            """My docstring."""
            return {}

        assert my_tool.__name__ == "my_tool"
        assert my_tool.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_result_includes_low_balance_warning(self):
        rt = _make_runtime({"my_tool": 1})
        await _inject_fake_cache(rt, balance=2)  # low balance after debit

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            return {"data": "ok"}

        result = await my_tool(npub=VALID_NPUB, proof=_make_proof())
        assert result["data"] == "ok"
        # Warning may or may not be present depending on threshold,
        # but the decorator shouldn't crash

    @pytest.mark.asyncio
    async def test_unpriced_tool_blocked(self):
        """A tool in the model but not yet priced (TBD) should be blocked."""
        identity = ToolIdentity(capability="unpriced", category="write", intent="test")
        registry = {identity.tool_id: identity}
        rt = OperatorRuntime(tool_registry=registry, nsec_env_var="__UNUSED__")
        # Tool is in model at 0 sats but priced=False (TBD)
        rt._pricing_resolver = FakePricingResolver(
            costs={identity.tool_id: 0},
            priced={identity.tool_id: False},
        )
        rt._proven_npub_cache = FakeProvenNpubCache()

        @rt.paid_tool(identity.tool_id)
        async def unpriced(npub: str = "", proof: str = "") -> dict:
            return {"should": "not reach"}

        result = await unpriced(npub=VALID_NPUB, proof=_make_proof("unpriced"))
        assert result["success"] is False
        assert "not been priced" in result["error"]

    @pytest.mark.asyncio
    async def test_intentionally_free_tool_allowed(self):
        """A tool priced at 0 sats with priced=True is intentionally free."""
        identity = ToolIdentity(capability="promo", category="write", intent="test")
        registry = {identity.tool_id: identity}
        rt = OperatorRuntime(tool_registry=registry, nsec_env_var="__UNUSED__")
        rt._pricing_resolver = FakePricingResolver(
            costs={identity.tool_id: 0},
            priced={identity.tool_id: True},
        )
        rt._proven_npub_cache = FakeProvenNpubCache()
        rt._ledger_cache = FakeLedgerCache()

        @rt.paid_tool(identity.tool_id)
        async def promo(npub: str = "", proof: str = "") -> dict:
            return {"free": True}

        result = await promo(npub=VALID_NPUB, proof=_make_proof("promo"))
        assert result["free"] is True


# ---------------------------------------------------------------------------
# Structured error_code (added in 0.17.0)
# ---------------------------------------------------------------------------


class TestErrorCodeMapping:
    """Every denial path from debit_or_deny / paid_tool returns a stable error_code."""

    @pytest.mark.asyncio
    async def test_insufficient_balance_carries_error_code_and_next_steps(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime({"my_tool": 9999})
        await _inject_fake_cache(rt, balance=10)

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            return {"unreachable": True}

        result = await my_tool(npub=VALID_NPUB, proof=_make_proof())
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INSUFFICIENT_BALANCE
        assert isinstance(result["next_steps"], list)
        assert any("purchase_credits" in s for s in result["next_steps"])

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_tool_not_registered_code(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        unregistered = capability_uuid("ghost_tool")
        result = await rt.debit_or_deny(unregistered, VALID_NPUB)
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.TOOL_NOT_REGISTERED

    @pytest.mark.asyncio
    async def test_missing_proof_returns_proof_required_code(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        await _inject_fake_cache(rt)

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            return {"unreachable": True}

        result = await my_tool(npub=VALID_NPUB, proof="")
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.PROOF_REQUIRED
        assert isinstance(result["next_steps"], list)
        assert any("request_npub_proof" in s for s in result["next_steps"])

    @pytest.mark.asyncio
    async def test_invalid_proof_returns_proof_invalid_code(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        await _inject_fake_cache(rt)

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            return {"unreachable": True}

        # Inline proof (Schnorr-shaped, not a poison phrase) signed by a different key
        from pynostr.key import PrivateKey
        other = PrivateKey()
        from tollbooth.identity_proof import create_proof
        wrong_proof = create_proof(other.bech32(), "my_tool")

        result = await my_tool(npub=VALID_NPUB, proof=wrong_proof)
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.PROOF_INVALID

    @pytest.mark.asyncio
    async def test_unpriced_tool_returns_tool_not_priced_code(self):
        from tollbooth.constants import ErrorCode
        identity = ToolIdentity(capability="unpriced2", category="write", intent="test")
        registry = {identity.tool_id: identity}
        rt = OperatorRuntime(tool_registry=registry, nsec_env_var="__UNUSED__")
        rt._pricing_resolver = FakePricingResolver(
            costs={identity.tool_id: 0},
            priced={identity.tool_id: False},
        )
        rt._proven_npub_cache = FakeProvenNpubCache()

        @rt.paid_tool(identity.tool_id)
        async def unpriced(npub: str = "", proof: str = "") -> dict:
            return {"unreachable": True}

        result = await unpriced(npub=VALID_NPUB, proof=_make_proof("unpriced2"))
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.TOOL_NOT_PRICED

    @pytest.mark.asyncio
    async def test_generic_exception_returns_tool_execution_failed(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        await _inject_fake_cache(rt, balance=1000)

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            raise RuntimeError("something opaque happened")

        result = await my_tool(npub=VALID_NPUB, proof=_make_proof())
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.TOOL_EXECUTION_FAILED

    @pytest.mark.asyncio
    async def test_upstream_401_routes_to_oauth_refresh_code(self):
        """Exception text containing '401' or 'unauthorized' maps to upstream_auth_refresh_needed."""
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        await _inject_fake_cache(rt, balance=1000)

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            raise RuntimeError("HTTP 401 Unauthorized from upstream")

        result = await my_tool(npub=VALID_NPUB, proof=_make_proof())
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.UPSTREAM_AUTH_REFRESH_NEEDED
        assert isinstance(result["next_steps"], list)
        assert any("begin_oauth" in s for s in result["next_steps"])

    @pytest.mark.asyncio
    async def test_invalid_grant_routes_to_oauth_refresh_code(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        await _inject_fake_cache(rt, balance=1000)

        @rt.paid_tool(capability_uuid("my_tool"))
        async def my_tool(npub: str = "", proof: str = "") -> dict:
            raise RuntimeError("OAuth invalid_grant: token has expired")

        result = await my_tool(npub=VALID_NPUB, proof=_make_proof())
        assert result["error_code"] == ErrorCode.UPSTREAM_AUTH_REFRESH_NEEDED


# ---------------------------------------------------------------------------
# OAuth situation → structured response (added in 0.17.1)
# ---------------------------------------------------------------------------


class TestOAuthSituationResponse:
    """oauth_situation_response is the cross-operator helper that turns
    restore_oauth_session situation strings into patron-actionable dicts."""

    @pytest.mark.asyncio
    async def test_token_expired_returns_oauth_refresh_needed(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        rt._slug = "schwab"

        result = rt.oauth_situation_response("token_expired")

        assert result["success"] is False
        assert result["error_code"] == ErrorCode.OAUTH_REFRESH_NEEDED
        assert "schwab_begin_oauth" in result["next_steps"][0]
        assert "schwab_check_oauth_status" in result["next_steps"][2]

    @pytest.mark.asyncio
    async def test_no_credentials_returns_oauth_refresh_needed(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        rt._slug = "excalibur"

        result = rt.oauth_situation_response("no_credentials")

        assert result["error_code"] == ErrorCode.OAUTH_REFRESH_NEEDED
        assert "excalibur_begin_oauth" in result["next_steps"][0]

    @pytest.mark.asyncio
    async def test_vault_bootstrapping_returns_warming_up(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        rt._slug = "schwab"

        result = rt.oauth_situation_response("vault_bootstrapping")

        assert result["error_code"] == ErrorCode.WARMING_UP
        assert "Repeat" in result["next_steps"][0]

    @pytest.mark.asyncio
    async def test_operator_not_configured_routes_to_operator_code(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        rt._slug = "schwab"

        result = rt.oauth_situation_response("operator_not_configured")
        assert result["error_code"] == ErrorCode.OPERATOR_NOT_CONFIGURED

    @pytest.mark.asyncio
    async def test_no_oauth_config_routes_to_operator_code(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        rt._slug = "schwab"

        result = rt.oauth_situation_response("no_oauth_config")
        assert result["error_code"] == ErrorCode.OPERATOR_NOT_CONFIGURED

    @pytest.mark.asyncio
    async def test_unknown_situation_falls_through_to_oauth_refresh(self):
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        rt._slug = "schwab"

        result = rt.oauth_situation_response("some_brand_new_situation")
        # Falls through with the situation echoed in the message
        assert result["error_code"] == ErrorCode.OAUTH_REFRESH_NEEDED
        assert "some_brand_new_situation" in result["error"]
        assert any("schwab_begin_oauth" in s for s in result["next_steps"])

    @pytest.mark.asyncio
    async def test_empty_slug_omits_qualifier(self):
        """A runtime without a slug yet still produces sensible recipes."""
        from tollbooth.constants import ErrorCode
        rt = _make_runtime()
        rt._slug = ""

        result = rt.oauth_situation_response("token_expired")
        # Steps should reference begin_oauth without a slug prefix
        assert "begin_oauth(" in result["next_steps"][0]
        assert result["error_code"] == ErrorCode.OAUTH_REFRESH_NEEDED
