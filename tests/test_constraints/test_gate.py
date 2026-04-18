"""Tests for the ConstraintGate middleware integration helper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from tollbooth.config import TollboothConfig
from tollbooth.constraints.gate import ConstraintGate


# ---------------------------------------------------------------------------
# Lightweight ledger stub — mirrors UserLedger's public attributes
# ---------------------------------------------------------------------------


@dataclass
class _StubLedger:
    balance_api_sats: int = 1000
    total_deposited_api_sats: int = 2000
    total_consumed_api_sats: int = 500
    total_expired_api_sats: int = 100


# ---------------------------------------------------------------------------
# Helpers — config JSON builders
# ---------------------------------------------------------------------------


def _free_trial_config(first_n_free: int = 3) -> str:
    """Return a JSON config with a free trial on 'search'."""
    return json.dumps(
        {
            "tool_constraints": {
                "search": {
                    "constraints": [
                        {"type": "free_trial", "first_n_free": first_n_free}
                    ]
                }
            }
        }
    )


def _coupon_config(discount_percent: float = 50.0) -> str:
    """Return a JSON config with a coupon discount on 'search'."""
    return json.dumps(
        {
            "tool_constraints": {
                "search": {
                    "constraints": [
                        {
                            "type": "coupon",
                            "code": "HALF_OFF",
                            "discount_percent": discount_percent,
                        }
                    ]
                }
            }
        }
    )


def _temporal_window_config(start: str = "09:00", end: str = "17:00") -> str:
    """Return a JSON config with a temporal window on 'search'."""
    return json.dumps(
        {
            "tool_constraints": {
                "search": {
                    "constraints": [
                        {
                            "type": "temporal_window",
                            "schedule_start": start,
                            "schedule_end": end,
                            "timezone": "UTC",
                        }
                    ]
                }
            }
        }
    )


def _supply_config(max_invocations: int = 5, current_count: int = 5) -> str:
    """Return a JSON config with a finite supply constraint on 'search'."""
    return json.dumps(
        {
            "tool_constraints": {
                "search": {
                    "constraints": [
                        {
                            "type": "finite_supply",
                            "max_invocations": max_invocations,
                            "current_count": current_count,
                            "scope": "global",
                        }
                    ]
                }
            }
        }
    )


def _wildcard_config() -> str:
    """Return a JSON config with a wildcard free trial applying to all tools."""
    return json.dumps(
        {
            "tool_constraints": {
                "*": {
                    "constraints": [
                        {"type": "free_trial", "first_n_free": 5}
                    ]
                }
            }
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDisabledByDefault:
    def test_disabled_by_default(self):
        """ConstraintGate with default TollboothConfig returns (None, base_cost)."""
        config = TollboothConfig()
        gate = ConstraintGate(config)

        assert not gate.enabled

        denial, cost = gate.check(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
        )
        assert denial is None
        assert cost == 100


class TestEnabledNoConfig:
    def test_enabled_no_config(self):
        """enabled=True but no config JSON falls back to disabled."""
        config = TollboothConfig(constraints_enabled=True)
        gate = ConstraintGate(config)

        assert not gate.enabled

        denial, cost = gate.check(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
        )
        assert denial is None
        assert cost == 100


class TestEnabledInvalidJson:
    def test_enabled_invalid_json(self):
        """Bad JSON falls back to disabled gracefully (no exception)."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config="NOT VALID JSON {{{",
        )
        gate = ConstraintGate(config)

        assert not gate.enabled

        denial, cost = gate.check(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
        )
        assert denial is None
        assert cost == 100


class TestEnabledValidConfigAllows:
    def test_enabled_valid_config_allows(self):
        """A constraint that passes returns (None, base_cost)."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_free_trial_config(first_n_free=3),
        )
        gate = ConstraintGate(config)

        assert gate.enabled

        # invocation_count=0 is within the free trial, but the result
        # is (None, effective_cost) where effective_cost=0 because free trial
        # sets PriceModifier(free=True).  For "allows" without price mod,
        # test a tool NOT covered by the config — it should pass through.
        denial, cost = gate.check(
            tool_name="unconstrained_tool",
            base_cost=200,
            ledger=_StubLedger(),
        )
        assert denial is None
        assert cost == 200


class TestEnabledWithFreeTrial:
    def test_enabled_with_free_trial(self):
        """Free trial applies PriceModifier(free=True), effective_cost=0."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_free_trial_config(first_n_free=3),
        )
        gate = ConstraintGate(config)

        denial, cost = gate.check(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
            invocation_count=0,  # within trial
        )
        assert denial is None
        assert cost == 0  # free!

    def test_free_trial_exhausted_full_price(self):
        """After trial, cost returns to base_cost (no price modifier)."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_free_trial_config(first_n_free=3),
        )
        gate = ConstraintGate(config)

        denial, cost = gate.check(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
            invocation_count=5,  # beyond trial
        )
        assert denial is None
        assert cost == 100  # full price


class TestEnabledWithCouponDiscount:
    def test_enabled_with_coupon_discount(self):
        """Coupon applies 50% discount, effective_cost halved."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_coupon_config(discount_percent=50.0),
        )
        gate = ConstraintGate(config)

        denial, cost = gate.check(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
        )
        assert denial is None
        assert cost == 50  # 50% off


class TestEnabledDeniesTemporalWindow:
    def test_enabled_denies_temporal_window(self):
        """Outside window returns (error_dict, 0) with retry_after."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_temporal_window_config("09:00", "17:00"),
        )
        gate = ConstraintGate(config)

        # Patch datetime.now so the gate sees 03:00 UTC (outside 09:00-17:00)
        fixed_time = datetime(2026, 2, 27, 3, 0, 0, tzinfo=timezone.utc)
        with patch(
            "tollbooth.constraints.gate.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = fixed_time
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            denial, cost = gate.check(
                tool_name="search",
                base_cost=100,
                ledger=_StubLedger(),
            )

        assert denial is not None
        assert cost == 0
        assert denial["success"] is False
        assert denial["constraint_reason"] == "outside_window"
        assert "retry_after" in denial


class TestEnabledDeniesSupplyExhausted:
    def test_enabled_denies_supply_exhausted(self):
        """Finite supply at cap returns (error_dict, 0)."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_supply_config(max_invocations=5, current_count=5),
        )
        gate = ConstraintGate(config)

        denial, cost = gate.check(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
        )

        assert denial is not None
        assert cost == 0
        assert denial["success"] is False
        assert denial["constraint_reason"] == "supply_exhausted"


class TestConfigFieldOnTollboothConfig:
    def test_config_field_on_tollboothconfig(self):
        """TollboothConfig accepts constraints_enabled and constraints_config."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config='{"tool_constraints": {}}',
        )
        assert config.constraints_enabled is True
        assert config.constraints_config == '{"tool_constraints": {}}'


class TestExistingConfigUnchanged:
    def test_existing_config_unchanged(self):
        """TollboothConfig() with only old fields still works."""
        config = TollboothConfig(btcpay_host="https://example.com")
        assert config.btcpay_host == "https://example.com"
        # New fields default to off
        assert config.constraints_enabled is False
        assert config.constraints_config is None


class TestCheckBuildsCorrectContext:
    def test_check_builds_correct_context(self):
        """Verify LedgerSnapshot/PatronIdentity/EnvironmentSnapshot are built
        correctly from inputs."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_free_trial_config(first_n_free=100),
        )
        gate = ConstraintGate(config)

        ledger = _StubLedger(
            balance_api_sats=999,
            total_deposited_api_sats=2000,
            total_consumed_api_sats=800,
            total_expired_api_sats=50,
        )

        # Capture the context passed to the engine by monkey-patching evaluate
        captured_contexts: list[Any] = []
        original_evaluate = gate._engine.evaluate  # type: ignore[union-attr]

        def capturing_evaluate(tool_name: str, context: Any) -> Any:
            captured_contexts.append(context)
            return original_evaluate(tool_name, context)

        gate._engine.evaluate = capturing_evaluate  # type: ignore[union-attr]

        gate.check(
            tool_name="search",
            base_cost=100,
            ledger=ledger,
            npub="npub1abc",
            membership_tier="gold",
            invocation_count=7,
        )

        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]

        # LedgerSnapshot
        assert ctx.ledger.balance_api_sats == 999
        assert ctx.ledger.total_deposited_api_sats == 2000
        assert ctx.ledger.total_consumed_api_sats == 800
        assert ctx.ledger.total_expired_api_sats == 50

        # PatronIdentity
        assert ctx.patron.npub == "npub1abc"
        assert ctx.patron.membership_tier == "gold"

        # EnvironmentSnapshot
        assert ctx.env.tool_name == "search"
        assert ctx.env.invocation_count == 7
        assert ctx.env.utc_now.tzinfo is not None


class TestWildcardConstraintsApplied:
    def test_wildcard_constraints_applied(self):
        """Constraints under '*' key apply to all tools."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_wildcard_config(),
        )
        gate = ConstraintGate(config)

        # Any tool name should get the free trial discount
        denial, cost = gate.check(
            tool_name="any_random_tool",
            base_cost=100,
            ledger=_StubLedger(),
            invocation_count=0,  # within trial
        )
        assert denial is None
        assert cost == 0  # free via wildcard free trial

        # A different tool also gets it
        denial2, cost2 = gate.check(
            tool_name="another_tool",
            base_cost=200,
            ledger=_StubLedger(),
            invocation_count=0,
        )
        assert denial2 is None
        assert cost2 == 0


# ---------------------------------------------------------------------------
# attach_resolver + check_async
# ---------------------------------------------------------------------------


class _MockResolver:
    """Minimal mock PricingResolver for gate tests."""

    def __init__(self, engine: Any = None, *, fail: bool = False):
        self._engine = engine
        self._fail = fail

    async def get_constraint_engine(self) -> Any:
        if self._fail:
            raise RuntimeError("resolver failed")
        return self._engine


class TestAttachResolver:
    def test_attach_sets_resolver(self):
        config = TollboothConfig()
        gate = ConstraintGate(config)
        resolver = _MockResolver()
        gate.attach_resolver(resolver)
        assert gate._resolver is resolver


class TestCheckAsyncNoResolver:
    @pytest.mark.asyncio
    async def test_passthrough_when_disabled(self):
        """check_async returns (None, base_cost) when gate is disabled."""
        config = TollboothConfig()
        gate = ConstraintGate(config)

        denial, cost = await gate.check_async(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
        )
        assert denial is None
        assert cost == 100

    @pytest.mark.asyncio
    async def test_uses_static_engine(self):
        """check_async uses static engine when no resolver attached."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_free_trial_config(first_n_free=3),
        )
        gate = ConstraintGate(config)

        denial, cost = await gate.check_async(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
            invocation_count=0,
        )
        assert denial is None
        assert cost == 0  # free trial


class TestCheckAsyncWithResolver:
    @pytest.mark.asyncio
    async def test_prefers_resolver_engine(self):
        """check_async uses dynamic engine from resolver over static one."""
        from tollbooth.constraints.config import load_constraints

        # Static engine: free trial (makes cost 0)
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_free_trial_config(first_n_free=3),
        )
        gate = ConstraintGate(config)

        # Dynamic engine: 50% coupon (makes cost 50)
        dynamic_cfg = json.loads(_coupon_config(discount_percent=50.0))
        dynamic_engine = load_constraints(dynamic_cfg)
        resolver = _MockResolver(engine=dynamic_engine)
        gate.attach_resolver(resolver)

        denial, cost = await gate.check_async(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
            invocation_count=0,
        )
        assert denial is None
        assert cost == 50  # coupon from resolver, not free trial from static

    @pytest.mark.asyncio
    async def test_falls_back_to_static_on_resolver_failure(self):
        """check_async falls back to static engine when resolver fails."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_free_trial_config(first_n_free=3),
        )
        gate = ConstraintGate(config)

        resolver = _MockResolver(fail=True)
        gate.attach_resolver(resolver)

        denial, cost = await gate.check_async(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
            invocation_count=0,
        )
        assert denial is None
        assert cost == 0  # free trial from static engine

    @pytest.mark.asyncio
    async def test_falls_back_to_static_when_resolver_returns_none(self):
        """check_async uses static engine when resolver returns None engine."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_free_trial_config(first_n_free=3),
        )
        gate = ConstraintGate(config)

        resolver = _MockResolver(engine=None)
        gate.attach_resolver(resolver)

        denial, cost = await gate.check_async(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
            invocation_count=0,
        )
        assert denial is None
        assert cost == 0  # free trial from static engine
