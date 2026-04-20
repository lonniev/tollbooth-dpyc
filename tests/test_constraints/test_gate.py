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


def _supply_config(max_invocations: int = 5) -> str:
    """Return a JSON config with a finite supply constraint on 'search'."""
    return json.dumps(
        {
            "tool_constraints": {
                "search": {
                    "constraints": [
                        {
                            "type": "finite_supply",
                            "max_invocations": max_invocations,
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
            constraints_config=_supply_config(max_invocations=5),
        )
        gate = ConstraintGate(config)

        denial, cost = gate.check(
            tool_name="search",
            base_cost=100,
            ledger=_StubLedger(),
            global_demand={"search:__total__": 5},
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


# ---------------------------------------------------------------------------
# End-to-end: FiniteSupply global scope with simulated vault loop
# ---------------------------------------------------------------------------


class TestFiniteSupplyE2E:
    """Simulate the runtime loop: fetch total -> gate.check -> increment -> repeat.

    With max_invocations=3 and 4 attempts, expect 3 allowed then 1 denied.
    """

    def test_three_allowed_then_denied(self):
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_supply_config(max_invocations=3),
        )
        gate = ConstraintGate(config)

        # Simulated vault: in-memory lifetime counter
        vault_totals: dict[str, int] = {}
        tool = "search"
        results: list[tuple[bool, int]] = []

        for _ in range(4):
            # --- fetch phase (mirrors runtime.get_global_demand) ---
            total = vault_totals.get(tool, 0)
            demand = {f"{tool}:__total__": total} if total else {}

            # --- gate phase ---
            denial, cost = gate.check(
                tool_name=tool,
                base_cost=100,
                ledger=_StubLedger(),
                global_demand=demand,
            )
            allowed = denial is None
            results.append((allowed, cost))

            # --- increment phase (mirrors fire_and_forget_supply_increment) ---
            if allowed:
                vault_totals[tool] = vault_totals.get(tool, 0) + 1

        # First 3 calls succeed at full price
        assert results[0] == (True, 100)
        assert results[1] == (True, 100)
        assert results[2] == (True, 100)

        # 4th call denied
        allowed, cost = results[3]
        assert allowed is False
        assert cost == 0

    def test_different_tools_have_independent_supply(self):
        """Each tool tracks its own lifetime counter independently."""
        config_json = json.dumps(
            {
                "tool_constraints": {
                    "alpha": {
                        "constraints": [
                            {"type": "finite_supply", "max_invocations": 2, "scope": "global"}
                        ]
                    },
                    "beta": {
                        "constraints": [
                            {"type": "finite_supply", "max_invocations": 1, "scope": "global"}
                        ]
                    },
                }
            }
        )
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=config_json,
        )
        gate = ConstraintGate(config)
        vault_totals: dict[str, int] = {}

        def call(tool: str) -> bool:
            total = vault_totals.get(tool, 0)
            demand = {f"{tool}:__total__": total} if total else {}
            denial, _ = gate.check(
                tool_name=tool,
                base_cost=50,
                ledger=_StubLedger(),
                global_demand=demand,
            )
            if denial is None:
                vault_totals[tool] = vault_totals.get(tool, 0) + 1
                return True
            return False

        # alpha: 2 allowed, then denied
        assert call("alpha") is True
        assert call("alpha") is True
        assert call("alpha") is False

        # beta: 1 allowed, then denied -- independent of alpha
        assert call("beta") is True
        assert call("beta") is False

    def test_removing_constraint_restores_access(self):
        """Operator removes finite_supply constraint; previously denied tool is accessible."""
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_supply_config(max_invocations=2),
        )
        gate = ConstraintGate(config)
        tool = "search"

        # Exhaust supply
        vault_total = 0
        for _ in range(2):
            demand = {f"{tool}:__total__": vault_total} if vault_total else {}
            denial, _ = gate.check(
                tool_name=tool, base_cost=100,
                ledger=_StubLedger(), global_demand=demand,
            )
            assert denial is None
            vault_total += 1

        # Confirm denied
        denial, cost = gate.check(
            tool_name=tool, base_cost=100,
            ledger=_StubLedger(),
            global_demand={f"{tool}:__total__": vault_total},
        )
        assert denial is not None
        assert denial["constraint_reason"] == "supply_exhausted"

        # Operator removes the constraint — no constraints on "search"
        new_config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=json.dumps({"tool_constraints": {}}),
        )
        gate = ConstraintGate(new_config)

        # Same tool, same exhausted vault total — now passes at base cost
        denial, cost = gate.check(
            tool_name=tool, base_cost=100,
            ledger=_StubLedger(),
            global_demand={f"{tool}:__total__": vault_total},
        )
        assert denial is None
        assert cost == 100

    def test_reapply_with_higher_cap_grants_new_runway(self):
        """Operator exhausts supply, then re-applies with a higher cap."""
        tool = "search"

        # Phase 1: cap at 2, exhaust it
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_supply_config(max_invocations=2),
        )
        gate = ConstraintGate(config)
        vault_total = 0

        for _ in range(2):
            demand = {f"{tool}:__total__": vault_total} if vault_total else {}
            denial, _ = gate.check(
                tool_name=tool, base_cost=100,
                ledger=_StubLedger(), global_demand=demand,
            )
            assert denial is None
            vault_total += 1

        # Confirm denied at cap
        denial, _ = gate.check(
            tool_name=tool, base_cost=100,
            ledger=_StubLedger(),
            global_demand={f"{tool}:__total__": vault_total},
        )
        assert denial is not None

        # Phase 2: operator raises cap to 5 — vault_total is still 2,
        # so 3 more calls should be allowed
        config2 = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_supply_config(max_invocations=5),
        )
        gate = ConstraintGate(config2)
        results: list[bool] = []

        for _ in range(4):
            demand = {f"{tool}:__total__": vault_total} if vault_total else {}
            denial, _ = gate.check(
                tool_name=tool, base_cost=100,
                ledger=_StubLedger(), global_demand=demand,
            )
            allowed = denial is None
            results.append(allowed)
            if allowed:
                vault_total += 1

        # 3 more allowed (totals 3, 4, 5), then denied at 5
        assert results == [True, True, True, False]
        assert vault_total == 5


# ---------------------------------------------------------------------------
# End-to-end: SurgePricing with simulated demand accumulation
# ---------------------------------------------------------------------------


def _surge_config(
    tool: str = "search",
    max_capacity: int = 10,
    tiers: list[dict] | None = None,
) -> str:
    """Return a JSON config with a surge pricing constraint."""
    if tiers is None:
        tiers = [
            {"capacity_pct": 0.5, "multiplier": 1.5},
            {"capacity_pct": 0.8, "multiplier": 2.0},
        ]
    return json.dumps(
        {
            "tool_constraints": {
                tool: {
                    "constraints": [
                        {
                            "type": "surge_pricing",
                            "max_capacity": max_capacity,
                            "tiers": tiers,
                        }
                    ]
                }
            }
        }
    )


class TestSurgePricingE2E:
    """Simulate the runtime loop: accumulate demand -> gate.check -> observe price changes."""

    def test_price_increases_with_demand(self):
        """As demand crosses tier thresholds, effective cost steps up."""
        tool = "search"
        base_cost = 100
        # max_capacity=10, tiers at 50% (1.5x) and 80% (2.0x)
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_surge_config(tool=tool, max_capacity=10),
        )
        gate = ConstraintGate(config)

        prices: list[int] = []
        for demand in range(11):
            demand_dict = {tool: demand} if demand else {}
            denial, cost = gate.check(
                tool_name=tool,
                base_cost=base_cost,
                ledger=_StubLedger(),
                global_demand=demand_dict,
            )
            assert denial is None  # surge never denies
            prices.append(cost)

        # demand 0-4 (utilization 0%-40%): below first tier -> base price
        assert prices[0] == 100
        assert prices[4] == 100

        # demand 5-7 (utilization 50%-70%): first tier -> 1.5x
        assert prices[5] == 150
        assert prices[7] == 150

        # demand 8-10 (utilization 80%-100%): second tier -> 2.0x
        assert prices[8] == 200
        assert prices[10] == 200

    def test_volume_discount_prices_decrease_with_demand(self):
        """Operator configures multipliers < 1.0: busier = cheaper."""
        tool = "search"
        base_cost = 200
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=_surge_config(
                tool=tool,
                max_capacity=20,
                tiers=[
                    {"capacity_pct": 0.25, "multiplier": 0.8},
                    {"capacity_pct": 0.5, "multiplier": 0.6},
                    {"capacity_pct": 0.75, "multiplier": 0.4},
                ],
            ),
        )
        gate = ConstraintGate(config)

        def price_at(demand: int) -> int:
            demand_dict = {tool: demand} if demand else {}
            denial, cost = gate.check(
                tool_name=tool,
                base_cost=base_cost,
                ledger=_StubLedger(),
                global_demand=demand_dict,
            )
            assert denial is None
            return cost

        # Low demand: base price
        assert price_at(0) == 200
        assert price_at(4) == 200

        # 25% utilization (demand=5): 0.8x -> 160
        assert price_at(5) == 160

        # 50% utilization (demand=10): 0.6x -> 120
        assert price_at(10) == 120

        # 75% utilization (demand=15): 0.4x -> 80
        assert price_at(15) == 80

        # Prices strictly decrease: busier = cheaper
        assert price_at(0) > price_at(5) > price_at(10) > price_at(15)

    def test_surge_and_supply_compose(self):
        """Surge pricing and finite supply on the same tool: price surges then supply exhausts."""
        tool = "search"
        base_cost = 100
        config = TollboothConfig(
            constraints_enabled=True,
            constraints_config=json.dumps(
                {
                    "tool_constraints": {
                        tool: {
                            "constraints": [
                                {
                                    "type": "surge_pricing",
                                    "max_capacity": 4,
                                    "tiers": [{"capacity_pct": 0.5, "multiplier": 2.0}],
                                },
                                {
                                    "type": "finite_supply",
                                    "max_invocations": 3,
                                    "scope": "global",
                                },
                            ]
                        }
                    }
                }
            ),
        )
        gate = ConstraintGate(config)
        vault_total = 0
        results: list[tuple[bool, int]] = []

        for demand in range(4):
            demand_dict: dict[str, int] = {}
            if demand:
                demand_dict[tool] = demand
            if vault_total:
                demand_dict[f"{tool}:__total__"] = vault_total

            denial, cost = gate.check(
                tool_name=tool,
                base_cost=base_cost,
                ledger=_StubLedger(),
                global_demand=demand_dict,
            )
            allowed = denial is None
            results.append((allowed, cost))
            if allowed:
                vault_total += 1

        # Call 0: demand=0 (below surge), supply 0/3 -> allowed at base
        assert results[0] == (True, 100)
        # Call 1: demand=1 (below 50% of 4), supply 1/3 -> allowed at base
        assert results[1] == (True, 100)
        # Call 2: demand=2 (50% of 4 -> surge 2x), supply 2/3 -> allowed at 200
        assert results[2] == (True, 200)
        # Call 3: demand=3 (surge 2x), supply 3/3 -> DENIED by supply
        allowed, cost = results[3]
        assert allowed is False
        assert cost == 0
