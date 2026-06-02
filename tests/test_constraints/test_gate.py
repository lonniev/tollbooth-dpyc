"""Tests for ConstraintGate — per-tool chain walk."""

from __future__ import annotations

from dataclasses import dataclass

from tollbooth.constraints.gate import ConstraintGate
from tollbooth.pricing_model import PipelineStep


# Lightweight ledger stub — mirrors UserLedger's public attributes used by the gate.
@dataclass
class _StubLedger:
    balance_api_sats: int = 1000
    total_deposited_api_sats: int = 2000
    total_consumed_api_sats: int = 500
    total_expired_api_sats: int = 100


def _walk(chain: list[PipelineStep], base_cost: int, *, npub: str = "npub1abc") -> tuple[dict | None, int]:
    """Drive evaluate_chain directly (no resolver needed)."""
    return ConstraintGate().evaluate_chain(
        chain=chain,
        tool_name="search",
        base_cost=base_cost,
        ledger=_StubLedger(),
        npub=npub,
    )


class TestPassthrough:
    def test_empty_chain_returns_base_cost(self):
        denial, eff = _walk([], 100)
        assert denial is None
        assert eff == 100

    def test_no_resolver_async_passthrough(self):
        import asyncio
        gate = ConstraintGate()
        denial, eff = asyncio.run(
            gate.evaluate_chain_async(
                tool_id="t1",
                tool_name="search",
                base_cost=100,
                ledger=_StubLedger(),
                npub="npub1abc",
            )
        )
        assert denial is None
        assert eff == 100


class TestSequentialApplication:
    def test_two_discounts_compound_sequentially(self):
        # Stub ledger has total_consumed=500; threshold 0 always triggers.
        # First loyalty step: 50% off → 100 → 50.
        # Second loyalty step: 20% off → 50 → 40.
        chain = [
            PipelineStep(
                id="loyalty50",
                type="loyalty_discount",
                params={"threshold_consumed_api_sats": 0, "discount_percent": 50.0},
            ),
            PipelineStep(
                id="loyalty20",
                type="loyalty_discount",
                params={"threshold_consumed_api_sats": 0, "discount_percent": 20.0},
            ),
        ]
        denial, eff = _walk(chain, 100)
        assert denial is None
        assert eff == 40

    def test_unqualified_loyalty_step_does_not_change_price(self):
        # Threshold above consumed → no discount applies, base price passes.
        chain = [
            PipelineStep(
                id="loyalty_high",
                type="loyalty_discount",
                params={"threshold_consumed_api_sats": 10_000, "discount_percent": 75.0},
            ),
        ]
        denial, eff = _walk(chain, 100)
        assert denial is None
        assert eff == 100


class TestDenialShortCircuits:
    def test_deny_stops_chain(self):
        chain = [
            PipelineStep(
                id="exhausted",
                type="finite_supply",
                # Zero supply → always denies.
                params={"max_invocations": 0, "scope": "global"},
            ),
            PipelineStep(
                id="loyalty",
                type="loyalty_discount",
                params={"threshold_consumed_api_sats": 0, "discount_percent": 50.0},
            ),
        ]
        denial, eff = _walk(chain, 100)
        assert denial is not None
        assert eff == 0
        assert denial.get("constraint_step_id") == "exhausted"
        assert denial.get("constraint_reason") == "supply_exhausted"


class TestPatronScoping:
    def test_step_skipped_when_patron_not_in_list(self):
        chain = [
            PipelineStep(
                id="loyalty_alice_only",
                type="loyalty_discount",
                params={"threshold_consumed_api_sats": 0, "discount_percent": 50.0},
                patron_npubs=["npub1alice"],
            ),
        ]
        denial, eff = _walk(chain, 100, npub="npub1bob")
        assert denial is None
        assert eff == 100  # step skipped, base price passes through

    def test_step_applies_when_patron_in_list(self):
        chain = [
            PipelineStep(
                id="loyalty_alice_only",
                type="loyalty_discount",
                params={"threshold_consumed_api_sats": 0, "discount_percent": 50.0},
                patron_npubs=["npub1alice"],
            ),
        ]
        denial, eff = _walk(chain, 100, npub="npub1alice")
        assert denial is None
        assert eff == 50
