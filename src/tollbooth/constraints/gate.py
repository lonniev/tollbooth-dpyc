"""Constraint gate — walks a tool's per-tool constraint chain.

Each tool owns an ordered chain of ``PipelineStep`` objects stored on
its ``ToolPrice``.  The gate fetches the chain from the
:class:`PricingResolver` and walks it sequentially:

* Build a :class:`ConstraintContext` from the ledger / npub / env.
* For each step in order: instantiate its :class:`ToolConstraint`,
  apply patron-npub scoping, evaluate, short-circuit on denial,
  otherwise apply ``result.price_modifier`` to the running price.
* Return the (possibly-signed) effective price.  Negative results
  mean the chain has driven the toll into a credit — the caller
  (``OperatorRuntime.debit_or_deny``) handles that as a credit on
  the patron's balance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.config import ConfigError, load_constraint
from tollbooth.pricing_model import PipelineStep

logger = logging.getLogger(__name__)


class ConstraintGate:
    """Walks a per-tool constraint chain at debit / preview time.

    Stateless aside from the optional resolver attachment.  The
    resolver (a :class:`PricingResolver`) is the source of the
    cached active pricing model — the gate asks it for a tool's
    chain on every evaluation.
    """

    def __init__(self) -> None:
        self._resolver: Any | None = None  # PricingResolver (avoid circular import)

    def attach_resolver(self, resolver: Any) -> None:
        """Wire a :class:`PricingResolver` as the chain source."""
        self._resolver = resolver

    @property
    def enabled(self) -> bool:
        """True when a resolver is attached.  Without one, the gate is
        a no-op and tool calls pass through at their base price."""
        return self._resolver is not None

    # ---- evaluation ----

    def _build_context(
        self,
        *,
        tool_name: str,
        ledger: Any,
        npub: str,
        membership_tier: str = "default",
        invocation_count: int = 0,
        global_demand: dict[str, int] | None = None,
        dpop_token: str = "",
        coupon_redemptions: Any | None = None,
    ) -> ConstraintContext:
        return ConstraintContext(
            ledger=LedgerSnapshot(
                balance_api_sats=ledger.balance_api_sats,
                total_deposited_api_sats=ledger.total_deposited_api_sats,
                total_consumed_api_sats=ledger.total_consumed_api_sats,
                total_expired_api_sats=ledger.total_expired_api_sats,
            ),
            patron=PatronIdentity(
                npub=npub,
                membership_tier=membership_tier,
            ),
            env=EnvironmentSnapshot(
                utc_now=datetime.now(timezone.utc),
                tool_name=tool_name,
                invocation_count=invocation_count,
                global_demand=tuple((global_demand or {}).items()),
            ),
            dpop_token=dpop_token,
            coupon_redemptions=coupon_redemptions,
        )

    def _walk(
        self,
        chain: list[PipelineStep],
        base_cost: int,
        context: ConstraintContext,
    ) -> tuple[dict[str, Any] | None, int, list[str]]:
        """Walk *chain* against *context*, returning
        ``(denial, effective_cost, consume_coupon_ids)``.

        Effective cost is signed — negative values are credits that the
        caller routes to the patron's balance rather than debiting.

        ``consume_coupon_ids`` is the deduped list of coupon ids that
        applied during the walk (collected from each step's
        ``metadata["consume_coupon_id"]``).  Callers use this when
        ``consume=True`` to burn one use per id after the debit lands.
        """
        running = base_cost
        patron_npub = context.patron.npub
        consumed_ids: list[str] = []
        seen_ids: set[str] = set()

        for step in chain:
            # patron_npubs is a per-step audience filter — skip the
            # step if the patron isn't in its (non-empty) list.
            if step.patron_npubs and patron_npub not in step.patron_npubs:
                continue

            try:
                constraint = load_constraint({"type": step.type, **step.params})
            except ConfigError as exc:
                logger.warning(
                    "Skipping malformed constraint step %s (%s): %s",
                    step.id, step.type, exc,
                )
                continue

            result = constraint.evaluate(context)

            if not result.allowed:
                error: dict[str, Any] = {
                    "success": False,
                    "error": result.message or f"Constraint denied: {result.reason}",
                    "constraint_reason": result.reason,
                    "constraint_step_id": step.id,
                }
                if result.retry_after:
                    error["retry_after"] = result.retry_after.isoformat()
                if result.metadata:
                    error["constraint_metadata"] = result.metadata
                return error, 0, []

            if result.price_modifier is not None:
                running = result.price_modifier.apply_to(running)
                # Negative running price is allowed — the chain can drive
                # a debit into a credit.  The runtime handles the sign at
                # the balance-update step.

            cid = result.metadata.get("consume_coupon_id") if result.metadata else None
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                consumed_ids.append(cid)

        return None, running, consumed_ids

    @staticmethod
    def collect_coupon_ids(chain: list[PipelineStep]) -> list[str]:
        """Scan a chain for ``coupon`` steps and return their ``coupon_id``s.

        Used by the runtime to batch-fetch the caller's redemption rows
        before the walk so ``CouponConstraint.evaluate`` stays sync.
        """
        out: list[str] = []
        for step in chain:
            if step.type == "coupon":
                cid = step.params.get("coupon_id") if step.params else None
                if cid and isinstance(cid, str):
                    out.append(cid)
        return out

    async def evaluate_chain_async(
        self,
        *,
        tool_id: str,
        tool_name: str,
        base_cost: int,
        ledger: Any,
        npub: str = "",
        membership_tier: str = "default",
        invocation_count: int = 0,
        global_demand: dict[str, int] | None = None,
        dpop_token: str = "",
        coupon_redemptions: Any | None = None,
    ) -> tuple[dict[str, Any] | None, int, list[str]]:
        """Fetch *tool_id*'s chain from the resolver and walk it.

        Returns ``(None, base_cost, [])`` unchanged if no resolver is
        attached or the tool has an empty chain.
        """
        if self._resolver is None:
            return None, base_cost, []
        try:
            chain = await self._resolver.get_chain(tool_id)
        except Exception:
            logger.warning(
                "PricingResolver.get_chain(%s) failed; passing base cost through",
                tool_id, exc_info=True,
            )
            return None, base_cost, []
        if not chain:
            return None, base_cost, []

        context = self._build_context(
            tool_name=tool_name,
            ledger=ledger,
            npub=npub,
            membership_tier=membership_tier,
            invocation_count=invocation_count,
            global_demand=global_demand,
            dpop_token=dpop_token,
            coupon_redemptions=coupon_redemptions,
        )
        return self._walk(chain, base_cost, context)

    def evaluate_chain(
        self,
        *,
        chain: list[PipelineStep],
        tool_name: str,
        base_cost: int,
        ledger: Any,
        npub: str = "",
        membership_tier: str = "default",
        invocation_count: int = 0,
        global_demand: dict[str, int] | None = None,
        dpop_token: str = "",
        coupon_redemptions: Any | None = None,
    ) -> tuple[dict[str, Any] | None, int, list[str]]:
        """Sync chain walk.  Caller supplies the chain directly — used
        from ``check_price`` previews that have already resolved the
        cached model synchronously."""
        if not chain:
            return None, base_cost, []
        context = self._build_context(
            tool_name=tool_name,
            ledger=ledger,
            npub=npub,
            membership_tier=membership_tier,
            invocation_count=invocation_count,
            global_demand=global_demand,
            dpop_token=dpop_token,
            coupon_redemptions=coupon_redemptions,
        )
        return self._walk(chain, base_cost, context)
