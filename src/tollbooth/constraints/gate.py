"""Constraint gate — drop-in helper for operator _debit_or_error flows."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from tollbooth.config import TollboothConfig
from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.config import ConfigError, load_constraints
from tollbooth.constraints.engine import ConstraintEngine

logger = logging.getLogger(__name__)


class ConstraintGate:
    """Evaluates constraints before a tool call.

    Usage in operator code::

        gate = ConstraintGate(config)

        async def _debit_or_error(tool_name, user_id, cache):
            cost = TOOL_COSTS.get(tool_name, 0)
            if cost == 0:
                return None, cost

            ledger = await cache.get(user_id)

            # Check constraints (may modify cost)
            denial, effective_cost = gate.check(
                tool_name=tool_name,
                base_cost=cost,
                ledger=ledger,
                npub=user_id,
            )
            if denial is not None:
                return denial, 0

            # Debit at effective_cost (may be discounted)
            if not ledger.debit(tool_name, effective_cost):
                return {"success": False, "error": "Insufficient balance"}, 0

            cache.mark_dirty(user_id)
            return None, effective_cost
    """

    def __init__(self, config: TollboothConfig) -> None:
        self._enabled = config.constraints_enabled
        self._engine: ConstraintEngine | None = None

        if self._enabled and config.constraints_config:
            try:
                raw = json.loads(config.constraints_config)
                self._engine = load_constraints(raw)
            except (json.JSONDecodeError, ConfigError) as e:
                logger.error("Failed to load constraint config: %s", e)
                self._enabled = False

    @property
    def enabled(self) -> bool:
        """True when the gate is active and has a valid engine."""
        return self._enabled and self._engine is not None

    def check(
        self,
        tool_name: str,
        base_cost: int,
        ledger: Any,  # UserLedger — use Any to avoid circular import
        npub: str = "",
        membership_tier: str = "default",
        invocation_count: int = 0,
    ) -> tuple[dict[str, Any] | None, int]:
        """Evaluate constraints for a tool call.

        Returns
        -------
        (denial_dict, effective_cost)
            If *denial_dict* is not ``None``, the tool call should be blocked
            with that error.  Otherwise *effective_cost* is the (possibly
            discounted) cost to debit.
        """
        if not self.enabled:
            return None, base_cost

        assert self._engine is not None  # guarded by self.enabled

        # Build context from ledger snapshot (read-only)
        context = ConstraintContext(
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
            ),
        )

        result = self._engine.evaluate(tool_name, context)

        if not result.allowed:
            error: dict[str, Any] = {
                "success": False,
                "error": result.message or f"Constraint denied: {result.reason}",
                "constraint_reason": result.reason,
            }
            if result.retry_after:
                error["retry_after"] = result.retry_after.isoformat()
            if result.metadata:
                error["constraint_metadata"] = result.metadata
            return error, 0

        # Apply price modifier if present
        effective_cost = base_cost
        if result.price_modifier:
            effective_cost = result.price_modifier.apply_to(base_cost)

        return None, effective_cost
