"""Pricing model dataclasses for runtime-configurable tool pricing.

A PricingModel is a named bundle of per-tool prices.  Each tool owns
its own ordered constraint *chain* — a sequence of constraint instances
the gate walks at evaluation time, transforming the price step by step.
There is no operator-level constraint pipeline: chains are strictly
per-tool, and the same constraint applied to multiple tools is
authored once per tool (no sharing, no references).

Tools are identified by a stable UUID (``tool_id``) derived from their
canonical capability name.  The pricing model references UUIDs, not
MCP-specific tool names, enabling portability across implementations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tollbooth.pricing import ToolPricing


@dataclass
class ToolPrice:
    """Per-tool price entry within a pricing model.

    ``tool_id`` is the canonical UUID for the capability.  ``tool_name``
    is kept for display/debugging but is NOT the primary key.

    ``chain`` is this tool's ordered constraint chain.  The gate walks
    it left-to-right at evaluation time; each step receives the running
    price and returns a (possibly-transformed) price.  Empty chain =
    base price applies unchanged.
    """

    tool_id: str
    tool_name: str
    price_sats: int
    category: str = ""
    intent: str = ""
    priced: bool = False               # False = TBD, True = operator has set a price
    price_type: str = "flat"           # "flat" | "percent" | "formula"
    price_formula: str | None = None   # percent expression or formula string
    min_cost: int = 0                  # floor — minimum cost in sats
    max_cost: int | None = None        # ceiling — maximum cost in sats
    # Categorical multipliers — see ToolPricing for shape. Serialized as a
    # JSON object {param_name: {value: multiplier}}. Used for enum-keyed
    # surcharges like Optionality's difficulty × historicity table.
    multipliers: dict[str, dict[str, float]] | None = None
    # Ordered constraint chain.  Each step is a PipelineStep whose owning
    # tool is this ToolPrice (no need for the step to carry tool_ids).
    chain: list[PipelineStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "price_sats": self.price_sats,
            "category": self.category,
            "intent": self.intent,
            "priced": self.priced,
        }
        if self.price_type != "flat":
            d["price_type"] = self.price_type
        if self.price_formula is not None:
            d["price_formula"] = self.price_formula
        if self.min_cost > 0:
            d["min_cost"] = self.min_cost
        if self.max_cost is not None:
            d["max_cost"] = self.max_cost
        if self.multipliers:
            d["multipliers"] = self.multipliers
        if self.chain:
            d["chain"] = [step.to_dict() for step in self.chain]
        return d

    def to_tool_pricing(self) -> "ToolPricing":
        """Convert this declarative price entry to a runtime ToolPricing."""
        from tollbooth.pricing import ToolPricing

        # Multipliers normalize to the frozen-tuple shape ToolPricing wants.
        mults: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
        if self.multipliers:
            mults = tuple(
                (str(param), tuple((str(k), float(v)) for k, v in lookup.items()))
                for param, lookup in self.multipliers.items()
            )

        if self.price_type == "percent":
            return ToolPricing(
                rate_percent=float(self.price_sats),
                rate_param=self.price_formula or "",
                min_cost=self.min_cost,
                max_cost=self.max_cost,
                multipliers=mults,
            )
        return ToolPricing(
            fixed=self.price_sats,
            min_cost=self.min_cost,
            max_cost=self.max_cost,
            multipliers=mults,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolPrice:
        tool_id = data.get("tool_id")
        if not tool_id:
            raise KeyError(
                f"tool_id is required for tool '{data.get('tool_name', '?')}'. "
                "Run reset_pricing_model to re-seed from the tool registry."
            )
        raw_mults = data.get("multipliers")
        mults: dict[str, dict[str, float]] | None = None
        if isinstance(raw_mults, dict):
            mults = {
                str(p): {str(k): float(v) for k, v in lookup.items()}
                for p, lookup in raw_mults.items()
                if isinstance(lookup, dict)
            }
        raw_chain = data.get("chain", [])
        chain: list[PipelineStep] = (
            [PipelineStep.from_dict(s) for s in raw_chain]
            if isinstance(raw_chain, list)
            else []
        )
        return cls(
            tool_id=tool_id,
            tool_name=data["tool_name"],
            price_sats=int(data["price_sats"]),
            category=data.get("category", ""),
            intent=data.get("intent", ""),
            priced=bool(data.get("priced", True)),  # legacy models without field are considered priced
            price_type=data.get("price_type", "flat"),
            price_formula=data.get("price_formula", None),
            min_cost=int(data.get("min_cost", 0)),
            max_cost=int(data["max_cost"]) if data.get("max_cost") is not None else None,
            multipliers=mults,
            chain=chain,
        )


@dataclass
class TrancheLifetime:
    """How long a tranche of api_sats endures before expiring.

    This is a property of the money, not a per-tool constraint.
    ``None`` for ttl_days means credits never expire.
    """

    ttl_days: int | None = None
    target_usage_pct: float = 0.75
    min_days: int = 3
    max_days: int = 90

    def __post_init__(self) -> None:
        if self.min_days > self.max_days:
            raise ValueError(
                f"min_days ({self.min_days}) must be <= max_days ({self.max_days})"
            )
        if self.ttl_days is not None:
            self.ttl_days = max(self.min_days, min(self.max_days, self.ttl_days))

    @property
    def ttl_seconds(self) -> int | None:
        """Lifetime in seconds, or None if credits never expire."""
        return self.ttl_days * 86400 if self.ttl_days is not None else None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"target_usage_pct": self.target_usage_pct}
        if self.ttl_days is not None:
            d["ttl_days"] = self.ttl_days
        if self.min_days != 3:
            d["min_days"] = self.min_days
        if self.max_days != 90:
            d["max_days"] = self.max_days
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrancheLifetime:
        ttl_raw = data.get("ttl_days")
        return cls(
            ttl_days=int(ttl_raw) if ttl_raw is not None else None,
            target_usage_pct=float(data.get("target_usage_pct", 0.75)),
            min_days=int(data.get("min_days", 3)),
            max_days=int(data.get("max_days", 90)),
        )


@dataclass
class PipelineStep:
    """A single step in a tool's constraint chain.

    The owning :class:`ToolPrice` carries the chain, so the step is
    implicitly scoped to one tool.  ``patron_npubs`` remains a
    per-step filter (max 10 npubs) — narrowing a constraint to a
    specific audience within the tool that owns the chain.
    """

    id: str
    type: str  # must match a key in CONSTRAINT_REGISTRY
    params: dict[str, Any] = field(default_factory=dict)
    patron_npubs: list[str] = field(default_factory=list)

    _MAX_PATRON_GROUP = 10

    def __post_init__(self) -> None:
        if len(self.patron_npubs) > self._MAX_PATRON_GROUP:
            raise ValueError(
                f"patron_npubs exceeds max group size of {self._MAX_PATRON_GROUP}. "
                "Clone the constraint for additional patron groups."
            )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "type": self.type}
        if self.params:
            d["params"] = self.params
        if self.patron_npubs:
            d["patron_npubs"] = self.patron_npubs
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStep:
        return cls(
            id=data["id"],
            type=data["type"],
            params=dict(data.get("params", {})),
            patron_npubs=list(data.get("patron_npubs", [])),
        )


@dataclass
class PricingModel:
    """A named pricing model — per-tool prices with per-tool constraint chains."""

    model_id: str = ""
    operator: str = ""
    name: str = ""
    is_active: bool = False
    tools: list[ToolPrice] = field(default_factory=list)
    tranche_lifetime: TrancheLifetime | None = None

    def tool_cost_map(self) -> dict[str, int]:
        """Return a flat {tool_id: price_sats} lookup dict."""
        return {tp.tool_id: tp.price_sats for tp in self.tools}

    def tool_id_set(self) -> set[str]:
        """Return the set of tool_ids that have explicit entries."""
        return {tp.tool_id for tp in self.tools}

    def tool_priced_map(self) -> dict[str, bool]:
        """Return {tool_id: priced} — whether the operator has set a price."""
        return {tp.tool_id: tp.priced for tp in self.tools}

    def chain_for(self, tool_id: str) -> list[PipelineStep]:
        """Return this tool's constraint chain (empty list if no entry)."""
        for tp in self.tools:
            if tp.tool_id == tool_id:
                return tp.chain
        return []

    def to_json(self) -> str:
        """Serialize to a JSON string (for ``model_json`` column)."""
        return json.dumps(self._to_model_dict())

    def _to_model_dict(self) -> dict[str, Any]:
        """Internal dict for the ``model_json`` JSONB column."""
        d: dict[str, Any] = {
            "name": self.name,
            "tools": [tp.to_dict() for tp in self.tools],
        }
        if self.tranche_lifetime is not None:
            d["tranche_lifetime"] = self.tranche_lifetime.to_dict()
        return d

    @classmethod
    def from_json(cls, raw: str, *, model_id: str = "", operator: str = "", is_active: bool = False) -> PricingModel:
        """Deserialize from a JSON string (``model_json`` column).

        Any top-level ``pipeline`` key from the pre-0.40 shape is silently
        ignored.  Per-tool chains live inside each ``tools[].chain`` now.
        """
        data = json.loads(raw)

        tl_data = data.get("tranche_lifetime")
        tranche_lifetime = TrancheLifetime.from_dict(tl_data) if isinstance(tl_data, dict) else None

        return cls(
            model_id=model_id,
            operator=operator,
            name=data.get("name", ""),
            is_active=is_active,
            tools=[ToolPrice.from_dict(t) for t in data.get("tools", [])],
            tranche_lifetime=tranche_lifetime,
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PricingModel:
        """Build from a Neon result row (``operator_pricing_models`` table)."""
        model_json = row["model_json"]
        raw = model_json if isinstance(model_json, str) else json.dumps(model_json)
        return cls.from_json(
            raw,
            model_id=str(row["id"]),
            operator=row["operator"],
            is_active=bool(row.get("is_active", False)),
        )
