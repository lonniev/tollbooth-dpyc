"""Pricing model dataclasses for runtime-configurable tool pricing.

A PricingModel is a named bundle of per-tool prices and an optional
constraint pipeline that an operator can activate at runtime via the
Pricing Studio.

Tools are identified by a stable UUID (``tool_id``) derived from their
canonical capability name.  The pricing model references UUIDs, not
MCP-specific tool names, enabling portability across implementations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tollbooth.tool_identity import capability_uuid


@dataclass
class ToolPrice:
    """Per-tool price entry within a pricing model.

    ``tool_id`` is the canonical UUID for the capability.  ``tool_name``
    is kept for display/debugging but is NOT the primary key.
    """

    tool_id: str
    tool_name: str
    price_sats: int
    category: str = ""
    intent: str = ""
    price_type: str = "flat"           # "flat" | "percent" | "formula"
    price_formula: str | None = None   # percent expression or formula string
    min_cost: int = 0                  # floor — minimum cost in sats
    max_cost: int | None = None        # ceiling — maximum cost in sats

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "price_sats": self.price_sats,
            "category": self.category,
            "intent": self.intent,
        }
        if self.price_type != "flat":
            d["price_type"] = self.price_type
        if self.price_formula is not None:
            d["price_formula"] = self.price_formula
        if self.min_cost > 0:
            d["min_cost"] = self.min_cost
        if self.max_cost is not None:
            d["max_cost"] = self.max_cost
        return d

    def to_tool_pricing(self) -> "ToolPricing":
        """Convert this declarative price entry to a runtime ToolPricing."""
        from tollbooth.pricing import ToolPricing

        if self.price_type == "percent":
            return ToolPricing(
                rate_percent=float(self.price_sats),
                rate_param=self.price_formula or "",
                min_cost=self.min_cost,
                max_cost=self.max_cost,
            )
        return ToolPricing(
            fixed=self.price_sats,
            min_cost=self.min_cost,
            max_cost=self.max_cost,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolPrice:
        # Legacy migration: if tool_id is missing, synthesize from tool_name
        tool_id = data.get("tool_id")
        if not tool_id:
            tool_id = capability_uuid(data["tool_name"])
        return cls(
            tool_id=tool_id,
            tool_name=data["tool_name"],
            price_sats=int(data["price_sats"]),
            category=data.get("category", ""),
            intent=data.get("intent", ""),
            price_type=data.get("price_type", "flat"),
            price_formula=data.get("price_formula", None),
            min_cost=int(data.get("min_cost", 0)),
            max_cost=int(data["max_cost"]) if data.get("max_cost") is not None else None,
        )


@dataclass
class PipelineStep:
    """A single step in the constraint pipeline."""

    id: str
    type: str  # must match a key in CONSTRAINT_REGISTRY
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "type": self.type}
        if self.params:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStep:
        return cls(
            id=data["id"],
            type=data["type"],
            params=dict(data.get("params", {})),
        )


@dataclass
class PricingModel:
    """A named pricing model — tool costs + optional constraint pipeline."""

    model_id: str = ""
    operator: str = ""
    name: str = ""
    is_active: bool = False
    tools: list[ToolPrice] = field(default_factory=list)
    pipeline: list[PipelineStep] = field(default_factory=list)

    def tool_cost_map(self) -> dict[str, int]:
        """Return a flat {tool_id: price_sats} lookup dict."""
        return {tp.tool_id: tp.price_sats for tp in self.tools}

    def tool_id_set(self) -> set[str]:
        """Return the set of tool_ids that have explicit entries."""
        return {tp.tool_id for tp in self.tools}

    def to_constraint_config(self) -> dict[str, Any] | None:
        """Convert pipeline to the format ``load_constraints()`` expects.

        Returns ``None`` if the pipeline is empty.

        The pipeline steps are applied as wildcard (``"*"``) constraints
        so they affect all tools uniformly.  Each step's ``type`` and
        ``params`` are merged into a single constraint dict.
        """
        if not self.pipeline:
            return None

        constraints: list[dict[str, Any]] = []
        for step in self.pipeline:
            entry: dict[str, Any] = {"type": step.type}
            entry.update(step.params)
            constraints.append(entry)

        return {
            "tool_constraints": {
                "*": {
                    "constraints": constraints,
                },
            },
        }

    def to_json(self) -> str:
        """Serialize to a JSON string (for ``model_json`` column)."""
        return json.dumps(self._to_model_dict())

    def _to_model_dict(self) -> dict[str, Any]:
        """Internal dict for the ``model_json`` JSONB column."""
        return {
            "name": self.name,
            "tools": [tp.to_dict() for tp in self.tools],
            "pipeline": [ps.to_dict() for ps in self.pipeline],
        }

    @classmethod
    def from_json(cls, raw: str, *, model_id: str = "", operator: str = "", is_active: bool = False) -> PricingModel:
        """Deserialize from a JSON string (``model_json`` column)."""
        data = json.loads(raw)
        return cls(
            model_id=model_id,
            operator=operator,
            name=data.get("name", ""),
            is_active=is_active,
            tools=[ToolPrice.from_dict(t) for t in data.get("tools", [])],
            pipeline=[PipelineStep.from_dict(p) for p in data.get("pipeline", [])],
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
