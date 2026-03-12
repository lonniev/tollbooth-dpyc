"""Pricing model dataclasses for runtime-configurable tool pricing.

A PricingModel is a named bundle of per-tool prices and an optional
constraint pipeline that an operator can activate at runtime via the
Pricing Studio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolPrice:
    """Per-tool price entry within a pricing model."""

    tool_name: str
    price_sats: int
    category: str = ""
    intent: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "tool_name": self.tool_name,
            "price_sats": self.price_sats,
        }
        if self.category:
            d["category"] = self.category
        if self.intent:
            d["intent"] = self.intent
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolPrice:
        return cls(
            tool_name=data["tool_name"],
            price_sats=int(data["price_sats"]),
            category=data.get("category", ""),
            intent=data.get("intent", ""),
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
        """Return a flat {tool_name: price_sats} lookup dict."""
        return {tp.tool_name: tp.price_sats for tp in self.tools}

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
