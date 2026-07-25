"""Tests for PricingModel, ToolPrice, and PipelineStep dataclasses."""

from __future__ import annotations

import json

import pytest

from tollbooth.pricing_model import PipelineStep, PricingModel, ToolPrice
from tollbooth.tool_identity import capability_uuid

# ---------------------------------------------------------------------------
# ToolPrice
# ---------------------------------------------------------------------------


class TestToolPrice:
    def test_round_trip(self) -> None:
        tp = ToolPrice(tool_id=capability_uuid("search"), tool_name="search", price_sats=5, category="read", intent="query")
        d = tp.to_dict()
        restored = ToolPrice.from_dict(d)
        assert restored.tool_id == tp.tool_id
        assert restored.tool_name == "search"
        assert restored.price_sats == 5
        assert restored.category == "read"
        assert restored.intent == "query"

    def test_always_emits_core_fields(self) -> None:
        tp = ToolPrice(tool_id=capability_uuid("search"), tool_name="search", price_sats=5)
        d = tp.to_dict()
        assert "tool_id" in d
        assert "category" in d  # always emitted, even when empty
        assert "intent" in d    # always emitted, even when empty

    def test_from_dict_defaults(self) -> None:
        tp = ToolPrice.from_dict({"tool_id": capability_uuid("x"), "tool_name": "x", "price_sats": 1})
        assert tp.category == ""
        assert tp.intent == ""

    def test_from_dict_rejects_missing_tool_id(self) -> None:
        """Data without tool_id raises KeyError — no legacy synthesis."""
        with pytest.raises(KeyError, match="tool_id is required"):
            ToolPrice.from_dict({"tool_name": "search", "price_sats": 5})

    def test_from_dict_preserves_explicit_tool_id(self) -> None:
        """Explicit tool_id in data is preserved, not overwritten."""
        explicit_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        tp = ToolPrice.from_dict({"tool_id": explicit_id, "tool_name": "search", "price_sats": 5})
        assert tp.tool_id == explicit_id

    def test_to_tool_pricing_flat(self) -> None:
        tp = ToolPrice(tool_id=capability_uuid("search"), tool_name="search", price_sats=5, min_cost=1, max_cost=100)
        pricing = tp.to_tool_pricing()
        assert pricing.fixed == 5
        assert pricing.rate_percent == 0.0
        assert pricing.min_cost == 1
        assert pricing.max_cost == 100
        assert pricing.compute() == 5

    def test_to_tool_pricing_percent(self) -> None:
        tp = ToolPrice(
            tool_id=capability_uuid("certify_credits"),
            tool_name="certify_credits", price_sats=2,
            price_type="percent", price_formula="amount_sats", min_cost=10,
        )
        pricing = tp.to_tool_pricing()
        assert pricing.rate_percent == 2.0
        assert pricing.rate_param == "amount_sats"
        assert pricing.min_cost == 10
        assert pricing.fixed == 0
        # 2% of 1000 = 20
        assert pricing.compute(amount_sats=1000) == 20
        # 2% of 100 = 2, but min_cost = 10
        assert pricing.compute(amount_sats=100) == 10


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------


class TestPipelineStep:
    def test_round_trip(self) -> None:
        ps = PipelineStep(id="s1", type="free_trial", params={"first_n_free": 3})
        d = ps.to_dict()
        restored = PipelineStep.from_dict(d)
        assert restored.id == "s1"
        assert restored.type == "free_trial"
        assert restored.params == {"first_n_free": 3}

    def test_empty_params_omitted(self) -> None:
        ps = PipelineStep(id="s1", type="coupon")
        d = ps.to_dict()
        assert "params" not in d

    def test_from_dict_defaults(self) -> None:
        ps = PipelineStep.from_dict({"id": "s1", "type": "coupon"})
        assert ps.params == {}


# ---------------------------------------------------------------------------
# PricingModel
# ---------------------------------------------------------------------------


class TestPricingModel:
    def _sample_model(self) -> PricingModel:
        return PricingModel(
            model_id="abc-123",
            operator="npub1test",
            name="Launch Pricing",
            is_active=True,
            tools=[
                ToolPrice(
                    tool_id=capability_uuid("search"),
                    tool_name="search",
                    price_sats=5,
                    chain=[
                        PipelineStep(id="s1", type="free_trial", params={"first_n_free": 3}),
                    ],
                ),
                ToolPrice(tool_id=capability_uuid("create"), tool_name="create", price_sats=10),
            ],
        )

    def test_tool_cost_map(self) -> None:
        model = self._sample_model()
        costs = model.tool_cost_map()
        # Keyed by tool_id (UUID), not tool_name
        search_id = capability_uuid("search")
        create_id = capability_uuid("create")
        assert costs == {search_id: 5, create_id: 10}

    def test_tool_id_set(self) -> None:
        model = self._sample_model()
        ids = model.tool_id_set()
        assert capability_uuid("search") in ids
        assert capability_uuid("create") in ids

    def test_tool_cost_map_empty(self) -> None:
        model = PricingModel()
        assert model.tool_cost_map() == {}

    def test_chain_for_returns_owning_tool_chain(self) -> None:
        model = self._sample_model()
        chain = model.chain_for(capability_uuid("search"))
        assert len(chain) == 1
        assert chain[0].type == "free_trial"
        # Tools without an explicit chain return empty list, not error.
        assert model.chain_for(capability_uuid("create")) == []
        assert model.chain_for("unknown-id") == []

    def test_json_round_trip(self) -> None:
        model = self._sample_model()
        raw = model.to_json()
        restored = PricingModel.from_json(
            raw,
            model_id="abc-123",
            operator="npub1test",
            is_active=True,
        )
        assert restored.name == "Launch Pricing"
        assert len(restored.tools) == 2
        assert restored.tools[0].tool_name == "search"
        assert restored.tools[0].tool_id == capability_uuid("search")
        # Chain rides along on its owning tool.
        assert len(restored.tools[0].chain) == 1
        assert restored.tools[0].chain[0].type == "free_trial"
        assert restored.tools[1].chain == []

    def test_from_json_silently_drops_old_pipeline_key(self) -> None:
        """Pre-0.40 models had a top-level ``pipeline`` array — ignored on read."""
        raw = json.dumps({
            "name": "old",
            "tools": [{"tool_id": capability_uuid("x"), "tool_name": "x", "price_sats": 1}],
            "pipeline": [{"id": "stray", "type": "happy_hour"}],
        })
        restored = PricingModel.from_json(raw)
        assert len(restored.tools) == 1
        assert restored.tools[0].chain == []
        # No attribute leak — PricingModel has no .pipeline.
        assert not hasattr(restored, "pipeline")

    def test_from_row_with_string_model_json(self) -> None:
        row = {
            "id": "uuid-1",
            "operator": "npub1op",
            "name": "Test",
            "model_json": json.dumps({
                "name": "Test",
                "tools": [{"tool_id": capability_uuid("a"), "tool_name": "a", "price_sats": 1}],
            }),
            "is_active": True,
        }
        model = PricingModel.from_row(row)
        assert model.model_id == "uuid-1"
        assert model.operator == "npub1op"
        assert model.is_active is True
        assert len(model.tools) == 1
        assert model.tools[0].tool_id == capability_uuid("a")

    def test_from_row_rejects_missing_tool_id(self) -> None:
        """Rows with tools missing tool_id raise KeyError."""
        row = {
            "id": "uuid-1",
            "operator": "npub1op",
            "name": "Test",
            "model_json": json.dumps({
                "name": "Test",
                "tools": [{"tool_name": "a", "price_sats": 1}],
            }),
            "is_active": True,
        }
        with pytest.raises(KeyError, match="tool_id is required"):
            PricingModel.from_row(row)

    def test_from_row_with_dict_model_json(self) -> None:
        """Neon may return JSONB as a dict instead of string."""
        row = {
            "id": "uuid-2",
            "operator": "npub1op",
            "name": "Test",
            "model_json": {
                "name": "Test",
                "tools": [],
            },
            "is_active": False,
        }
        model = PricingModel.from_row(row)
        assert model.model_id == "uuid-2"
        assert model.is_active is False
