"""Tests for PricingModel, ToolPrice, and PipelineStep dataclasses."""

from __future__ import annotations

import json


from tollbooth.pricing_model import PipelineStep, PricingModel, ToolPrice


# ---------------------------------------------------------------------------
# ToolPrice
# ---------------------------------------------------------------------------


class TestToolPrice:
    def test_round_trip(self) -> None:
        tp = ToolPrice(tool_name="search", price_sats=5, category="read", intent="query")
        d = tp.to_dict()
        restored = ToolPrice.from_dict(d)
        assert restored.tool_name == "search"
        assert restored.price_sats == 5
        assert restored.category == "read"
        assert restored.intent == "query"

    def test_optional_fields_omitted(self) -> None:
        tp = ToolPrice(tool_name="search", price_sats=5)
        d = tp.to_dict()
        assert "category" not in d
        assert "intent" not in d

    def test_from_dict_defaults(self) -> None:
        tp = ToolPrice.from_dict({"tool_name": "x", "price_sats": 1})
        assert tp.category == ""
        assert tp.intent == ""

    def test_to_tool_pricing_flat(self) -> None:
        tp = ToolPrice(tool_name="search", price_sats=5, min_cost=1, max_cost=100)
        pricing = tp.to_tool_pricing()
        assert pricing.fixed == 5
        assert pricing.rate_percent == 0.0
        assert pricing.min_cost == 1
        assert pricing.max_cost == 100
        assert pricing.compute() == 5

    def test_to_tool_pricing_percent(self) -> None:
        tp = ToolPrice(
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
                ToolPrice(tool_name="search", price_sats=5),
                ToolPrice(tool_name="create", price_sats=10),
            ],
            pipeline=[
                PipelineStep(id="s1", type="free_trial", params={"first_n_free": 3}),
            ],
        )

    def test_tool_cost_map(self) -> None:
        model = self._sample_model()
        costs = model.tool_cost_map()
        assert costs == {"search": 5, "create": 10}

    def test_tool_cost_map_empty(self) -> None:
        model = PricingModel()
        assert model.tool_cost_map() == {}

    def test_to_constraint_config(self) -> None:
        model = self._sample_model()
        cfg = model.to_constraint_config()
        assert cfg is not None
        assert "*" in cfg["tool_constraints"]
        constraints = cfg["tool_constraints"]["*"]["constraints"]
        assert len(constraints) == 1
        assert constraints[0]["type"] == "free_trial"
        assert constraints[0]["first_n_free"] == 3

    def test_to_constraint_config_empty_pipeline(self) -> None:
        model = PricingModel()
        assert model.to_constraint_config() is None

    def test_to_constraint_config_feeds_load_constraints(self) -> None:
        """Verify the output can be consumed by load_constraints()."""
        from tollbooth.constraints.config import load_constraints

        model = self._sample_model()
        cfg = model.to_constraint_config()
        assert cfg is not None
        engine = load_constraints(cfg)
        assert engine is not None

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
        assert len(restored.pipeline) == 1
        assert restored.pipeline[0].type == "free_trial"

    def test_from_row_with_string_model_json(self) -> None:
        row = {
            "id": "uuid-1",
            "operator": "npub1op",
            "name": "Test",
            "model_json": json.dumps({
                "name": "Test",
                "tools": [{"tool_name": "a", "price_sats": 1}],
                "pipeline": [],
            }),
            "is_active": True,
        }
        model = PricingModel.from_row(row)
        assert model.model_id == "uuid-1"
        assert model.operator == "npub1op"
        assert model.is_active is True
        assert len(model.tools) == 1

    def test_from_row_with_dict_model_json(self) -> None:
        """Neon may return JSONB as a dict instead of string."""
        row = {
            "id": "uuid-2",
            "operator": "npub1op",
            "name": "Test",
            "model_json": {
                "name": "Test",
                "tools": [],
                "pipeline": [],
            },
            "is_active": False,
        }
        model = PricingModel.from_row(row)
        assert model.model_id == "uuid-2"
        assert model.is_active is False
