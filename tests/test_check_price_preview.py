"""Unit tests for the pure check_price helpers (audit M2.1b).

build_pricing_preview / apply_constraint_preview were extracted out of the
check_price closure so the percent/flat/multiplier branching and the
denied/discount/credit effect formatting are testable without a runtime,
pricing resolver, or constraint gate.
"""

from __future__ import annotations

from types import SimpleNamespace

from tollbooth.tools.pricing import apply_constraint_preview, build_pricing_preview


def _pricing(*, rate_percent=0.0, rate_param="amount_sats", min_cost=0,
             multipliers=None, compute=None):
    return SimpleNamespace(
        rate_percent=rate_percent,
        rate_param=rate_param,
        min_cost=min_cost,
        multipliers=multipliers,
        compute=compute or (lambda **kw: 100),
    )


# ── build_pricing_preview ─────────────────────────────────────────────

def test_percent_with_kwargs_computes_cost():
    p = _pricing(rate_percent=2.0, rate_param="amount_sats", min_cost=10,
                 compute=lambda **kw: kw["amount_sats"] // 50)
    r = build_pricing_preview("tid", "svc_tool", p, {"amount_sats": 5000})
    assert r["pricing_type"] == "percent"
    assert r["rate_percent"] == 2.0 and r["rate_param"] == "amount_sats"
    assert r["min_cost_sats"] == 10
    assert r["base_cost_api_sats"] == 100
    assert r["effective_cost_api_sats"] == 100
    assert "hint" not in r
    # skeleton fields
    assert r["success"] is True and r["tool_id"] == "tid" and r["tool_name"] == "svc_tool"
    assert r["constraints_enabled"] is False and r["constraint_effects"] == []


def test_percent_without_kwargs_returns_hint_and_null_costs():
    p = _pricing(rate_percent=2.0, rate_param="notional")
    r = build_pricing_preview("tid", "t", p, {})
    assert r["base_cost_api_sats"] is None
    assert r["effective_cost_api_sats"] is None
    assert "notional" in r["hint"]


def test_flat_pricing():
    p = _pricing(rate_percent=0.0, compute=lambda **kw: 42)
    r = build_pricing_preview("tid", "t", p, {})
    assert r["pricing_type"] == "flat"
    assert r["base_cost_api_sats"] == 42 and r["effective_cost_api_sats"] == 42
    assert "multipliers" not in r


def test_flat_with_multipliers_exposes_table():
    # multipliers as iterable of (param, lookup) where lookup is (k, v) pairs
    mult = [("difficulty", [("easy", 1.0), ("sovereign", 3.0)])]
    p = _pricing(rate_percent=0.0, multipliers=mult, compute=lambda **kw: 50)
    r = build_pricing_preview("tid", "t", p, {"difficulty": "sovereign"})
    assert r["pricing_type"] == "flat+multipliers"
    assert r["base_cost_api_sats"] == 50
    assert r["multipliers"] == {"difficulty": {"easy": 1.0, "sovereign": 3.0}}


# ── apply_constraint_preview ──────────────────────────────────────────

def _result(base=100):
    return {
        "success": True, "tool_id": "t", "tool_name": "t",
        "constraints_enabled": True, "constraint_effects": [],
        "base_cost_api_sats": base, "effective_cost_api_sats": base,
    }


def test_denial_zeroes_cost_and_records_reason():
    r = _result()
    apply_constraint_preview(r, 100, 100, {"constraint_reason": "rate_limited"}, 0)
    assert r["effective_cost_api_sats"] == 0
    assert r["constraint_effects"] == [{"type": "denied", "reason": "rate_limited"}]


def test_denial_without_reason_defaults_blocked():
    r = _result()
    # Truthy denial lacking constraint_reason → "blocked" default. (An empty
    # dict is falsy and is treated as "no denial" — see test below.)
    apply_constraint_preview(r, 100, 100, {"denied": True}, 0)
    assert r["constraint_effects"][0] == {"type": "denied", "reason": "blocked"}


def test_empty_denial_dict_is_not_a_denial():
    r = _result()
    apply_constraint_preview(r, 100, 80, {}, 0)
    # Falsy denial → else branch: discount recorded, not a denial.
    assert r["constraint_effects"] == [{"type": "discount", "from": 100, "to": 80}]


def test_discount_effect():
    r = _result()
    apply_constraint_preview(r, 100, 60, None, 0)
    assert r["effective_cost_api_sats"] == 60
    assert r["constraint_effects"] == [{"type": "discount", "from": 100, "to": 60}]


def test_credit_effect_when_effective_negative():
    r = _result()
    apply_constraint_preview(r, 100, -25, None, 0)
    assert r["constraint_effects"] == [{"type": "credit", "from": 100, "to": -25}]


def test_no_effect_when_unchanged():
    r = _result()
    apply_constraint_preview(r, 100, 100, None, 0)
    assert r["effective_cost_api_sats"] == 100
    assert r["constraint_effects"] == []


def test_demand_recorded_only_when_positive():
    r = _result()
    apply_constraint_preview(r, 100, 100, None, 7)
    assert r["current_demand"] == 7

    r2 = _result()
    apply_constraint_preview(r2, 100, 100, None, 0)
    assert "current_demand" not in r2
