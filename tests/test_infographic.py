"""Tests for tollbooth.infographic — SVG account statement rendering (M3.6)."""

from html import escape

from tollbooth.infographic import (
    THEME_BITCOIN_ORANGE,
    render_account_infographic,
)

MINIMAL = {
    "account_summary": {
        "balance_api_sats": 1234567,
        "total_deposited_api_sats": 2000000,
    },
    "generated_at": "2026-06-11T12:00:00Z",
}


def _svg(data=None, **kw):
    return render_account_infographic(data if data is not None else MINIMAL, **kw)


def test_returns_wellformed_svg_envelope():
    svg = _svg()
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg
    assert "viewBox=" in svg


def test_default_service_name_and_override():
    assert "Account" in _svg()              # default theme service_name
    assert "My Brain" in _svg(service_name="My Brain")


def test_balance_rendered_with_thousands_separator():
    assert "1,234,567" in _svg()


def test_service_name_is_escaped():
    svg = _svg(service_name="<script>alert(1)</script>")
    assert escape("<script>alert(1)</script>") in svg
    assert "<script>" not in svg            # raw payload never emitted


def test_empty_data_does_not_crash():
    svg = render_account_infographic({})
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")


def test_with_name_returns_independent_copy():
    named = THEME_BITCOIN_ORANGE.with_name("Renamed")
    assert named.service_name == "Renamed"
    assert THEME_BITCOIN_ORANGE.service_name == "Account"  # frozen original intact


def test_populated_tranches_and_usage_render():
    data = {
        "account_summary": {"balance_api_sats": 500, "total_deposited_api_sats": 1000},
        "active_tranches": [
            {"invoice_id": "inv-abc", "original_sats": 1000,
             "remaining_sats": 500, "expires_at": "2026-07-01T00:00:00Z"},
        ],
        "tool_usage_all_time": [{"tool_name": "brain_query", "calls": 42}],
        "generated_at": "2026-06-11T12:00:00Z",
    }
    svg = _svg(data)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "inv-abc" in svg                 # tranche source rendered
