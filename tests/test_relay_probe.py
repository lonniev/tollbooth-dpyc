"""Tests for probe_relay_liveness, test_ws_connectivity, and resolve_relays."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tollbooth import nostr_diagnostics
from tollbooth.nostr_diagnostics import (
    probe_relay_liveness,
    resolve_relays,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_create_connection_factory(live_relays: dict[str, float]):
    """Return a mock create_connection that succeeds for relays in *live_relays*.

    *live_relays* maps relay URL → simulated latency in seconds.
    Unlisted relays raise ConnectionRefusedError.
    """

    def _mock_create(url, timeout=10):
        if url in live_relays:
            ws = MagicMock()
            ws.close = MagicMock()
            return ws
        raise ConnectionRefusedError(f"Connection refused: {url}")

    return _mock_create


# ---------------------------------------------------------------------------
# TestProbeRelayLiveness
# ---------------------------------------------------------------------------

class TestProbeRelayLiveness:
    """Tests for probe_relay_liveness()."""

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_all_reachable_sorted_by_latency(self, mock_cc):
        """All relays connect — results sorted by latency (lowest first)."""
        relays = ["wss://a.example", "wss://b.example", "wss://c.example"]
        mock_cc.side_effect = _mock_create_connection_factory({
            "wss://a.example": 0.01,
            "wss://b.example": 0.05,
            "wss://c.example": 0.10,
        })

        results = probe_relay_liveness(relays, timeout=3)

        assert len(results) == 3
        for r in results:
            assert r["connected"] is True
            assert r["error"] is None
            assert r["latency_ms"] is not None
        # All connected — sorted by latency ascending
        latencies = [r["latency_ms"] for r in results]
        assert latencies == sorted(latencies)

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_some_down_connected_first(self, mock_cc):
        """Mixed liveness — connected relays appear before disconnected."""
        live = {"wss://good.example": 0.005}
        mock_cc.side_effect = _mock_create_connection_factory(live)

        relays = ["wss://bad.example", "wss://good.example"]
        results = probe_relay_liveness(relays, timeout=2)

        assert len(results) == 2
        assert results[0]["connected"] is True
        assert results[0]["relay"] == "wss://good.example"
        assert results[1]["connected"] is False
        assert results[1]["relay"] == "wss://bad.example"

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_all_down(self, mock_cc):
        """All relays down — every result has connected=False."""
        mock_cc.side_effect = ConnectionRefusedError("nope")

        relays = ["wss://a.example", "wss://b.example"]
        results = probe_relay_liveness(relays, timeout=2)

        assert len(results) == 2
        for r in results:
            assert r["connected"] is False
            assert r["error"] is not None

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_custom_timeout_passed_through(self, mock_cc):
        """Custom timeout is forwarded to create_connection."""
        mock_cc.side_effect = _mock_create_connection_factory({"wss://r.example": 0.001})

        probe_relay_liveness(["wss://r.example"], timeout=7)

        mock_cc.assert_called_once_with("wss://r.example", timeout=7)

    def test_empty_list(self):
        """Empty relay list returns empty results."""
        results = probe_relay_liveness([], timeout=1)
        assert results == []


# ---------------------------------------------------------------------------
# TestWsConnectivityPublic
# ---------------------------------------------------------------------------

class TestWsConnectivityPublic:
    """Verify test_ws_connectivity is importable and functional."""

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_importable_and_returns_dict(self, mock_cc):
        """test_ws_connectivity is a public function returning expected keys."""
        ws = MagicMock()
        mock_cc.return_value = ws

        result = nostr_diagnostics.test_ws_connectivity("wss://test.example")

        assert isinstance(result, dict)
        assert result["relay"] == "wss://test.example"
        assert result["connected"] is True
        assert result["latency_ms"] is not None
        assert result["error"] is None

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_custom_timeout(self, mock_cc):
        """Custom timeout parameter is honoured."""
        ws = MagicMock()
        mock_cc.return_value = ws

        nostr_diagnostics.test_ws_connectivity("wss://test.example", timeout=3)

        mock_cc.assert_called_once_with("wss://test.example", timeout=3)

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_connection_failure(self, mock_cc):
        """Connection failure returns connected=False with error message."""
        mock_cc.side_effect = ConnectionRefusedError("refused")

        result = nostr_diagnostics.test_ws_connectivity("wss://down.example")

        assert result["connected"] is False
        assert "refused" in result["error"]
        assert result["relay"] == "wss://down.example"


# ---------------------------------------------------------------------------
# TestResolveRelays
# ---------------------------------------------------------------------------

_REGISTRY_SET = [
    "wss://relay.primal.net",
    "wss://relay.damus.io",
    "wss://nos.lol",
    "wss://relay.nostr.band",
]


class TestResolveRelays:
    """Tests for resolve_relays() — the relay set now comes solely from the
    DPYC community registry (relay_registry.get_relays)."""

    @patch("tollbooth.relay_registry.get_relays", return_value=list(_REGISTRY_SET))
    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_returns_live_registry_relays_in_order(self, mock_cc, _mock_get):
        """Live relays are returned in registry (primary-first) order."""
        live = {_REGISTRY_SET[0]: 0.01, _REGISTRY_SET[2]: 0.02}
        mock_cc.side_effect = _mock_create_connection_factory(live)

        result = resolve_relays()

        # Registry order preserved, dead relays dropped.
        assert result == [_REGISTRY_SET[0], _REGISTRY_SET[2]]

    @patch("tollbooth.relay_registry.get_relays", return_value=list(_REGISTRY_SET))
    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_all_dead_returns_full_registry_set(self, mock_cc, _mock_get):
        """When no relay responds, the full registry set is returned unprobed."""
        mock_cc.side_effect = ConnectionRefusedError("nope")

        result = resolve_relays()

        assert result == _REGISTRY_SET

    @patch("tollbooth.relay_registry.get_relays", return_value=list(_REGISTRY_SET))
    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_all_live_preserves_registry_order(self, mock_cc, _mock_get):
        """All relays live → returned in registry order (primary first)."""
        live = {url: 0.01 for url in _REGISTRY_SET}
        mock_cc.side_effect = _mock_create_connection_factory(live)

        result = resolve_relays()

        assert result == _REGISTRY_SET
