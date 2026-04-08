"""Tests for probe_relay_liveness, test_ws_connectivity, and resolve_relays."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from tollbooth.nostr_diagnostics import (
    probe_relay_liveness,
    resolve_relays,
    DEFAULT_RELAY,
    FALLBACK_RELAY_POOL,
)
from tollbooth import nostr_diagnostics


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

class TestResolveRelays:
    """Tests for resolve_relays()."""

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_configured_string_live(self, mock_cc):
        """Comma-separated string with live relays returns those relays."""
        live = {"wss://a.example": 0.01, "wss://b.example": 0.01}
        mock_cc.side_effect = _mock_create_connection_factory(live)

        result = resolve_relays("wss://a.example, wss://b.example")

        assert set(result) == {"wss://a.example", "wss://b.example"}

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_configured_list_live(self, mock_cc):
        """Explicit list with live relays returns those relays."""
        live = {"wss://x.example": 0.01}
        mock_cc.side_effect = _mock_create_connection_factory(live)

        result = resolve_relays(["wss://x.example"])

        assert result == ["wss://x.example"]

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_none_uses_default_relay(self, mock_cc):
        """None falls back to DEFAULT_RELAY."""
        live = {DEFAULT_RELAY: 0.01}
        mock_cc.side_effect = _mock_create_connection_factory(live)

        result = resolve_relays(None)

        assert result == [DEFAULT_RELAY]

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_empty_string_uses_default(self, mock_cc):
        """Empty string falls back to DEFAULT_RELAY."""
        live = {DEFAULT_RELAY: 0.01}
        mock_cc.side_effect = _mock_create_connection_factory(live)

        result = resolve_relays("")

        assert result == [DEFAULT_RELAY]

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_configured_dead_falls_back(self, mock_cc):
        """All configured relays dead -- falls back to FALLBACK_RELAY_POOL."""
        # Only fallback pool relays are live
        live = {url: 0.01 for url in FALLBACK_RELAY_POOL}
        mock_cc.side_effect = _mock_create_connection_factory(live)

        result = resolve_relays("wss://dead.example")

        assert set(result) == set(FALLBACK_RELAY_POOL)

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_everything_dead_returns_combined(self, mock_cc):
        """All relays dead -- returns configured + fallback pool."""
        mock_cc.side_effect = ConnectionRefusedError("nope")

        result = resolve_relays("wss://dead.example")

        assert result == ["wss://dead.example"] + FALLBACK_RELAY_POOL

    @patch("tollbooth.nostr_diagnostics.create_connection")
    def test_empty_list_uses_default(self, mock_cc):
        """Empty list falls back to DEFAULT_RELAY."""
        live = {DEFAULT_RELAY: 0.01}
        mock_cc.side_effect = _mock_create_connection_factory(live)

        result = resolve_relays([])

        assert result == [DEFAULT_RELAY]
