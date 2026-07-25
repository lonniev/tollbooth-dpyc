"""Tests for Nostr Relay Health Diagnostics (courier_health / courier_ping)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.nostr_diagnostics import (
    _KIND_ENCRYPTED_DM,
    _KIND_GIFT_WRAP,
    _extract_nip_support,
    _fetch_nip11,
    _self_dm_roundtrip,
    _test_subscription_filter,
    _test_ws_connectivity,
    courier_health,
    courier_ping,
)

# ── Helpers ─────────────────────────────────────────────────────────────

def _operator_keys():
    """Generate a fresh operator keypair."""
    pk = PrivateKey()
    return pk.nsec, pk.hex(), pk.public_key.hex(), pk.public_key.bech32()


def _make_mock_ws(recv_messages: list[str] | None = None, *, raise_on_recv: Exception | None = None):
    """Build a mock WebSocket that returns the given recv messages in order."""
    ws = MagicMock()
    if raise_on_recv is not None:
        ws.recv = MagicMock(side_effect=raise_on_recv)
    elif recv_messages is not None:
        ws.recv = MagicMock(side_effect=recv_messages)
    else:
        ws.recv = MagicMock(side_effect=TimeoutError("no data"))
    return ws


# ── NIP-11 Parsing Tests ───────────────────────────────────────────────

class TestNip11Fetch:
    """Tests for NIP-11 information document fetching."""

    @pytest.mark.asyncio
    async def test_successful_nip11(self):
        """Valid NIP-11 response is parsed correctly."""
        nip11_doc = {
            "name": "Test Relay",
            "description": "A test relay",
            "supported_nips": [1, 4, 11, 17, 44],
            "software": "test-relay",
            "version": "1.0.0",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = nip11_doc

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_nip11("wss://relay.test.com")

        assert result["name"] == "Test Relay"
        assert 4 in result["supported_nips"]
        assert 44 in result["supported_nips"]

    @pytest.mark.asyncio
    async def test_nip11_converts_wss_to_https(self):
        """wss:// URL is converted to https:// for NIP-11 fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "relay"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_client):
            await _fetch_nip11("wss://relay.test.com")

        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://relay.test.com"
        assert call_args[1]["headers"]["Accept"] == "application/nostr+json"

    @pytest.mark.asyncio
    async def test_nip11_converts_ws_to_http(self):
        """ws:// URL is converted to http:// for NIP-11 fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "relay"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_client):
            await _fetch_nip11("ws://localhost:7777")

        call_args = mock_client.get.call_args
        assert call_args[0][0] == "http://localhost:7777"

    @pytest.mark.asyncio
    async def test_nip11_http_error(self):
        """Non-200 response returns error dict."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_nip11("wss://relay.test.com")

        assert "error" in result
        assert "404" in result["error"]

    @pytest.mark.asyncio
    async def test_nip11_timeout(self):
        """Timeout returns error dict."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_nip11("wss://relay.test.com")

        assert result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_nip11_connect_error(self):
        """Connection error returns error dict."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_nip11("wss://relay.test.com")

        assert "error" in result
        assert "connect_error" in result["error"]

    @pytest.mark.asyncio
    async def test_nip11_invalid_json(self):
        """Invalid JSON body returns error dict."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("bad", "", 0)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_nip11("wss://relay.test.com")

        assert result["error"] == "invalid_json"


# ── NIP Capability Detection Tests ──────────────────────────────────────

class TestNipSupport:
    """Tests for NIP capability extraction from NIP-11 documents."""

    def test_all_nips_present(self):
        """Extracts NIP-04, NIP-17, NIP-44 when all present."""
        nip11 = {"supported_nips": [1, 4, 11, 17, 44, 50]}
        result = _extract_nip_support(nip11)
        assert result == {"nip04": True, "nip17": True, "nip44": True}

    def test_only_nip04(self):
        """Only NIP-04 detected."""
        nip11 = {"supported_nips": [1, 4, 11]}
        result = _extract_nip_support(nip11)
        assert result == {"nip04": True, "nip17": False, "nip44": False}

    def test_no_supported_nips(self):
        """Empty supported_nips list."""
        nip11 = {"supported_nips": []}
        result = _extract_nip_support(nip11)
        assert result == {"nip04": False, "nip17": False, "nip44": False}

    def test_missing_supported_nips_key(self):
        """NIP-11 document without supported_nips key."""
        nip11 = {"name": "relay", "description": "no nips listed"}
        result = _extract_nip_support(nip11)
        assert result == {"nip04": False, "nip17": False, "nip44": False}

    def test_string_nip_values(self):
        """NIP values as strings (some relays do this)."""
        nip11 = {"supported_nips": ["4", "17", "44"]}
        result = _extract_nip_support(nip11)
        assert result == {"nip04": True, "nip17": True, "nip44": True}

    def test_mixed_types(self):
        """Mixed int and string NIP values."""
        nip11 = {"supported_nips": [4, "17", 44, "invalid"]}
        result = _extract_nip_support(nip11)
        assert result == {"nip04": True, "nip17": True, "nip44": True}


# ── WebSocket Connectivity Tests ────────────────────────────────────────

class TestWsConnectivity:
    """Tests for WebSocket connectivity probing."""

    def test_successful_connection(self):
        """Connected relay returns success with latency."""
        mock_ws = MagicMock()

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = _test_ws_connectivity("wss://relay.test.com")

        assert result["connected"] is True
        assert result["latency_ms"] is not None
        assert result["latency_ms"] >= 0
        assert result["error"] is None

    def test_connection_refused(self):
        """Unreachable relay returns failure."""
        with patch(
            "tollbooth.nostr_diagnostics.create_connection",
            side_effect=ConnectionRefusedError("refused"),
        ):
            result = _test_ws_connectivity("wss://dead.relay.com")

        assert result["connected"] is False
        assert result["error"] is not None
        assert result["latency_ms"] is not None  # still measured

    def test_connection_timeout(self):
        """Timeout returns failure with timing."""
        with patch(
            "tollbooth.nostr_diagnostics.create_connection",
            side_effect=TimeoutError("timeout"),
        ):
            result = _test_ws_connectivity("wss://slow.relay.com")

        assert result["connected"] is False
        assert "timeout" in result["error"].lower()

    @patch("tollbooth.nostr_diagnostics._HAS_WEBSOCKET", False)
    def test_missing_websocket_lib(self):
        """Missing websocket-client returns descriptive error."""
        result = _test_ws_connectivity("wss://relay.test.com")
        assert result["connected"] is False
        assert "not installed" in result["error"]


# ── Subscription Filter Tests ───────────────────────────────────────────

class TestSubscriptionFilter:
    """Tests for relay subscription filter acceptance."""

    def test_accepted_subscription(self):
        """Relay accepting a kind:4 subscription returns success."""
        mock_ws = _make_mock_ws([
            json.dumps(["EOSE", "diag-4-test"]),
        ])

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = _test_subscription_filter("wss://relay.test.com", "aabb" * 16, _KIND_ENCRYPTED_DM)

        assert result["accepted"] is True
        assert result["error"] is None

    def test_subscription_with_events(self):
        """Relay returning events before EOSE counts them."""
        fake_event = {"id": "e1", "kind": 4, "content": "x"}
        mock_ws = _make_mock_ws([
            json.dumps(["EVENT", "sub1", fake_event]),
            json.dumps(["EVENT", "sub1", fake_event]),
            json.dumps(["EOSE", "sub1"]),
        ])

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = _test_subscription_filter("wss://relay.test.com", "aabb" * 16, _KIND_ENCRYPTED_DM)

        assert result["accepted"] is True
        assert result["event_count"] == 2

    def test_subscription_closed_by_relay(self):
        """Relay sending CLOSED rejects the subscription."""
        mock_ws = _make_mock_ws([
            json.dumps(["CLOSED", "sub1", "restricted: kind not allowed"]),
        ])

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = _test_subscription_filter("wss://relay.test.com", "aabb" * 16, _KIND_GIFT_WRAP)

        assert result["accepted"] is False
        assert result["event_count"] == 0

    def test_subscription_connect_failure(self):
        """Connection failure returns error."""
        with patch(
            "tollbooth.nostr_diagnostics.create_connection",
            side_effect=ConnectionError("down"),
        ):
            result = _test_subscription_filter("wss://dead.relay.com", "aabb" * 16, _KIND_ENCRYPTED_DM)

        assert result["accepted"] is False
        assert result["error"] is not None

    def test_kind_1059_subscription(self):
        """kind:1059 (gift wrap) subscription accepted."""
        mock_ws = _make_mock_ws([
            json.dumps(["EOSE", "sub1"]),
        ])

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = _test_subscription_filter("wss://relay.test.com", "aabb" * 16, _KIND_GIFT_WRAP)

        assert result["accepted"] is True

    @patch("tollbooth.nostr_diagnostics._HAS_WEBSOCKET", False)
    def test_missing_websocket_lib(self):
        """Missing websocket-client returns descriptive error."""
        result = _test_subscription_filter("wss://relay.test.com", "aabb" * 16, _KIND_ENCRYPTED_DM)
        assert result["accepted"] is False
        assert "not installed" in result["error"]


# ── Self-DM Round-Trip Tests ───────────────────────────────────────────

class TestSelfDmRoundtrip:
    """Tests for self-DM publish + subscribe verification."""

    def test_successful_roundtrip(self):
        """Self-DM published and retrieved successfully."""
        _nsec, privkey_hex, pubkey_hex, _npub = _operator_keys()

        # Capture sent messages from the publish WS so the subscribe WS
        # can echo back the correct event_id.
        sent_messages: list[str] = []
        publish_ws = MagicMock()
        publish_ws.send = MagicMock(side_effect=lambda msg: sent_messages.append(msg))
        publish_ws.recv = MagicMock(return_value=json.dumps(["OK", "evt123", True, ""]))

        subscribe_ws = MagicMock()
        subscribe_recv_calls = {"n": 0}

        def subscribe_recv():
            subscribe_recv_calls["n"] += 1
            if subscribe_recv_calls["n"] == 1 and sent_messages:
                try:
                    pub_msg = json.loads(sent_messages[0])
                    if isinstance(pub_msg, list) and pub_msg[0] == "EVENT":
                        event_id = pub_msg[1].get("id", "unknown")
                        return json.dumps([
                            "EVENT", "sub1",
                            {"id": event_id, "kind": 4, "content": "x", "pubkey": pubkey_hex},
                        ])
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass
            return json.dumps(["EOSE", "sub1"])

        subscribe_ws.recv = MagicMock(side_effect=subscribe_recv)

        call_count = {"n": 0}

        def create_conn_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return publish_ws
            return subscribe_ws

        with patch("tollbooth.nostr_diagnostics.create_connection", side_effect=create_conn_side_effect):
            result = _self_dm_roundtrip("wss://relay.test.com", privkey_hex, pubkey_hex)

        assert result["success"] is True
        assert result["roundtrip_ms"] is not None
        assert result["event_id"] is not None
        assert result["error"] is None

    def test_publish_rejected(self):
        """Relay rejecting the publish returns failure."""
        _nsec, privkey_hex, pubkey_hex, _npub = _operator_keys()

        publish_ws = MagicMock()
        publish_ws.recv = MagicMock(
            return_value=json.dumps(["OK", "evt123", False, "blocked: rate limit"]),
        )

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=publish_ws):
            result = _self_dm_roundtrip("wss://relay.test.com", privkey_hex, pubkey_hex)

        assert result["success"] is False
        assert "rejected" in result["error"]

    def test_event_not_found(self):
        """Event published but not found on subscribe returns failure."""
        _nsec, privkey_hex, pubkey_hex, _npub = _operator_keys()

        publish_ws = MagicMock()
        publish_ws.recv = MagicMock(return_value=json.dumps(["OK", "evt123", True, ""]))

        subscribe_ws = MagicMock()
        # Return EOSE immediately without any matching events
        subscribe_ws.recv = MagicMock(
            return_value=json.dumps(["EOSE", "sub1"]),
        )

        call_count = {"n": 0}

        def create_conn_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return publish_ws
            return subscribe_ws

        with patch("tollbooth.nostr_diagnostics.create_connection", side_effect=create_conn_side_effect):
            result = _self_dm_roundtrip("wss://relay.test.com", privkey_hex, pubkey_hex)

        assert result["success"] is False
        assert result["event_id"] is not None
        assert "not found" in result["error"]

    def test_connect_failure(self):
        """Connection failure returns error."""
        _nsec, privkey_hex, pubkey_hex, _npub = _operator_keys()

        with patch(
            "tollbooth.nostr_diagnostics.create_connection",
            side_effect=ConnectionError("down"),
        ):
            result = _self_dm_roundtrip("wss://dead.relay.com", privkey_hex, pubkey_hex)

        assert result["success"] is False
        assert "connect" in result["error"].lower()

    @patch("tollbooth.nostr_diagnostics._HAS_PYNOSTR", False)
    def test_missing_pynostr(self):
        """Missing pynostr returns descriptive error."""
        result = _self_dm_roundtrip("wss://relay.test.com", "aa" * 32, "bb" * 32)
        assert result["success"] is False
        assert "pynostr" in result["error"]

    @patch("tollbooth.nostr_diagnostics._HAS_NIP04", False)
    def test_missing_nip04(self):
        """Missing NIP-04 module returns descriptive error."""
        result = _self_dm_roundtrip("wss://relay.test.com", "aa" * 32, "bb" * 32)
        assert result["success"] is False
        assert "nip04" in result["error"]


class TestSelfDmRoundtripWithCapture:
    """Additional self-DM tests using send capture to get event_id."""

    def test_successful_roundtrip_captured(self):
        """Self-DM with captured event_id for precise verification."""
        _nsec, privkey_hex, pubkey_hex, _npub = _operator_keys()

        publish_ws = MagicMock()
        publish_ws.recv = MagicMock(return_value=json.dumps(["OK", "evt123", True, ""]))

        # We'll capture the event_id from the publish send call
        sent_messages = []
        original_send = publish_ws.send

        def capture_send(msg):
            sent_messages.append(msg)
            return original_send(msg)

        publish_ws.send = MagicMock(side_effect=capture_send)

        subscribe_ws = MagicMock()
        subscribe_recv_calls = {"n": 0}

        def subscribe_recv():
            subscribe_recv_calls["n"] += 1
            if subscribe_recv_calls["n"] == 1 and sent_messages:
                # Parse the published event to get its ID
                try:
                    pub_msg = json.loads(sent_messages[0])
                    if isinstance(pub_msg, list) and pub_msg[0] == "EVENT":
                        event_id = pub_msg[1].get("id", "unknown")
                        return json.dumps([
                            "EVENT", "sub1",
                            {"id": event_id, "kind": 4, "content": "x", "pubkey": pubkey_hex},
                        ])
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass
            return json.dumps(["EOSE", "sub1"])

        subscribe_ws.recv = MagicMock(side_effect=subscribe_recv)

        call_count = {"n": 0}

        def create_conn_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return publish_ws
            return subscribe_ws

        with patch("tollbooth.nostr_diagnostics.create_connection", side_effect=create_conn_side_effect):
            result = _self_dm_roundtrip("wss://relay.test.com", privkey_hex, pubkey_hex)

        assert result["success"] is True
        assert result["roundtrip_ms"] is not None
        assert result["roundtrip_ms"] >= 0
        assert result["event_id"] is not None


# ── courier_health Integration Tests ────────────────────────────────────

class TestCourierHealth:
    """Tests for the full courier_health() diagnostic function."""

    @pytest.mark.asyncio
    async def test_healthy_relay(self):
        """Single healthy relay reports all green."""
        nsec, _privkey_hex, _pubkey_hex, npub = _operator_keys()

        nip11_doc = {
            "name": "Good Relay",
            "supported_nips": [1, 4, 11, 17, 44],
        }

        # Mock NIP-11
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = nip11_doc

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        # Mock WebSocket — all operations succeed
        sent_messages = []

        def make_ws(*args, **kwargs):
            ws = MagicMock()
            recv_calls = {"n": 0}

            def ws_send(msg):
                sent_messages.append(msg)

            ws.send = MagicMock(side_effect=ws_send)

            def ws_recv():
                recv_calls["n"] += 1
                # Check what was sent to determine response
                if sent_messages:
                    last = sent_messages[-1]
                    try:
                        parsed = json.loads(last)
                        if isinstance(parsed, list):
                            if parsed[0] == "EVENT":
                                event_data = parsed[1]
                                event_id = event_data.get("id", "unknown")
                                return json.dumps(["OK", event_id, True, ""])
                            elif parsed[0] == "REQ":
                                # Check if this is a self-ping verification
                                filters = parsed[2] if len(parsed) > 2 else {}
                                if "ids" in filters:
                                    target_id = filters["ids"][0]
                                    return json.dumps([
                                        "EVENT", parsed[1],
                                        {"id": target_id, "kind": 4},
                                    ])
                                return json.dumps(["EOSE", parsed[1]])
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
                return json.dumps(["EOSE", "sub"])

            ws.recv = MagicMock(side_effect=ws_recv)
            return ws

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_http), \
             patch("tollbooth.nostr_diagnostics.create_connection", side_effect=make_ws):
            result = await courier_health(["wss://relay.test.com"], nsec)

        assert result["success"] is True
        assert result["operator_npub"] == npub
        assert result["relay_count"] == 1
        assert result["summary"]["connected"] == 1
        assert result["summary"]["unreachable"] == 0
        assert result["summary"]["nip04_capable"] == 1
        assert result["summary"]["nip17_capable"] == 1

        relay = result["relays"][0]
        assert relay["relay"] == "wss://relay.test.com"
        assert relay["connectivity"]["connected"] is True
        assert relay["nip_support"]["nip04"] is True
        assert relay["nip_support"]["nip17"] is True
        assert relay["nip_support"]["nip44"] is True

    @pytest.mark.asyncio
    async def test_unreachable_relay(self):
        """Unreachable relay is reported cleanly."""
        nsec, _, _, _npub = _operator_keys()

        with patch(
            "tollbooth.nostr_diagnostics.create_connection",
            side_effect=ConnectionRefusedError("refused"),
        ):
            result = await courier_health(["wss://dead.relay.com"], nsec)

        assert result["success"] is True
        assert result["summary"]["connected"] == 0
        assert result["summary"]["unreachable"] == 1

        relay = result["relays"][0]
        assert relay["connectivity"]["connected"] is False
        # Skipped tests should be noted
        assert relay["nip11"]["error"] == "skipped (not connected)"
        assert relay["self_dm_roundtrip"]["error"] == "skipped"

    @pytest.mark.asyncio
    async def test_multiple_relays_mixed(self):
        """Mix of healthy and unreachable relays."""
        nsec, _privkey_hex, _pubkey_hex, _npub = _operator_keys()

        nip11_doc = {"name": "Good Relay", "supported_nips": [4]}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = nip11_doc

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        call_count = {"n": 0}

        def create_conn(url, **kwargs):
            call_count["n"] += 1
            if "dead" in url:
                raise ConnectionRefusedError("refused")
            ws = MagicMock()
            ws.recv = MagicMock(return_value=json.dumps(["EOSE", "sub"]))
            return ws

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_http), \
             patch("tollbooth.nostr_diagnostics.create_connection", side_effect=create_conn):
            result = await courier_health(
                ["wss://good.relay.com", "wss://dead.relay.com"],
                nsec,
            )

        assert result["summary"]["connected"] == 1
        assert result["summary"]["unreachable"] == 1
        assert len(result["relays"]) == 2

    @pytest.mark.asyncio
    async def test_invalid_nsec(self):
        """Invalid nsec returns error."""
        result = await courier_health(["wss://relay.test.com"], "nsec1invalid")
        assert result["success"] is False
        assert "Invalid operator nsec" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_relays(self):
        """Empty relay list returns error."""
        nsec, _, _, _ = _operator_keys()
        result = await courier_health([], nsec)
        assert result["success"] is False
        assert "No relays" in result["error"]

    @pytest.mark.asyncio
    @patch("tollbooth.nostr_diagnostics._HAS_PYNOSTR", False)
    async def test_missing_pynostr(self):
        """Missing pynostr returns error."""
        result = await courier_health(["wss://relay.test.com"], "nsec1test")
        assert result["success"] is False
        assert "pynostr" in result["error"]

    @pytest.mark.asyncio
    async def test_relay_url_whitespace_stripped(self):
        """Relay URLs with whitespace are trimmed."""
        nsec, _, _, _ = _operator_keys()

        with patch(
            "tollbooth.nostr_diagnostics.create_connection",
            side_effect=ConnectionRefusedError("refused"),
        ):
            result = await courier_health(["  wss://relay.test.com  ", "  "], nsec)

        # Only one real relay (empty string stripped)
        assert result["relay_count"] == 1

    @pytest.mark.asyncio
    async def test_nip11_error_doesnt_crash(self):
        """NIP-11 fetch failure doesn't crash the overall diagnostic."""
        nsec, _, _, _ = _operator_keys()

        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(return_value=json.dumps(["EOSE", "sub"]))

        import httpx as httpx_mod
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx_mod.TimeoutException("nip11 timeout"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("tollbooth.nostr_diagnostics.httpx.AsyncClient", return_value=mock_http), \
             patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = await courier_health(["wss://relay.test.com"], nsec)

        assert result["success"] is True
        relay = result["relays"][0]
        assert "error" in relay["nip11"]
        # NIP support falls back to all-false when NIP-11 fails
        assert relay["nip_support"] == {"nip04": False, "nip17": False, "nip44": False}


# ── courier_ping Tests ──────────────────────────────────────────────────

class TestCourierPing:
    """Tests for the lightweight courier_ping() function."""

    @pytest.mark.asyncio
    async def test_successful_ping(self):
        """Ping confirmed by relay."""
        nsec, _privkey_hex, _pubkey_hex, npub = _operator_keys()
        recipient = PrivateKey()
        recipient_npub = recipient.public_key.bech32()

        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(
            return_value=json.dumps(["OK", "evt123", True, ""]),
        )

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = await courier_ping(recipient_npub, ["wss://relay.test.com"], nsec)

        assert result["success"] is True
        assert result["event_id"] is not None
        assert result["recipient_npub"] == recipient_npub
        assert result["operator_npub"] == npub
        assert result["relays_confirmed"] == 1
        assert result["relays_attempted"] == 1
        assert result["timestamp"] > 0
        assert result["nonce"] is not None
        assert len(result["relay_results"]) == 1
        assert result["relay_results"][0]["confirmed"] is True

    @pytest.mark.asyncio
    async def test_ping_relay_rejects(self):
        """Relay rejecting the ping event."""
        nsec, _, _, _npub = _operator_keys()
        recipient = PrivateKey()

        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(
            return_value=json.dumps(["OK", "evt123", False, "blocked: spam"]),
        )

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = await courier_ping(
                recipient.public_key.bech32(),
                ["wss://relay.test.com"],
                nsec,
            )

        assert result["success"] is False
        assert result["relays_confirmed"] == 0
        assert result["relay_results"][0]["confirmed"] is False
        assert "blocked" in result["relay_results"][0]["error"]

    @pytest.mark.asyncio
    async def test_ping_relay_down(self):
        """Unreachable relay during ping."""
        nsec, _, _, _ = _operator_keys()
        recipient = PrivateKey()

        with patch(
            "tollbooth.nostr_diagnostics.create_connection",
            side_effect=ConnectionError("down"),
        ):
            result = await courier_ping(
                recipient.public_key.bech32(),
                ["wss://dead.relay.com"],
                nsec,
            )

        assert result["success"] is False
        assert result["relays_confirmed"] == 0
        assert result["relay_results"][0]["confirmed"] is False

    @pytest.mark.asyncio
    async def test_ping_multiple_relays(self):
        """Ping to multiple relays — partial success."""
        nsec, _, _, _ = _operator_keys()
        recipient = PrivateKey()

        call_count = {"n": 0}

        def create_conn(url, **kwargs):
            call_count["n"] += 1
            if "dead" in url:
                raise ConnectionError("down")
            ws = MagicMock()
            ws.recv = MagicMock(
                return_value=json.dumps(["OK", "evt", True, ""]),
            )
            return ws

        with patch("tollbooth.nostr_diagnostics.create_connection", side_effect=create_conn):
            result = await courier_ping(
                recipient.public_key.bech32(),
                ["wss://good.relay.com", "wss://dead.relay.com"],
                nsec,
            )

        assert result["success"] is True  # at least one confirmed
        assert result["relays_confirmed"] == 1
        assert result["relays_attempted"] == 2

    @pytest.mark.asyncio
    async def test_ping_invalid_recipient(self):
        """Invalid recipient npub returns error."""
        nsec, _, _, _ = _operator_keys()

        result = await courier_ping("npub1invalid", ["wss://relay.test.com"], nsec)
        assert result["success"] is False
        assert "Invalid recipient" in result["error"]

    @pytest.mark.asyncio
    async def test_ping_invalid_nsec(self):
        """Invalid operator nsec returns error."""
        recipient = PrivateKey()
        result = await courier_ping(
            recipient.public_key.bech32(),
            ["wss://relay.test.com"],
            "nsec1invalid",
        )
        assert result["success"] is False
        assert "Invalid operator nsec" in result["error"]

    @pytest.mark.asyncio
    async def test_ping_empty_relays(self):
        """Empty relay list returns error."""
        nsec, _, _, _ = _operator_keys()
        recipient = PrivateKey()
        result = await courier_ping(recipient.public_key.bech32(), [], nsec)
        assert result["success"] is False
        assert "No relays" in result["error"]

    @pytest.mark.asyncio
    @patch("tollbooth.nostr_diagnostics._HAS_PYNOSTR", False)
    async def test_ping_missing_pynostr(self):
        """Missing pynostr returns error."""
        result = await courier_ping("npub1test", ["wss://relay.test.com"], "nsec1test")
        assert result["success"] is False
        assert "pynostr" in result["error"]

    @pytest.mark.asyncio
    @patch("tollbooth.nostr_diagnostics._HAS_WEBSOCKET", False)
    async def test_ping_missing_websocket(self):
        """Missing websocket-client returns error."""
        nsec, _, _, _ = _operator_keys()
        recipient = PrivateKey()
        result = await courier_ping(
            recipient.public_key.bech32(),
            ["wss://relay.test.com"],
            nsec,
        )
        assert result["success"] is False
        assert "websocket-client" in result["error"]

    @pytest.mark.asyncio
    @patch("tollbooth.nostr_diagnostics._HAS_NIP04", False)
    async def test_ping_missing_nip04(self):
        """Missing NIP-04 module returns error."""
        nsec, _, _, _ = _operator_keys()
        recipient = PrivateKey()
        result = await courier_ping(
            recipient.public_key.bech32(),
            ["wss://relay.test.com"],
            nsec,
        )
        assert result["success"] is False
        assert "NIP-04" in result["error"]

    @pytest.mark.asyncio
    async def test_ping_message_content(self):
        """Ping message includes operator npub and nonce."""
        nsec, _, _, _npub = _operator_keys()
        recipient = PrivateKey()

        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(
            return_value=json.dumps(["OK", "evt", True, ""]),
        )

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = await courier_ping(
                recipient.public_key.bech32(),
                ["wss://relay.test.com"],
                nsec,
            )

        assert result["nonce"] is not None
        assert len(result["nonce"]) == 12
        assert result["timestamp_iso"] is not None
        assert "T" in result["timestamp_iso"]

    @pytest.mark.asyncio
    async def test_ping_recv_timeout(self):
        """Relay recv timeout during ping."""
        nsec, _, _, _ = _operator_keys()
        recipient = PrivateKey()

        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(side_effect=TimeoutError("recv timeout"))

        with patch("tollbooth.nostr_diagnostics.create_connection", return_value=mock_ws):
            result = await courier_ping(
                recipient.public_key.bech32(),
                ["wss://relay.test.com"],
                nsec,
            )

        assert result["success"] is False
        assert result["relay_results"][0]["confirmed"] is False
        assert "recv" in result["relay_results"][0]["error"].lower()
