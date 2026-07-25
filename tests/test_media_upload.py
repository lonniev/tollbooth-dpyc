"""Tests for media_upload.py — NIP-96 image upload to nostr.build."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tollbooth.media_upload import MediaUploadError, upload_to_nostr_build

# ---------------------------------------------------------------------------
# upload_to_nostr_build tests
# ---------------------------------------------------------------------------

FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # minimal PNG-ish bytes
FAKE_URL = "https://image.nostr.build/abc123.png"


def _mock_response(*, status_code: int = 200, json_data: dict | None = None) -> httpx.Response:
    """Build a fake httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = '{"status":"success"}'
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


@pytest.mark.asyncio
async def test_upload_success():
    """Happy path: mock httpx returns URL."""
    resp = _mock_response(json_data={"status": "success", "data": [{"url": FAKE_URL}]})

    mock_client = AsyncMock()
    mock_client.post.return_value = resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.media_upload.httpx.AsyncClient", return_value=mock_client):
        url = await upload_to_nostr_build(FAKE_PNG)

    assert url == FAKE_URL
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_upload_server_error():
    """HTTP 500 raises MediaUploadError."""
    resp = _mock_response(status_code=500)
    resp.text = "Internal Server Error"

    mock_client = AsyncMock()
    mock_client.post.return_value = resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.media_upload.httpx.AsyncClient", return_value=mock_client):  # noqa: SIM117
        with pytest.raises(MediaUploadError, match="HTTP 500"):
            await upload_to_nostr_build(FAKE_PNG)


@pytest.mark.asyncio
async def test_upload_timeout():
    """Timeout raises MediaUploadError."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ReadTimeout("timed out")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.media_upload.httpx.AsyncClient", return_value=mock_client):  # noqa: SIM117
        with pytest.raises(MediaUploadError, match="timed out"):
            await upload_to_nostr_build(FAKE_PNG)


@pytest.mark.asyncio
async def test_upload_bad_response():
    """Missing URL in response raises MediaUploadError."""
    resp = _mock_response(json_data={"status": "success", "data": []})

    mock_client = AsyncMock()
    mock_client.post.return_value = resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.media_upload.httpx.AsyncClient", return_value=mock_client):  # noqa: SIM117
        with pytest.raises(MediaUploadError, match="Unexpected response"):
            await upload_to_nostr_build(FAKE_PNG)


# ---------------------------------------------------------------------------
# _deliver_card_dm integration tests (via SecureCourierService)
# ---------------------------------------------------------------------------

# These tests verify the DM text formatting with and without QR upload.
# They mock the courier internals to avoid real Nostr connections.


def _make_courier_service():
    """Build a minimal SecureCourierService with mocked internals."""
    # We can't construct SecureCourierService without nostr deps, so we
    # test _deliver_card_dm as a standalone coroutine by building a
    # minimal mock that has the method bound.
    from tollbooth.secure_courier import SecureCourierService

    svc = object.__new__(SecureCourierService)
    svc._operator_nsec = "nsec1fake"
    svc._send_card_dm = True
    svc._sessions = {}

    # Mock the exchange
    svc._exchange = MagicMock()
    svc._exchange.send_dm = MagicMock()
    svc._on_received = None

    return svc


@pytest.mark.asyncio
async def test_dm_with_qr_includes_url():
    """DM text contains image URL when upload succeeds."""
    svc = _make_courier_service()

    with patch(
        "tollbooth.media_upload.upload_to_nostr_build",
        new_callable=AsyncMock,
        return_value=FAKE_URL,
    ):
        await svc._deliver_card_dm(
            "npub1test", "mybrain", "ncred1abc123", qr_png=FAKE_PNG,
        )

    dm_text = svc._exchange.send_dm.call_args[0][1]
    assert FAKE_URL in dm_text
    assert "ncred1abc123" in dm_text


@pytest.mark.asyncio
async def test_dm_without_qr_no_url():
    """DM text unchanged when qr_png=None."""
    svc = _make_courier_service()

    await svc._deliver_card_dm("npub1test", "mybrain", "ncred1abc123")

    dm_text = svc._exchange.send_dm.call_args[0][1]
    assert "nostr.build" not in dm_text
    assert "ncred1abc123" in dm_text


@pytest.mark.asyncio
async def test_dm_upload_failure_still_sends():
    """Upload fails but DM is still sent without URL (non-fatal)."""
    svc = _make_courier_service()

    with patch(
        "tollbooth.media_upload.upload_to_nostr_build",
        new_callable=AsyncMock,
        side_effect=Exception("upload boom"),
    ):
        await svc._deliver_card_dm(
            "npub1test", "mybrain", "ncred1abc123", qr_png=FAKE_PNG,
        )

    # DM should still be sent
    svc._exchange.send_dm.assert_called_once()
    dm_text = svc._exchange.send_dm.call_args[0][1]
    assert "ncred1abc123" in dm_text
    # No URL since upload failed
    assert "nostr.build" not in dm_text
