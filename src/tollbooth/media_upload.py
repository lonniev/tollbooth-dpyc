"""NIP-96 media upload for Nostr-compatible image hosting.

Uploads PNG images to nostr.build for inclusion in Nostr DMs.
Used by SecureCourierService to send credential card QR codes
as inline images rather than raw text strings.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

NOSTR_BUILD_UPLOAD_URL = "https://nostr.build/api/v2/upload/files"


class MediaUploadError(Exception):
    """Raised when a media upload fails."""


async def upload_to_nostr_build(
    png_bytes: bytes,
    *,
    alt: str = "DPYC credential card QR",
    timeout: float = 15.0,
) -> str:
    """Upload PNG to nostr.build NIP-96 API. Returns the image URL.

    Args:
        png_bytes: Raw PNG image bytes.
        alt: Alt-text for the uploaded image.
        timeout: HTTP timeout in seconds.

    Returns:
        The public URL of the uploaded image.

    Raises:
        MediaUploadError: On HTTP errors, timeouts, or unexpected responses.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                NOSTR_BUILD_UPLOAD_URL,
                files={"file": ("credential_card.png", png_bytes, "image/png")},
                data={"alt": alt},
            )
        except httpx.TimeoutException as exc:
            raise MediaUploadError(f"Upload timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise MediaUploadError(f"Upload failed: {exc}") from exc

    if resp.status_code != 200:
        raise MediaUploadError(
            f"Upload returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise MediaUploadError(f"Invalid JSON response: {exc}") from exc

    # nostr.build returns {"status": "success", "data": [{"url": "..."}]}
    try:
        url = data["data"][0]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MediaUploadError(
            f"Unexpected response structure: {exc}"
        ) from exc

    if not isinstance(url, str) or not url.startswith("http"):
        raise MediaUploadError(f"Invalid URL in response: {url!r}")

    return url
