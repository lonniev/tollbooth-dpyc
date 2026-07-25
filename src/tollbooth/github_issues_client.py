"""Async HTTP client for GitHub's REST Issues API.

Deliberately minimal — the SDK only needs to open an issue on the operator's own
repo for the standard ``report_issue`` tool. Uses ``httpx`` so it rides the same
WASI seam (httpx→wasi:http) as the rest of the wheel and runs unchanged on both
FastMCP and Spin hosts.
"""

from __future__ import annotations

from typing import Any, Self

import httpx

_API_BASE = "https://api.github.com"
_API_VERSION = "2022-11-28"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class GitHubError(Exception):
    """Base exception for GitHub Issues operations."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubAuthError(GitHubError):
    """401/403 — bad or under-scoped token."""


class GitHubNotFoundError(GitHubError):
    """404 — repo not found or token cannot see it."""


class GitHubValidationError(GitHubError):
    """422 — request rejected (e.g. nonexistent label, empty title)."""


class GitHubServerError(GitHubError):
    """5xx — GitHub-side error (retryable)."""


class GitHubConnectionError(GitHubError):
    """Network/DNS failure (retryable)."""


class GitHubTimeoutError(GitHubError):
    """Request timeout (retryable)."""


_STATUS_MAP: dict[int, type[GitHubError]] = {
    401: GitHubAuthError,
    403: GitHubAuthError,
    404: GitHubNotFoundError,
    422: GitHubValidationError,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GitHubIssuesClient:
    """Async client for the subset of the GitHub REST API the wheel needs.

    Constructor takes an explicit token — no env-var loading. The token is a
    fine-grained PAT (or App installation token) with ``issues:write`` on the
    target repo and nothing more.
    """

    def __init__(self, token: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=_API_BASE,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": _API_VERSION,
            },
            timeout=httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0),
        )

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
    ) -> Any:
        """Send a request and map errors to the GitHub exception hierarchy."""
        try:
            response = await self._client.request(method, endpoint, json=json_data)
        except httpx.ConnectError as exc:
            raise GitHubConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise GitHubTimeoutError(str(exc)) from exc

        if response.status_code >= 400:
            body = response.text
            exc_cls = _STATUS_MAP.get(response.status_code)
            if exc_cls is not None:
                raise exc_cls(body, status_code=response.status_code)
            if response.status_code >= 500:
                raise GitHubServerError(body, status_code=response.status_code)
            raise GitHubError(body, status_code=response.status_code)

        return response.json()

    async def create_issue(
        self,
        repo: str,
        title: str,
        body: str,
    ) -> dict[str, Any]:
        """Open an issue on ``owner/repo``.

        Returns the created issue's ``number`` and ``html_url``.
        """
        payload: dict[str, Any] = {"title": title, "body": body}
        data = await self._request("POST", f"/repos/{repo}/issues", payload)
        return {
            "number": data.get("number"),
            "html_url": data.get("html_url", ""),
        }

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
