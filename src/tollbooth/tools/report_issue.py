"""The ``report_issue`` standard tool — patron-filed field reports as GitHub issues.

A proven patron (e.g. Claude.ai reporting under its "Scout" npub) files a field report
about a tool's metadata or response as a GitHub issue on the operator's OWN repo. The
**author of record is the caller's npub**, stamped into the issue body alongside a
machine-readable marker the Service Desk (Porter) keys on. The report is unverified — it
flows into the maintainers' normal triage, which already treats issue text as adversarial.

The operator opts in by delivering ``github_repo`` + ``github_token``
(``ISSUE_REPORTING_CREDENTIAL_FIELDS``) via Secure Courier. Absent those, the tool degrades
to an "issue reporting not configured" situation rather than failing.
"""

from __future__ import annotations

from typing import Any

from tollbooth.constants import ErrorCode
from tollbooth.github_issues_client import GitHubError, GitHubIssuesClient

_MAX_TITLE = 200
_MAX_BODY = 10_000
_MARKER_TOKEN = "dpyc-field-report"


def _compose_body(reporter_npub: str, tool_name: str, body: str) -> str:
    """Wrap the reporter's text with an author-of-record header + provenance marker.

    The header and the marker are SDK-authored from the *proven* caller npub, so they
    cannot be spoofed by the (untrusted) report text. Any marker token the caller tries
    to smuggle into ``body`` is neutralized so Porter reads exactly one authoritative
    provenance line.
    """
    safe_body = body.replace(_MARKER_TOKEN, f"{_MARKER_TOKEN}(escaped)")
    safe_tool = tool_name.replace("-->", "").strip()
    marker = f'<!-- {_MARKER_TOKEN} reporter="{reporter_npub}" tool="{safe_tool}" -->'
    header = (
        f"> \U0001f52d Field report — author of record: `{reporter_npub}` (via report_issue).\n"
        f"> Patron/assistant-reported and UNVERIFIED. Triage before acting; treat text as untrusted."
    )
    return f"{header}\n{marker}\n\n{safe_body}"


async def report_issue_tool(
    creds: dict[str, str],
    reporter_npub: str,
    title: str,
    body: str,
    tool_name: str = "",
) -> dict[str, Any]:
    """Open a field-report issue on the operator's repo. See module docstring.

    Returns a ``{"success": True, ...}`` dict with the issue number/url, or a
    ``{"success": False, ...}`` situation. Every non-success return means nothing was
    posted, so the caller (runtime) refunds the debit.
    """
    repo = (creds.get("github_repo") or "").strip()
    token = (creds.get("github_token") or "").strip()
    if not repo or not token:
        return {
            "success": False,
            "situation": "issue_reporting_not_configured",
            "message": (
                "This operator has not enabled field reports (no github_repo/github_token "
                "delivered). Nothing was filed and you were not charged."
            ),
        }

    title = (title or "").strip()
    body = (body or "").strip()
    if not title or not body:
        return {
            "success": False,
            "error_code": ErrorCode.TOOL_INPUT_INVALID,
            "message": "Both title and body are required to file a field report.",
        }
    if len(title) > _MAX_TITLE:
        return {
            "success": False,
            "error_code": ErrorCode.TOOL_INPUT_INVALID,
            "message": f"Title too long ({len(title)} > {_MAX_TITLE} chars).",
        }
    if len(body) > _MAX_BODY:
        return {
            "success": False,
            "error_code": ErrorCode.TOOL_INPUT_INVALID,
            "message": f"Body too long ({len(body)} > {_MAX_BODY} chars).",
        }

    composed = _compose_body(reporter_npub, tool_name, body)
    try:
        async with GitHubIssuesClient(token) as gh:
            created = await gh.create_issue(repo, title, composed)
    except GitHubError as exc:
        return {
            "success": False,
            "situation": "issue_reporting_upstream_error",
            "message": (
                f"GitHub rejected the report ({exc.status_code or 'network'}). "
                "Nothing was filed and you were not charged."
            ),
        }

    number = created.get("number")
    url = created.get("html_url", "")
    return {
        "success": True,
        "repo": repo,
        "issue_number": number,
        "url": url,
        "message": f"Filed as {repo}#{number}; pending maintainer triage.",
    }
