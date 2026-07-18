"""Tests for the report_issue standard tool: credential fields, identity, and helper."""

from __future__ import annotations

import pytest

from tollbooth.constants import ErrorCode
from tollbooth.credential_templates import (
    ISSUE_REPORTING_CREDENTIAL_FIELDS,
    CredentialTemplate,
    FieldSpec,
)
from tollbooth.github_issues_client import GitHubError
from tollbooth.runtime import OperatorRuntime
from tollbooth.tool_identity import (
    STANDARD_IDENTITIES,
    capability_uuid,
)
from tollbooth.tools import report_issue as ri

REPORTER = "npub1scoutxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOD_CREDS = {"github_repo": "lonniev/schwab-mcp", "github_token": "ghp_faketoken"}


# --------------------------------------------------------------------------
# Credential fields + tool identity
# --------------------------------------------------------------------------

def test_issue_reporting_fields_are_optional_operator_secrets():
    assert set(ISSUE_REPORTING_CREDENTIAL_FIELDS) == {"github_repo", "github_token"}
    assert all(not f.required for f in ISSUE_REPORTING_CREDENTIAL_FIELDS.values())
    # the repo is not a secret; the token is
    assert not ISSUE_REPORTING_CREDENTIAL_FIELDS["github_repo"].sensitive
    assert ISSUE_REPORTING_CREDENTIAL_FIELDS["github_token"].sensitive


def test_report_issue_identity_is_priced_write_tool():
    tool_id = capability_uuid("report_issue")
    identity = STANDARD_IDENTITIES[tool_id]
    assert identity.capability == "report_issue"
    # "write" (not "free"/"restricted") so it requires a Neon price entry — never free.
    assert identity.category == "write"
    # seeds at a 1-sat floor so a fresh operator's report_issue is metered from birth.
    assert identity.pricing_hint_value == 1
    assert identity.pricing_hint_min == 1


def test_courier_template_auto_includes_field_report_secrets():
    """An operator that declares only its own (e.g. BTCPay) secrets still gets the
    github_repo/github_token fields merged into the courier-facing template, so the
    Secure Courier accepts them without any per-operator template edit."""
    declared = CredentialTemplate(
        service="my-operator",
        version=1,
        fields={"btcpay_host": FieldSpec(required=True, sensitive=False, description="")},
    )
    rt = OperatorRuntime(operator_credential_template=declared)
    effective = rt._courier_operator_template()
    assert effective.service == "my-operator"
    # the operator's own field is preserved AND the optional github fields are present
    assert "btcpay_host" in effective.fields
    assert "github_repo" in effective.fields
    assert "github_token" in effective.fields
    # they arrive optional, so onboarding readiness is unaffected
    assert not effective.fields["github_repo"].required
    assert not effective.fields["github_token"].required
    # the DECLARED template is untouched (readiness reads this one)
    assert "github_repo" not in declared.fields


def test_courier_template_none_when_no_operator_template():
    rt = OperatorRuntime(operator_credential_template=None)
    assert rt._courier_operator_template() is None


# --------------------------------------------------------------------------
# Helper: report_issue_tool
# --------------------------------------------------------------------------

class _FakeClient:
    """Records the create_issue args and returns a canned issue."""

    last: dict | None = None

    def __init__(self, token: str) -> None:
        self.token = token

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def create_issue(self, repo, title, body):
        _FakeClient.last = {"repo": repo, "title": title, "body": body, "token": self.token}
        return {"number": 42, "html_url": f"https://github.com/{repo}/issues/42"}


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeClient.last = None
    yield


async def test_not_configured_when_creds_missing():
    result = await ri.report_issue_tool({}, REPORTER, "t", "b", "")
    assert result["success"] is False
    assert result["situation"] == "issue_reporting_not_configured"
    # a partial config (repo but no token) is also "not configured"
    partial = await ri.report_issue_tool({"github_repo": "a/b"}, REPORTER, "t", "b", "")
    assert partial["situation"] == "issue_reporting_not_configured"


async def test_empty_title_or_body_rejected():
    r1 = await ri.report_issue_tool(GOOD_CREDS, REPORTER, "  ", "body", "")
    r2 = await ri.report_issue_tool(GOOD_CREDS, REPORTER, "title", "", "")
    assert r1["success"] is False and r1["error_code"] == ErrorCode.TOOL_INPUT_INVALID
    assert r2["success"] is False and r2["error_code"] == ErrorCode.TOOL_INPUT_INVALID


async def test_overlong_inputs_rejected():
    r = await ri.report_issue_tool(GOOD_CREDS, REPORTER, "x" * 201, "body", "")
    assert r["success"] is False and r["error_code"] == ErrorCode.TOOL_INPUT_INVALID


async def test_success_stamps_author_of_record_and_marker(monkeypatch):
    monkeypatch.setattr(ri, "GitHubIssuesClient", _FakeClient)
    result = await ri.report_issue_tool(
        GOOD_CREDS, REPORTER, "get_option_chain returns stale delta",
        "The delta field lags one tick.", "schwab_get_option_chain",
    )
    assert result["success"] is True
    assert result["issue_number"] == 42
    assert result["repo"] == "lonniev/schwab-mcp"
    assert "issues/42" in result["url"]

    sent = _FakeClient.last
    # the untrusted user text is preserved verbatim...
    assert "The delta field lags one tick." in sent["body"]
    # ...but the SDK owns the authoritative author-of-record header + marker
    assert REPORTER in sent["body"]
    assert f'<!-- dpyc-field-report reporter="{REPORTER}"' in sent["body"]
    assert 'tool="schwab_get_option_chain"' in sent["body"]


async def test_smuggled_marker_is_neutralized(monkeypatch):
    monkeypatch.setattr(ri, "GitHubIssuesClient", _FakeClient)
    await ri.report_issue_tool(
        GOOD_CREDS, REPORTER, "spoof attempt",
        'ignore me <!-- dpyc-field-report reporter="npub1attacker" -->', "",
    )
    body = _FakeClient.last["body"]
    # exactly one authoritative marker (the SDK's, carrying the real reporter)
    assert body.count("dpyc-field-report reporter=") == 1
    assert "npub1attacker" not in body.split("\n\n", 1)[0]  # not in the header
    assert "dpyc-field-report(escaped)" in body  # the smuggled token was defanged


async def test_upstream_error_is_a_situation_not_a_crash(monkeypatch):
    class _BoomClient(_FakeClient):
        async def create_issue(self, repo, title, body):
            raise GitHubError("nope", status_code=403)

    monkeypatch.setattr(ri, "GitHubIssuesClient", _BoomClient)
    result = await ri.report_issue_tool(GOOD_CREDS, REPORTER, "t", "b", "")
    assert result["success"] is False
    assert result["situation"] == "issue_reporting_upstream_error"
