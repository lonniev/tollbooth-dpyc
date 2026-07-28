"""Unit tests for tollbooth.tools.status.build_service_status (audit M2.1d)."""

from __future__ import annotations

import hashlib

from tollbooth.tools.status import (
    build_docket_diagnostic,
    build_service_status,
    build_upstream_oauth_block,
)


def _call(**over):
    kw = {
        "service": "svc", "slug": "svc", "version": "1.2.3",
        "tollbooth_version": "0.44.3", "vault_ok": True, "courier_ok": True,
        "operator_npub": "npub1op", "process_id": 4242, "env": {},
    }
    kw.update(over)
    return build_service_status(**kw)


def test_core_fields_passthrough():
    r = _call()
    assert r["success"] is True
    assert r["service"] == "svc" and r["slug"] == "svc" and r["version"] == "1.2.3"
    assert r["tollbooth_dpyc_version"] == "0.44.3"
    assert r["vault_configured"] is True and r["courier_has_vault"] is True
    assert r["process_id"] == 4242


def test_docket_diagnostic_unset_is_ephemeral_memory():
    d = build_docket_diagnostic({})
    assert d == {
        "docket_url_set": False,
        "backend": "memory (default)",
        "durable_across_recycles": False,
    }


def test_docket_diagnostic_redis_is_durable_and_never_leaks_url():
    secret = "rediss://user:s3cr3t@my-redis.example.com:6379/0"
    d = build_docket_diagnostic({"FASTMCP_DOCKET_URL": secret})
    assert d["docket_url_set"] is True
    assert d["backend"] == "rediss" and d["durable_across_recycles"] is True
    # The credential-bearing URL must NEVER appear in the diagnostic.
    assert "s3cr3t" not in str(d) and secret not in str(d)


def test_docket_diagnostic_memory_scheme_is_not_durable():
    d = build_docket_diagnostic({"FASTMCP_DOCKET_URL": "memory://"})
    assert d["docket_url_set"] is True
    assert d["backend"] == "memory" and d["durable_across_recycles"] is False


def test_service_status_includes_async_jobs_block_when_opted_in():
    assert _call(
        include_async_jobs=True, env={"FASTMCP_DOCKET_URL": "redis://h:6379"}
    )["async_jobs"]["durable_across_recycles"] is True
    assert _call(include_async_jobs=True)["async_jobs"]["docket_url_set"] is False


def test_service_status_omits_async_jobs_block_by_default():
    # A server that never opted into async jobs must not advertise an inert
    # background-task subsystem — the block is absent entirely, not falsy.
    assert "async_jobs" not in _call()
    assert "async_jobs" not in _call(env={"FASTMCP_DOCKET_URL": "redis://h:6379"})


def test_operator_npub_fingerprint_is_sha256_prefix():
    r = _call(operator_npub="npub1abc")
    assert r["operator_npub_hash"] == hashlib.sha256(b"npub1abc").hexdigest()[:12]


def test_none_npub_yields_empty_hash():
    assert _call(operator_npub=None)["operator_npub_hash"] == ""


def test_empty_string_npub_is_still_hashed():
    # Faithful to the original: a resolved-but-empty npub is fingerprinted.
    r = _call(operator_npub="")
    assert r["operator_npub_hash"] == hashlib.sha256(b"").hexdigest()[:12]
    assert r["operator_npub_hash"] != ""


def test_build_info_from_env_lowercased_and_filtered():
    env = {
        "FASTMCP_CLOUD_URL": "https://x.fastmcp.app",
        "FASTMCP_CLOUD_GIT_COMMIT_SHA": "",  # empty → excluded
        "FASTMCP_CLOUD_GIT_REPO": "lonniev/x",
        "UNRELATED": "ignored",
    }
    r = _call(env=env)
    assert r["build_info"] == {
        "fastmcp_cloud_url": "https://x.fastmcp.app",
        "fastmcp_cloud_git_repo": "lonniev/x",
    }


def test_build_info_none_when_empty():
    assert _call(env={})["build_info"] is None


def test_flags_reflect_inputs():
    r = _call(vault_ok=False, courier_ok=False)
    assert r["vault_configured"] is False and r["courier_has_vault"] is False


# ── build_upstream_oauth_block ────────────────────────────────────────

def test_oauth_block_none_when_no_creds():
    assert build_upstream_oauth_block(None, service_name="schwab", refresh_enabled=True, now_ts=0) is None


def test_oauth_block_none_without_access_token():
    creds = {"refresh_token": "r"}
    assert build_upstream_oauth_block(creds, service_name="schwab", refresh_enabled=True, now_ts=0) is None


def test_oauth_block_with_expiry_computes_remaining():
    creds = {"access_token": "a", "refresh_token": "r", "expires_at": "1000"}
    block = build_upstream_oauth_block(creds, service_name="schwab", refresh_enabled=True, now_ts=600)
    assert block == {
        "service": "schwab",
        "has_access_token": True,
        "has_refresh_token": True,
        "refresh_enabled": True,
        "access_token_expires_at": 1000.0,
        "access_token_expires_in_seconds": 400,
    }


def test_oauth_block_expiry_clamped_at_zero():
    creds = {"access_token": "a", "expires_at": "100"}
    block = build_upstream_oauth_block(creds, service_name="x", refresh_enabled=False, now_ts=9999)
    assert block["access_token_expires_in_seconds"] == 0
    assert block["has_refresh_token"] is False and block["refresh_enabled"] is False


def test_oauth_block_omits_expiry_fields_when_unset_or_unparseable():
    # Missing expires_at → no expiry fields.
    b1 = build_upstream_oauth_block({"access_token": "a"}, service_name="x", refresh_enabled=True, now_ts=0)
    assert "access_token_expires_at" not in b1 and "access_token_expires_in_seconds" not in b1
    # Garbage expires_at → parsed to 0.0, treated as unset.
    b2 = build_upstream_oauth_block(
        {"access_token": "a", "expires_at": "not-a-number"},
        service_name="x", refresh_enabled=True, now_ts=0,
    )
    assert "access_token_expires_at" not in b2
