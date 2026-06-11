"""Unit tests for tollbooth.tools.status.build_service_status (audit M2.1d)."""

from __future__ import annotations

import hashlib

from tollbooth.tools.status import build_service_status


def _call(**over):
    kw = dict(
        service="svc", slug="svc", version="1.2.3",
        tollbooth_version="0.44.3", vault_ok=True, courier_ok=True,
        operator_npub="npub1op", process_id=4242, env={},
    )
    kw.update(over)
    return build_service_status(**kw)


def test_core_fields_passthrough():
    r = _call()
    assert r["success"] is True
    assert r["service"] == "svc" and r["slug"] == "svc" and r["version"] == "1.2.3"
    assert r["tollbooth_dpyc_version"] == "0.44.3"
    assert r["vault_configured"] is True and r["courier_has_vault"] is True
    assert r["process_id"] == 4242


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
