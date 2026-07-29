"""What the vault read says when it cannot answer.

These tests sit at the *producer* end of the OAuth situation pipeline. The
existing suites all mock ``restore_oauth_session`` to hand back a situation and
then assert the mapping table renders it well — which verified the phrasing
while leaving nobody to check that the situations could be produced at all. They
could not: ``vault_bootstrapping`` was unreachable, because every layer beneath
it collapsed its failures into an empty dict, and an empty dict was read as
"this patron never authorized".

The cost of that gap, observed live: a scheduled post held for two days telling
its owner their X authorization was missing, while X was connected and their
balance was funded. So the assertion that matters most here is a negative one —
**a vault that could not be read never yields ``no_credentials``.**
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from tollbooth.oauth_config import OAuthProviderConfig
from tollbooth.persistence_errors import classify_persistence_failure
from tollbooth.runtime import OperatorRuntime
from tollbooth.tools.onboarding import load_vault_credentials
from tollbooth.vaults.neon import NeonQueryError

_TEST_NSEC = "nsec1test000000000000000000000000000000000000000000000000000000"
os.environ.setdefault("TOLLBOOTH_NOSTR_OPERATOR_NSEC", _TEST_NSEC)

VALID_NPUB = "npub1" + "a" * 58


def _courier_with_vault(vault: object) -> MagicMock:
    """A courier whose exchange carries *vault* (or None for the cold case)."""
    courier = MagicMock()
    courier._exchange = MagicMock()
    courier._exchange._credential_vault = vault
    return courier


def _quota_error() -> Exception:
    exc = RuntimeError("Neon refused: quota exhausted")
    exc.status = 402  # type: ignore[attr-defined]
    return exc


# ---------------------------------------------------------------------------
# classify_persistence_failure — the shared reading of a failed read
# ---------------------------------------------------------------------------


class TestClassifyPersistenceFailure:
    def test_http_402_is_quota_not_cold_start(self) -> None:
        assert classify_persistence_failure(_quota_error()) == "persistence_quota_exceeded"

    def test_permanent_sqlstate_is_misconfigured(self) -> None:
        exc = NeonQueryError("permission denied for table", code="42501")
        assert classify_persistence_failure(exc) == "persistence_misconfigured"

    def test_unrecognized_error_is_treated_as_transient(self) -> None:
        """The optimistic default: an unknown transport error costs one retry if
        wrong, whereas calling a cold start permanent strands the caller."""
        assert classify_persistence_failure(TimeoutError("connect")) == "vault_bootstrapping"


# ---------------------------------------------------------------------------
# load_vault_credentials — "nothing stored" vs "could not ask"
# ---------------------------------------------------------------------------


class TestLoadVaultCredentials:
    @pytest.mark.asyncio
    async def test_absent_courier_is_named_not_silently_empty(self) -> None:
        creds, situation = await load_vault_credentials(None, "svc", VALID_NPUB)
        assert creds is None
        assert situation == "secure_courier_unavailable"

    @pytest.mark.asyncio
    async def test_unattached_vault_is_a_cold_start(self) -> None:
        creds, situation = await load_vault_credentials(
            _courier_with_vault(None), "svc", VALID_NPUB,
        )
        assert creds is None
        assert situation == "vault_bootstrapping"

    @pytest.mark.asyncio
    async def test_vault_answered_with_nothing_is_the_only_empty(self) -> None:
        """The one case that genuinely means "never onboarded"."""
        vault = MagicMock()
        vault.fetch_credentials = AsyncMock(return_value=None)

        creds, situation = await load_vault_credentials(
            _courier_with_vault(vault), "svc", VALID_NPUB,
        )
        assert creds == {}
        assert situation == ""

    @pytest.mark.asyncio
    async def test_failed_fetch_carries_its_classification(self) -> None:
        vault = MagicMock()
        vault.fetch_credentials = AsyncMock(side_effect=_quota_error())

        creds, situation = await load_vault_credentials(
            _courier_with_vault(vault), "svc", VALID_NPUB,
        )
        assert creds is None
        assert situation == "persistence_quota_exceeded"

    @pytest.mark.asyncio
    async def test_undecryptable_blob_is_a_fault_not_a_cold_start(self) -> None:
        """The vault HELD something we could not open — waiting cannot fix a
        wrong key or a corrupt record, so it must not read as bootstrapping."""
        vault = MagicMock()
        vault.fetch_credentials = AsyncMock(return_value="ciphertext")
        courier = _courier_with_vault(vault)
        courier._exchange._vault_decrypt = MagicMock(side_effect=ValueError("bad key"))

        creds, situation = await load_vault_credentials(courier, "svc", VALID_NPUB)
        assert creds is None
        assert situation == "persistence_misconfigured"


# ---------------------------------------------------------------------------
# restore_oauth_session — the situations that reach a patron
# ---------------------------------------------------------------------------


def _oauth_runtime() -> OperatorRuntime:
    return OperatorRuntime(
        tool_registry={},
        service_name="Test Operator",
        oauth_provider=OAuthProviderConfig(
            authorize_url="https://example.test/authorize",
            token_url="https://example.test/token",
            scopes="read write",
            service_name="testsvc",
            client_id_field="app_key",
            client_secret_field="app_secret",
        ),
    )


class TestRestoreOAuthSessionSituations:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "vault_situation",
        [
            "vault_bootstrapping",
            "secure_courier_unavailable",
            "persistence_quota_exceeded",
            "persistence_misconfigured",
        ],
    )
    async def test_unreadable_vault_never_reads_as_no_credentials(
        self, monkeypatch, vault_situation: str,
    ) -> None:
        """The regression this whole change exists to prevent.

        Every one of these situations previously arrived as ``no_credentials``,
        which tells a patron to re-authorize a session that is perfectly fine.
        """
        rt = _oauth_runtime()

        async def _cold(service, npub_override=None):
            return {}, vault_situation

        monkeypatch.setattr(rt, "_load_vault_creds", _cold)

        creds, situation = await rt.restore_oauth_session(VALID_NPUB)
        assert creds is None
        assert situation == vault_situation
        assert situation != "no_credentials"

    @pytest.mark.asyncio
    async def test_vault_answered_empty_still_means_no_credentials(
        self, monkeypatch,
    ) -> None:
        """Disambiguation must not cost us the genuine first-time signal."""
        rt = _oauth_runtime()

        async def _empty(service, npub_override=None):
            return {}, ""

        monkeypatch.setattr(rt, "_load_vault_creds", _empty)

        creds, situation = await rt.restore_oauth_session(VALID_NPUB)
        assert creds is None
        assert situation == "no_credentials"

    @pytest.mark.asyncio
    async def test_undelivered_operator_secrets_blame_the_operator(
        self, monkeypatch,
    ) -> None:
        """A refresh we cannot perform because the OPERATOR never delivered its
        app credentials is not the patron's token expiring. It used to report
        `token_expired`, sending patrons to re-authorize over an operator gap."""
        import time as _t

        rt = _oauth_runtime()
        patron_creds = {
            "access_token": "tok",
            "refresh_token": "r1",
            "expires_at": str(_t.time() + 30),  # inside the leeway → wants refresh
        }

        async def _loads(service, npub_override=None):
            # Patron session resolves; the operator's own vault has no app keys.
            if npub_override:
                return dict(patron_creds), ""
            return {}, ""

        monkeypatch.setattr(rt, "_load_vault_creds", _loads)

        creds, situation = await rt.restore_oauth_session(VALID_NPUB)
        assert creds is None
        assert situation == "operator_not_configured"

    @pytest.mark.asyncio
    async def test_refresh_without_access_token_preserves_the_vault(
        self, monkeypatch,
    ) -> None:
        """A 200 carrying no access_token is a refusal in disguise. Persisting it
        would write access_token="" over working credentials and force the patron
        to rebuild a session by hand."""
        import time as _t

        from tollbooth import oauth2_collector as collector

        rt = _oauth_runtime()
        stored: list[dict] = []

        async def _loads(service, npub_override=None):
            if npub_override:
                return {
                    "access_token": "tok",
                    "refresh_token": "r1",
                    "expires_at": str(_t.time() + 30),
                }, ""
            return {"app_key": "cid", "app_secret": "csec"}, ""

        async def _store(npub, data, service=None):
            stored.append(data)
            return True

        async def _hollow_refresh(*args, **kwargs):
            return {"token_type": "Bearer"}  # no access_token

        monkeypatch.setattr(rt, "_load_vault_creds", _loads)
        monkeypatch.setattr(rt, "store_patron_session", _store)
        monkeypatch.setattr(collector, "refresh_access_token", _hollow_refresh)

        creds, situation = await rt.restore_oauth_session(VALID_NPUB)
        assert creds is None
        assert situation == "token_expired"
        assert stored == [], "a hollow refresh must not touch the vault"


# ---------------------------------------------------------------------------
# Read-merge-write — an unread blob must never be written back
# ---------------------------------------------------------------------------


class TestCredentialMergeGuards:
    @pytest.mark.asyncio
    async def test_update_refuses_to_merge_into_an_unread_blob(
        self, monkeypatch,
    ) -> None:
        """Merging into ``{}`` and storing it overwrites every field we failed to
        read — silently destroying the patron's access and refresh tokens."""
        rt = _oauth_runtime()
        stored: list[dict] = []

        async def _cold(service, npub_override=None):
            return {}, "vault_bootstrapping"

        async def _store(npub, data, service=None):
            stored.append(data)
            return True

        monkeypatch.setattr(rt, "_load_vault_creds", _cold)
        monkeypatch.setattr(rt, "store_patron_session", _store)

        ok = await rt.update_patron_credential(VALID_NPUB, "account_hash", "H")
        assert ok is False
        assert stored == []

    @pytest.mark.asyncio
    async def test_delete_refuses_on_an_unread_blob(self, monkeypatch) -> None:
        rt = _oauth_runtime()
        stored: list[dict] = []

        async def _cold(service, npub_override=None):
            return {}, "vault_bootstrapping"

        async def _store(npub, data, service=None):
            stored.append(data)
            return True

        monkeypatch.setattr(rt, "_load_vault_creds", _cold)
        monkeypatch.setattr(rt, "store_patron_session", _store)

        ok = await rt.delete_patron_credential(VALID_NPUB, "account_hash")
        assert ok is False
        assert stored == []


# ---------------------------------------------------------------------------
# The situation table renders every situation the producer can now emit
# ---------------------------------------------------------------------------


class TestSituationsAreAllRendered:
    @pytest.mark.parametrize(
        ("situation", "expected_code"),
        [
            ("vault_bootstrapping", "warming_up"),
            ("secure_courier_unavailable", "secure_courier_unavailable"),
            ("persistence_quota_exceeded", "persistence_quota_exceeded"),
            ("persistence_misconfigured", "persistence_misconfigured"),
            ("operator_not_configured", "operator_credentials_missing"),
        ],
    )
    def test_no_producible_situation_falls_through_to_unknown(
        self, situation: str, expected_code: str,
    ) -> None:
        rt = _oauth_runtime()
        response = rt.oauth_situation_response(situation)
        assert response["error_code"] == expected_code
        assert response["error_code"] != "oauth_situation_unknown"

    def test_non_retryable_situations_do_not_advise_retrying(self) -> None:
        """A quota ceiling and a permission fault outlast any amount of patience;
        telling someone to retry sends them to wait out an endless outage."""
        rt = _oauth_runtime()
        for situation in ("persistence_quota_exceeded", "persistence_misconfigured"):
            steps = " ".join(rt.oauth_situation_response(situation)["next_steps"]).lower()
            assert "retrying will not help" in steps or "not by retrying" in steps or (
                "notify the operator" in steps
            )
