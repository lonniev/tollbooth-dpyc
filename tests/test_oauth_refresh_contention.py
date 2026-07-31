"""What happens when a refresh token is asked to do two things at once.

Providers that honor OAuth2 refresh-token rotation — X among them — issue a new
refresh token on every use and retire the presented one. So the interesting
failure here is not a slow endpoint; it is *arithmetic*. An operator whose
scheduler launches one background job per due post fires them together, so four
posts due at 10:30 produce four simultaneous refreshes of one token: one wins,
three are told ``invalid_grant``, and three owners are advised to reconnect an
account that was never disconnected. A strict provider goes further and revokes
the grant it saw replayed, which turns the advice into a self-fulfilling one.

Five properties are asserted here, and they are the whole fix:

1. **Concurrency collapses to one refresh.** N callers, one HTTP exchange, and
   the waiters read the winner's token rather than spending a retired one.
2. **Only the provider may declare a session dead.** A timeout, a 429, a 5xx —
   none of them are evidence about the grant, and none may reach a patron as
   "your access expired".
3. **A rejected access token renews itself.** A 401 from the upstream API is
   about the short-lived half of the grant; retiring the cached expiry lets the
   next call spend the refresh token instead of stranding the patron behind a
   cache that keeps insisting the dead token is good.
4. **A refusal names its culprit.** RFC 6749 §5.2 has four refusals and they do
   not share one. ``invalid_client`` is the operator's app credentials;
   ``invalid_request`` is our own bug; only ``invalid_grant`` is the patron's
   authorization. Reported as one code, an operator rotating an app secret
   watched every patron get told their session expired.
5. **A lost renewal is remembered, and named.** On a rotating provider a renewal
   whose answer never arrives leaves the provider holding a replacement we never
   saw and us holding a spent token. Every later attempt draws ``invalid_grant``,
   so the grant reads as though it aged out — which is how eXcalibur's operator
   came to reconnect X daily on advice that never mentioned the real cause.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, ClassVar

import httpx
import pytest

from tollbooth.oauth_config import OAuthProviderConfig
from tollbooth.oauth_situation import OAuthSituation
from tollbooth.runtime import OperatorRuntime

_TEST_NSEC = "nsec1test000000000000000000000000000000000000000000000000000000"
os.environ.setdefault("TOLLBOOTH_NOSTR_OPERATOR_NSEC", _TEST_NSEC)

NPUB_A = "npub1" + "a" * 58
NPUB_B = "npub1" + "b" * 58


def _runtime(*, leeway: int = 300) -> OperatorRuntime:
    return OperatorRuntime(
        tool_registry={},
        service_name="Test Operator",
        oauth_provider=OAuthProviderConfig(
            authorize_url="https://provider.test/authorize",
            token_url="https://provider.test/token",
            service_name="testsvc",
            client_id_field="app_key",
            client_secret_field="app_secret",
            refresh_enabled=True,
            refresh_leeway_seconds=leeway,
        ),
    )


class _RotatingProvider:
    """A provider that rotates single-use refresh tokens, and says so.

    Presenting a token it has already retired earns ``OAuthRefreshDenied`` —
    exactly what X answers, and exactly what the old code reported to every
    loser of the race as "your session expired".
    """

    def __init__(self, *, live: str = "r1", delay: float = 0.02) -> None:
        self._live = live
        self._spent: set[str] = set()
        self._delay = delay
        self.calls: list[str] = []
        self.serial = 0

    async def __call__(self, client_id, client_secret, refresh_token, token_url):
        from tollbooth.oauth2_collector import OAuthRefreshDenied

        self.calls.append(refresh_token)
        await asyncio.sleep(self._delay)  # a real network hop; widens the race
        if refresh_token in self._spent or refresh_token != self._live:
            raise OAuthRefreshDenied(
                "400 invalid_grant: Value passed for the refresh token was invalid.",
                status_code=400,
                oauth_error="invalid_grant",
            )
        self._spent.add(refresh_token)
        self.serial += 1
        self._live = f"r{self.serial + 1}"
        return {
            "access_token": f"access-{self.serial}",
            "refresh_token": self._live,
            "expires_at": time.time() + 7200,
            "token_type": "Bearer",
        }


class _FakeVault:
    """The patron/operator credential reads and writes, in a dict."""

    def __init__(self, runtime: OperatorRuntime, sessions: dict[str, dict[str, str]]):
        self.sessions = sessions
        self.reads = 0
        self.writes = 0
        runtime.load_patron_session = self.load  # type: ignore[method-assign]
        runtime.store_patron_session = self.store  # type: ignore[method-assign]
        runtime._load_vault_creds = self.load_operator  # type: ignore[method-assign]

    async def load(self, npub, service=None):
        self.reads += 1
        row = self.sessions.get(npub)
        return (dict(row) if row else None), ""

    async def store(self, npub, data, service=None):
        self.writes += 1
        self.sessions[npub] = dict(data)
        return True

    async def load_operator(self, service, npub_override=None):
        return {"app_key": "cid", "app_secret": "csec"}, ""


def _stale_session(refresh_token: str = "r1") -> dict[str, str]:
    """A vaulted session whose access token has already lapsed."""
    return {
        "access_token": "access-0",
        "refresh_token": refresh_token,
        "expires_at": str(time.time() - 10),
    }


# ---------------------------------------------------------------------------
# 1. Concurrency collapses to one refresh
# ---------------------------------------------------------------------------


class TestConcurrentRefresh:
    @pytest.mark.asyncio
    async def test_four_simultaneous_restores_perform_one_refresh(self, monkeypatch):
        """The scheduler's fan-out, reproduced: four publishers, one token.

        Before the refresh lock this issued four refresh POSTs, and the three
        that lost held their posts for ``oauth_token_expired``.
        """
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _stale_session()})
        provider = _RotatingProvider()
        monkeypatch.setattr(collector, "refresh_access_token", provider)

        results = await asyncio.gather(
            *(rt.restore_oauth_session(NPUB_A) for _ in range(4)),
        )

        assert len(provider.calls) == 1, (
            f"one token, one refresh — got {provider.calls}"
        )
        assert [situation for _, situation in results] == [None] * 4
        assert {creds["access_token"] for creds, _ in results} == {"access-1"}
        assert vault.sessions[NPUB_A]["refresh_token"] == "r2"  # rotation persisted
        assert vault.writes == 1

    @pytest.mark.asyncio
    async def test_a_waiter_reads_the_winners_token_without_calling_out(self, monkeypatch):
        """The waiters must not merely survive — they must skip the refresh.

        Asserted through the vault: each waiter re-reads the session behind the
        lock, finds it comfortably fresh, and returns it.
        """
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        _FakeVault(rt, {NPUB_A: _stale_session()})
        provider = _RotatingProvider(delay=0.05)
        monkeypatch.setattr(collector, "refresh_access_token", provider)

        first, second = await asyncio.gather(
            rt.restore_oauth_session(NPUB_A),
            rt.restore_oauth_session(NPUB_A),
        )

        assert provider.serial == 1
        assert first[0]["access_token"] == second[0]["access_token"] == "access-1"
        assert first[1] is second[1] is None

    @pytest.mark.asyncio
    async def test_two_patrons_refresh_in_parallel(self, monkeypatch):
        """The lock is per patron. One patron's refresh must not queue another's."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        _FakeVault(rt, {
            NPUB_A: _stale_session("a-r1"),
            NPUB_B: _stale_session("b-r1"),
        })

        in_flight, peak = 0, 0

        async def _refresh(client_id, client_secret, refresh_token, token_url):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return {
                "access_token": f"new-{refresh_token}",
                "refresh_token": f"{refresh_token}-next",
                "expires_at": time.time() + 7200,
            }

        monkeypatch.setattr(collector, "refresh_access_token", _refresh)

        results = await asyncio.gather(
            rt.restore_oauth_session(NPUB_A),
            rt.restore_oauth_session(NPUB_B),
        )

        assert peak == 2, "distinct patrons must refresh concurrently"
        assert [s for _, s in results] == [None, None]

    @pytest.mark.asyncio
    async def test_a_grant_the_provider_refuses_is_still_reported_expired(self, monkeypatch):
        """The lock must not paper over a genuinely dead grant."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _stale_session("retired-token")})
        provider = _RotatingProvider(live="some-other-token")
        monkeypatch.setattr(collector, "refresh_access_token", provider)

        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert creds is None
        assert situation.code == "token_expired"
        assert vault.writes == 0


# ---------------------------------------------------------------------------
# 2. Only the provider may declare a session dead
# ---------------------------------------------------------------------------


class TestTransientRefreshFailures:
    @pytest.mark.asyncio
    async def test_unreachable_endpoint_is_not_an_expired_session(self, monkeypatch):
        """A refresh that never completed says nothing about the grant.

        This is the path that cost real posts: a five-second default timeout
        against a provider's auth host, reported to the owner as expired access.
        """
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _stale_session()})

        async def _unreachable(*args, **kwargs):
            raise collector.OAuthRefreshUnavailable(
                "token endpoint unreachable: ConnectTimeout",
            )

        monkeypatch.setattr(collector, "refresh_access_token", _unreachable)

        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert creds is None
        assert situation.code == "refresh_unavailable"
        assert vault.writes == 0, "a failed refresh must not touch the vault"

    def test_the_transient_situation_never_advises_reconnecting(self):
        """``next_steps`` is the part a patron acts on, so it carries the fix."""
        from tollbooth.constants import ErrorCode

        rt = _runtime()
        response = rt.oauth_situation_response(OAuthSituation("refresh_unavailable"))

        assert response["error_code"] == ErrorCode.OAUTH_REFRESH_UNAVAILABLE
        assert response["success"] is False
        steps = " ".join(response["next_steps"]).lower()
        assert "no re-authentication needed" in steps
        assert "begin_oauth" not in steps

    def test_a_refused_grant_still_advises_reconnecting(self):
        """The distinction is only worth drawing if the other side still works."""
        rt = _runtime()
        steps = " ".join(
            rt.oauth_situation_response(OAuthSituation("token_expired"))["next_steps"]
        )
        assert "begin_oauth" in steps

    @pytest.mark.asyncio
    async def test_an_unreadable_expiry_does_not_abort_the_restore(self, monkeypatch):
        """A token exchange whose response omitted ``expires_in`` vaults
        ``expires_at: ""``. A bare float() on that raises, and the restore fails
        with a string-conversion error instead of refreshing."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        session = _stale_session()
        session["expires_at"] = ""
        _FakeVault(rt, {NPUB_A: session})
        monkeypatch.setattr(collector, "refresh_access_token", _RotatingProvider())

        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert situation is None
        assert creds["access_token"] == "access-1"


# ---------------------------------------------------------------------------
# 3. A rejected ACCESS token is not a dead grant
# ---------------------------------------------------------------------------


def _fresh_session(refresh_token: str = "r1") -> dict[str, str]:
    """A vaulted session whose access token our records still call good."""
    return {
        "access_token": "access-0",
        "refresh_token": refresh_token,
        "expires_at": str(time.time() + 7200),
    }


class TestRejectedAccessToken:
    """The upstream API says 401 while the vault says the token is fine.

    The access token is the short-lived half of the grant, and the refresh
    token is right there — so this is a renewal, not a re-authorization. What
    made it a re-authorization was the *stored expiry*: it kept asserting the
    dead token was good, so every retry short-circuited on the cache and failed
    identically until the real expiry hours later.
    """

    @pytest.mark.asyncio
    async def test_invalidating_makes_the_next_restore_actually_refresh(
        self, monkeypatch,
    ):
        """The whole point: the cached-token shortcut must stop firing."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _fresh_session()})
        provider = _RotatingProvider()
        monkeypatch.setattr(collector, "refresh_access_token", provider)

        # Precondition — without invalidation the cache serves and nobody calls out.
        creds, situation = await rt.restore_oauth_session(NPUB_A)
        assert (creds["access_token"], situation) == ("access-0", None)
        assert provider.calls == [], "a fresh cached token must not be refreshed"

        assert await rt.invalidate_oauth_access_token(NPUB_A) is True

        creds, situation = await rt.restore_oauth_session(NPUB_A)
        assert situation is None
        assert creds["access_token"] == "access-1", "the rejected token was renewed"
        assert provider.calls == ["r1"], "exactly one refresh, spending the live token"
        assert vault.sessions[NPUB_A]["refresh_token"] == "r2"

    @pytest.mark.asyncio
    async def test_invalidating_keeps_the_refresh_token(self):
        """Retire the expiry and nothing else — the refresh token IS the cure.

        Clobbering it here would convert a self-healing 401 into the dead grant
        the patron was wrongly told they already had.
        """
        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _fresh_session()})

        await rt.invalidate_oauth_access_token(NPUB_A)

        stored = vault.sessions[NPUB_A]
        assert stored["refresh_token"] == "r1"
        assert stored["access_token"] == "access-0"
        assert float(stored["expires_at"]) <= 0

    @pytest.mark.asyncio
    async def test_a_grant_that_really_is_dead_still_says_so(self, monkeypatch):
        """Renewal is an attempt, not an excuse — if the provider refuses the
        refresh token, the patron does need to reconnect and must be told."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        _FakeVault(rt, {NPUB_A: _fresh_session(refresh_token="retired")})
        monkeypatch.setattr(collector, "refresh_access_token", _RotatingProvider())

        await rt.invalidate_oauth_access_token(NPUB_A)
        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert creds is None
        assert situation.code == "token_expired"

    @pytest.mark.asyncio
    async def test_nothing_to_invalidate_is_not_a_failure(self):
        """No vaulted session means the next call re-resolves from scratch
        anyway, so there is nothing to retire and nothing to report."""
        rt = _runtime()
        _FakeVault(rt, {})

        assert await rt.invalidate_oauth_access_token(NPUB_A) is False

    def test_the_rejected_situation_never_advises_reconnecting(self):
        """The failure this whole section exists to prevent."""
        from tollbooth.constants import ErrorCode

        rt = _runtime()
        response = rt.oauth_situation_response(OAuthSituation("token_rejected"))

        assert response["error_code"] == ErrorCode.OAUTH_TOKEN_REJECTED
        assert response["success"] is False
        steps = " ".join(response["next_steps"]).lower()
        assert "begin_oauth" not in steps
        assert "repeat your request" in steps


# ---------------------------------------------------------------------------
# 4. The collector's own reading of a token-endpoint answer
# ---------------------------------------------------------------------------


class _StubClient:
    """Stands in for httpx.AsyncClient, recording how it was configured."""

    instances: ClassVar[list[Any]] = []

    def __init__(self, *, timeout=None, **kwargs):
        self.timeout = timeout
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        return await type(self).handler(url, **kwargs)


def _install_stub(monkeypatch, handler) -> type[_StubClient]:
    stub = type("Stub", (_StubClient,), {"handler": staticmethod(handler)})
    stub.instances = []
    monkeypatch.setattr(httpx, "AsyncClient", stub)
    return stub


def _token_response(status: int, body: Any) -> httpx.Response:
    request = httpx.Request("POST", "https://provider.test/token")
    if isinstance(body, str):
        return httpx.Response(status_code=status, text=body, request=request)
    return httpx.Response(status_code=status, json=body, request=request)


async def _refresh(monkeypatch, handler):
    """Run one refresh against *handler*, returning ``(token, stub_client_class)``."""
    from tollbooth.oauth2_collector import refresh_access_token

    stub = _install_stub(monkeypatch, handler)
    token = await refresh_access_token(
        "cid", "csec", "r1", "https://provider.test/token",
    )
    return token, stub


class TestTokenEndpointClassification:
    @pytest.mark.asyncio
    async def test_invalid_grant_is_denied(self, monkeypatch):
        from tollbooth.oauth2_collector import OAuthRefreshDenied

        async def _handler(url, **kwargs):
            return _token_response(400, {
                "error": "invalid_grant",
                "error_description": "Value passed for the refresh token was invalid.",
            })

        with pytest.raises(OAuthRefreshDenied) as caught:
            await _refresh(monkeypatch, _handler)
        assert caught.value.oauth_error == "invalid_grant"
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    async def test_provider_asking_for_patience_is_not_a_dead_grant(
        self, monkeypatch, status,
    ):
        """A 429 or 5xx is the provider being busy. Reading it as a revoked
        grant is how an outage became a fleet of re-authorization prompts."""
        from tollbooth.oauth2_collector import OAuthRefreshUnavailable

        async def _handler(url, **kwargs):
            return _token_response(status, {"title": "Too Many Requests"})

        with pytest.raises(OAuthRefreshUnavailable) as caught:
            await _refresh(monkeypatch, _handler)
        assert caught.value.status_code == status

    @pytest.mark.asyncio
    async def test_an_unnamed_4xx_is_not_assumed_fatal(self, monkeypatch):
        """A 4xx with no OAuth2 error code is likelier a gateway artifact than a
        considered refusal — we don't guess a patron into re-authorizing."""
        from tollbooth.oauth2_collector import OAuthRefreshUnavailable

        async def _handler(url, **kwargs):
            return _token_response(403, "<html>Forbidden</html>")

        with pytest.raises(OAuthRefreshUnavailable):
            await _refresh(monkeypatch, _handler)

    @pytest.mark.asyncio
    async def test_a_read_timeout_admits_the_token_may_be_spent(self, monkeypatch):
        """The request arrived; only the answer was lost. If this provider
        rotates single-use tokens, ours may already be retired — say so rather
        than reporting a clean failure."""
        from tollbooth.oauth2_collector import OAuthRefreshUnavailable

        async def _handler(url, **kwargs):
            raise httpx.ReadTimeout("timed out waiting for the answer")

        with pytest.raises(OAuthRefreshUnavailable) as caught:
            await _refresh(monkeypatch, _handler)
        assert caught.value.token_may_have_rotated is True

    @pytest.mark.asyncio
    async def test_an_unopened_connection_is_retried_then_reported_clean(
        self, monkeypatch,
    ):
        """A connect failure provably never reached the provider, so retrying is
        safe and the stored token is provably untouched."""
        import tollbooth.oauth2_collector as collector

        attempts = {"n": 0}

        async def _handler(url, **kwargs):
            attempts["n"] += 1
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(collector, "_TOKEN_CONNECT_BACKOFF_S", 0)
        with pytest.raises(collector.OAuthRefreshUnavailable) as caught:
            await _refresh(monkeypatch, _handler)

        assert attempts["n"] == collector._TOKEN_CONNECT_ATTEMPTS
        assert caught.value.token_may_have_rotated is False

    @pytest.mark.asyncio
    async def test_the_token_endpoint_gets_more_than_the_bare_default(self, monkeypatch):
        """httpx's bare default allows 5s per phase. The consequence of losing
        that race is not a slow page — it is a patron told to reconnect."""
        from tollbooth.oauth2_collector import TOKEN_ENDPOINT_TIMEOUT

        async def _handler(url, **kwargs):
            return _token_response(200, {
                "access_token": "a", "refresh_token": "r2", "expires_in": 7200,
            })

        token, stub = await _refresh(monkeypatch, _handler)

        assert token["access_token"] == "a"
        assert stub.instances[0].timeout is TOKEN_ENDPOINT_TIMEOUT
        assert TOKEN_ENDPOINT_TIMEOUT.connect > 5.0
        assert TOKEN_ENDPOINT_TIMEOUT.read > 5.0

    @pytest.mark.asyncio
    async def test_a_provider_cannot_echo_the_token_into_our_logs(self, monkeypatch):
        """The refusal detail is logged. A provider that quotes the rejected
        grant back at us must not thereby write a live credential to disk."""
        from tollbooth.oauth2_collector import OAuthRefreshDenied

        async def _handler(url, **kwargs):
            return _token_response(400, {
                "error": "invalid_grant",
                "error_description": "refresh token r1-secret-value is invalid",
            })

        stub = _install_stub(monkeypatch, _handler)
        assert stub is not None
        from tollbooth.oauth2_collector import refresh_access_token

        with pytest.raises(OAuthRefreshDenied) as caught:
            await refresh_access_token(
                "cid", "csec", "r1-secret-value", "https://provider.test/token",
            )

        assert "r1-secret-value" not in caught.value.detail
        assert "<redacted>" in caught.value.detail

    @pytest.mark.asyncio
    async def test_expires_at_is_derived_from_expires_in(self, monkeypatch):
        async def _handler(url, **kwargs):
            return _token_response(200, {"access_token": "a", "expires_in": 7200})

        before = time.time()
        token, _ = await _refresh(monkeypatch, _handler)
        assert before + 7200 <= token["expires_at"] <= time.time() + 7200

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("oauth_error", "fault"), [
        ("invalid_grant", "grant"),
        ("invalid_client", "client"),
        ("unauthorized_client", "client"),
        ("invalid_request", "request"),
    ])
    async def test_a_refusal_names_whose_fault_it_is(
        self, monkeypatch, oauth_error, fault,
    ):
        """All four are terminal; only one of them is the patron's to fix."""
        from tollbooth.oauth2_collector import OAuthRefreshDenied

        async def _handler(url, **kwargs):
            return _token_response(400, {"error": oauth_error})

        with pytest.raises(OAuthRefreshDenied) as caught:
            await _refresh(monkeypatch, _handler)
        assert caught.value.fault == fault


# ---------------------------------------------------------------------------
# 5. A refusal reaches the right person, and a lost renewal is named
# ---------------------------------------------------------------------------


class _Refusing:
    """A provider that always refuses, with the error code it was given."""

    def __init__(self, oauth_error: str, fault: str) -> None:
        self.oauth_error = oauth_error
        self.fault = fault

    async def __call__(self, client_id, client_secret, refresh_token, token_url):
        from tollbooth.oauth2_collector import OAuthRefreshDenied

        raise OAuthRefreshDenied(
            f"400 {self.oauth_error}: refused",
            status_code=400, oauth_error=self.oauth_error, fault=self.fault,
        )


class TestRefusalIsAttributed:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("oauth_error", "fault", "expected"), [
        ("invalid_client", "client", "operator_app_credentials_rejected"),
        ("unauthorized_client", "client", "operator_app_credentials_rejected"),
        ("invalid_request", "request", "refresh_request_malformed"),
        ("invalid_grant", "grant", "token_expired"),
    ])
    async def test_the_situation_follows_the_fault(
        self, monkeypatch, oauth_error, fault, expected,
    ):
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        _FakeVault(rt, {NPUB_A: _stale_session()})
        monkeypatch.setattr(
            collector, "refresh_access_token", _Refusing(oauth_error, fault),
        )

        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert creds is None
        assert situation.code == expected
        assert situation.oauth_error == oauth_error
        assert situation.status_code == 400

    @pytest.mark.parametrize("code", [
        "operator_app_credentials_rejected",
        "refresh_request_malformed",
        "refresh_failed_unclassified",
    ])
    def test_a_fault_that_is_not_the_patrons_never_advises_reconnecting(self, code):
        """The failure this section exists to prevent. Re-authorizing against a
        provider that is refusing the operator's app can only burn a live grant."""
        rt = _runtime()
        steps = " ".join(
            rt.oauth_situation_response(OAuthSituation(code))["next_steps"]
        ).lower()
        assert "begin_oauth" not in steps

    @pytest.mark.asyncio
    async def test_an_unclassified_failure_is_not_reported_as_expired(
        self, monkeypatch,
    ):
        """It used to be. An unknown cause dressed as a dead grant is an
        instruction to throw away a working authorization on a guess."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _stale_session()})

        async def _boom(*args, **kwargs):
            raise RuntimeError("something nobody anticipated")

        monkeypatch.setattr(collector, "refresh_access_token", _boom)

        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert creds is None
        assert situation.code == "refresh_failed_unclassified"
        assert "RuntimeError" in situation.detail
        assert vault.sessions[NPUB_A]["refresh_token"] == "r1", "vault untouched"

    @pytest.mark.asyncio
    async def test_a_session_with_no_refresh_token_says_so(self):
        """No refresh token is a scope problem, not a clock problem. Called an
        expiry, it sends the patron to reconnect into the identical grant."""
        rt = _runtime()
        _FakeVault(rt, {NPUB_A: {
            "access_token": "access-0", "expires_at": str(time.time() - 10),
        }})

        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert creds is None
        assert situation.code == "no_refresh_token"


class TestLostRenewal:
    """The one-way door: a renewal that arrives but whose answer does not."""

    @staticmethod
    def _timing_out():
        async def _handler(*args, **kwargs):
            from tollbooth.oauth2_collector import OAuthRefreshUnavailable

            raise OAuthRefreshUnavailable(
                "token endpoint did not answer: ReadTimeout",
                token_may_have_rotated=True,
            )
        return _handler

    @pytest.mark.asyncio
    async def test_a_lost_answer_is_written_down(self, monkeypatch):
        """We cannot know whether the token was spent — but we can record the
        moment, which is the only way a later refusal can cite it."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _stale_session()})
        monkeypatch.setattr(collector, "refresh_access_token", self._timing_out())

        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert creds is None
        assert situation.code == "refresh_unavailable", "still not a dead grant"
        assert vault.sessions[NPUB_A]["refresh_lost_at"], "the moment was recorded"
        assert vault.sessions[NPUB_A]["refresh_token"] == "r1", "credential untouched"

    @pytest.mark.asyncio
    async def test_a_connect_failure_is_not_written_down(self, monkeypatch):
        """A connection that never opened provably never reached the provider,
        so nothing was rotated and there is no suspicion to record."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _stale_session()})

        async def _unreachable(*args, **kwargs):
            raise collector.OAuthRefreshUnavailable("unreachable: ConnectTimeout")

        monkeypatch.setattr(collector, "refresh_access_token", _unreachable)

        await rt.restore_oauth_session(NPUB_A)

        assert "refresh_lost_at" not in vault.sessions[NPUB_A]

    @pytest.mark.asyncio
    async def test_the_later_refusal_blames_the_lost_answer_not_the_clock(
        self, monkeypatch,
    ):
        """The whole point, end to end: timeout, then `invalid_grant`, and the
        report names the timeout instead of inventing an expiry."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _stale_session()})

        monkeypatch.setattr(collector, "refresh_access_token", self._timing_out())
        await rt.restore_oauth_session(NPUB_A)
        lost_at = vault.sessions[NPUB_A]["refresh_lost_at"]

        monkeypatch.setattr(
            collector, "refresh_access_token",
            _Refusing("invalid_grant", "grant"),
        )
        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert creds is None
        assert situation.code == "refresh_token_lost"
        assert situation.observed_at == lost_at
        assert lost_at in situation.detail

        response = rt.oauth_situation_response(situation)
        assert response["error_code"] == "oauth_refresh_token_lost"
        # A reconnect IS the fix here — what changes is that it comes with a
        # reason, so an operator living through it daily can see the pattern.
        assert "begin_oauth" in " ".join(response["next_steps"])
        assert lost_at in response["error"]

    @pytest.mark.asyncio
    async def test_the_first_loss_is_the_one_remembered(self, monkeypatch):
        """A second timeout must not overwrite the moment the token was spent —
        the first loss is the culprit; later ones are its symptoms."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _stale_session()})
        monkeypatch.setattr(collector, "refresh_access_token", self._timing_out())

        await rt.restore_oauth_session(NPUB_A)
        first = vault.sessions[NPUB_A]["refresh_lost_at"]
        await rt.restore_oauth_session(NPUB_A)

        assert vault.sessions[NPUB_A]["refresh_lost_at"] == first

    @pytest.mark.asyncio
    async def test_a_successful_refresh_forgets_the_suspicion(self, monkeypatch):
        """The renewal went through after all, so the earlier lost answer did
        not cost us the grant. Leaving the marker would misattribute a genuine
        expiry months later to a timeout that turned out to be harmless."""
        import tollbooth.oauth2_collector as collector

        rt = _runtime()
        vault = _FakeVault(rt, {NPUB_A: _stale_session()})

        monkeypatch.setattr(collector, "refresh_access_token", self._timing_out())
        await rt.restore_oauth_session(NPUB_A)
        assert vault.sessions[NPUB_A]["refresh_lost_at"]

        monkeypatch.setattr(collector, "refresh_access_token", _RotatingProvider())
        creds, situation = await rt.restore_oauth_session(NPUB_A)

        assert situation is None
        assert creds["access_token"] == "access-1"
        assert "refresh_lost_at" not in vault.sessions[NPUB_A]
