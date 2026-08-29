"""A relay that refused us is reported, so ordering is measured and not guessed.

The curated relay order in `relays.json` is a starting guess, and a guess cannot
know what is up this minute. On 2026-08-29 an eXcalibur Modal container found all
four of its bootstrap relays unreachable, could not read its own vault, and the
scheduler published an author's fallback text instead of the resolve a patron had
paid for. Nothing told the Oracle, so the same order was handed to the next
container, and the one after that.

A detached runner is the fleet's best-placed reporter: it cold-boots and
re-bootstraps on EVERY job, so it meets the weather as it actually is, where a
warm front bootstraps once and never looks again.

The report is not an assertion of rank. The Oracle probes the relay itself and its
own measurement decides — so these tests care that we report the RIGHT relays,
honestly, and never at the cost of the caller's own verdict.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from nostr_sdk import Event, Keys

from tollbooth.bootstrap import BootstrapClient

# A real key: the Oracle verifies the Schnorr signature and matches it to the
# reporter's npub, so a placeholder would test nothing that the Oracle checks.
KEYS = Keys.generate()
NSEC_HEX = KEYS.secret_key().to_hex()

PRIMAL = "wss://relay.primal.net"
DAMUS = "wss://relay.damus.io"
NOS = "wss://nos.lol"
RELAYS = [NOS, DAMUS, PRIMAL]

# The shape `receive_bootstrap_config` actually returns: nos.lol and damus.io
# refused; primal answered and simply held nothing for us.
DIAG_TWO_FAILED = (
    f"relays=3, events=0, errors=[{NOS}: Connection timed out; "
    f"{DAMUS}: Handshake status 503 Service Unavailable]"
)


def _client() -> BootstrapClient:
    return BootstrapClient(nsec_hex=NSEC_HEX)


def _oracle(answer=None, side_effect=None):
    """Patch the Oracle client the helper resolves at call time."""
    oracle = AsyncMock()
    oracle.report_relay_failure = AsyncMock(
        return_value=answer or {"success": True, "accepted": True, "probed": "unreachable"},
        side_effect=side_effect,
    )
    return patch("tollbooth.oracle_client.default_oracle_client", return_value=oracle), oracle


class TestWhichRelaysGetReported:
    @pytest.mark.asyncio
    async def test_a_reachable_relay_holding_nothing_is_never_reported(self):
        """THE LOAD-BEARING NEGATIVE ASSERTION.

        `events=0` is not evidence a relay is down — primal answered, it just had
        no config for us. Reporting it would ask the Oracle to spend a probe
        demoting a healthy relay over our own empty mailbox, and would turn "this
        operator was never registered" into "the relay set is broken".
        """
        ctx, oracle = _oracle()
        with ctx:
            await _client()._report_unreachable_relays(RELAYS, DIAG_TWO_FAILED)

        reported = {c.kwargs["relay"] for c in oracle.report_relay_failure.await_args_list}
        assert reported == {NOS, DAMUS}
        assert PRIMAL not in reported, "a relay that answered must never be reported"

    @pytest.mark.asyncio
    async def test_nothing_is_reported_when_no_relay_errored(self):
        ctx, oracle = _oracle()
        with ctx:
            await _client()._report_unreachable_relays(RELAYS, "relays=3, events=0")
        oracle.report_relay_failure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_same_relay_is_not_reported_twice_in_one_process(self):
        """A container re-bootstraps per job; the Oracle should not re-probe on
        each one for a relay we have already named."""
        ctx, oracle = _oracle()
        client = _client()
        with ctx:
            await client._report_unreachable_relays(RELAYS, DIAG_TWO_FAILED)
            await client._report_unreachable_relays(RELAYS, DIAG_TWO_FAILED)
        assert oracle.report_relay_failure.await_count == 2  # not 4


class TestTheReportSatisfiesTheOracle:
    @pytest.mark.asyncio
    async def test_the_signed_event_verifies_and_names_its_relay(self):
        """Everything the Oracle checks, checked here — otherwise this ships a
        report that is refused in production and silently logged at debug."""
        ctx, oracle = _oracle()
        with ctx:
            await _client()._report_unreachable_relays([NOS], DIAG_TWO_FAILED)

        kw = oracle.report_relay_failure.await_args.kwargs
        event = Event.from_json(kw["signed_event"])
        event.verify()  # raises if the Schnorr signature is bad
        assert kw["relay"] == NOS
        assert kw["reporter_npub"] == KEYS.public_key().to_bech32()
        assert event.author().to_hex() == KEYS.public_key().to_hex()
        assert NOS in event.content(), "the Oracle requires the content to name the relay"
        assert kw["mode"] == "read"

    @pytest.mark.asyncio
    async def test_each_relay_gets_its_own_signature(self):
        """One blob replayed across relays is exactly what the Oracle's
        content check exists to refuse."""
        ctx, oracle = _oracle()
        with ctx:
            await _client()._report_unreachable_relays(RELAYS, DIAG_TWO_FAILED)

        calls = oracle.report_relay_failure.await_args_list
        for c in calls:
            assert c.kwargs["relay"] in Event.from_json(c.kwargs["signed_event"]).content()


class TestReportingIsNeverLoadBearing:
    @pytest.mark.asyncio
    async def test_an_unreachable_oracle_does_not_raise(self):
        """This runs on a path that has ALREADY failed. The caller still has a
        verdict to return, and a failed report must not cost it that."""
        ctx, _ = _oracle(side_effect=RuntimeError("oracle down"))
        with ctx:
            await _client()._report_unreachable_relays(RELAYS, DIAG_TWO_FAILED)  # no raise

    @pytest.mark.asyncio
    async def test_a_refusal_the_oracle_can_articulate_is_not_an_error(self):
        """Non-members are ignored rather than errored, by the Oracle's design."""
        ctx, oracle = _oracle(answer={"success": True, "accepted": False, "note": "not a member"})
        with ctx:
            await _client()._report_unreachable_relays([NOS], DIAG_TWO_FAILED)
        oracle.report_relay_failure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_broken_signing_stack_is_survivable(self):
        with patch("tollbooth.oracle_client.default_oracle_client",
                   side_effect=ImportError("no nostr_sdk")):
            await _client()._report_unreachable_relays(RELAYS, DIAG_TWO_FAILED)  # no raise
