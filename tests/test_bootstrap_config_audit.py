"""The weekly bootstrap-config audit runs in the foreground and heals what it finds.

2026-09-29: nos.lol dropped months-old config events and left four operators with
no copy on any relay. The Authority's refresh had been a background task, and
Horizon freezes a process between requests, so it rarely finished. The audit is
now a tool a weekly GitHub Action calls: it measures coverage, republishes any
config held by fewer than three relays or sent more than six days ago, and
measures again — all before it answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

import tollbooth.authority.tools as at
from tollbooth.bootstrap_relay import _config_d_tag, config_coverage

AUTH = PrivateKey()
THIN, STALE, FRESH = (PrivateKey().public_key for _ in range(3))
NPUBS = {"thin": THIN.bech32(), "stale": STALE.bech32(), "fresh": FRESH.bech32()}
R = [f"wss://r{i}" for i in range(4)]
WEEK_AGO = str(int(time.time()) - 7 * 24 * 3600)
TODAY = str(int(time.time()))


# ---------------------------------------------------------------------------
# config_coverage — where the events are, never what they say
# ---------------------------------------------------------------------------


def _relay_holding(*d_tags: str, refuse: bool = False, sent: list | None = None):
    def connect(*_a, **_k):
        if refuse:
            raise ConnectionRefusedError("refused")
        ws = MagicMock()
        frames = [json.dumps(["EVENT", "s", {"tags": [["d", d]]}]) for d in d_tags]
        frames.append(json.dumps(["EOSE", "s"]))
        ws.recv.side_effect = lambda: frames.pop(0)
        ws.send.side_effect = lambda m: sent.append(json.loads(m)) if sent is not None else None
        return ws
    return connect


def test_coverage_counts_each_operator_on_each_relay_that_answers():
    a, b = THIN.hex(), FRESH.hex()
    table = {
        "wss://both": _relay_holding(_config_d_tag(a), _config_d_tag(b)),
        "wss://only-b": _relay_holding(_config_d_tag(b)),
        "wss://down": _relay_holding(refuse=True),
    }
    with patch("websocket.create_connection", side_effect=lambda url, *x, **k: table[url](url)):
        got = config_coverage(AUTH.public_key.hex(), [a, b], list(table))
    assert got[a] == ("wss://both",)
    assert sorted(got[b]) == ["wss://both", "wss://only-b"]


def test_coverage_asks_only_for_the_authoritys_own_events():
    """A copy signed by anyone else must never count as coverage."""
    sent: list = []
    with patch("websocket.create_connection", side_effect=_relay_holding(sent=sent)):
        config_coverage(AUTH.public_key.hex(), [THIN.hex()], ["wss://r"])
    req = next(m for m in sent if m[0] == "REQ")[2]
    assert req["authors"] == [AUTH.public_key.hex()]
    assert req["kinds"] == [30078]


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def _vault():
    v = MagicMock()
    v._t = lambda n: f"authority.{n}"
    v._execute = AsyncMock(return_value={"rows": [{"npub": n} for n in NPUBS.values()]})
    return v


async def _audit(before: dict, after: dict, resend):
    runtime = MagicMock()
    runtime.vault = AsyncMock(return_value=_vault())
    signer = MagicMock(pubkey_hex=AUTH.public_key.hex())
    stamps = {NPUBS["thin"]: TODAY, NPUBS["stale"]: WEEK_AGO, NPUBS["fresh"]: TODAY}
    with patch.object(at, "_get_runtime", return_value=runtime), \
         patch.object(at, "_get_nostr_signer", return_value=signer), \
         patch.object(at, "_resend_bootstrap_dm", side_effect=resend) as sent, \
         patch("tollbooth.relay_registry.get_relays", return_value=R), \
         patch("tollbooth.bootstrap_relay.config_coverage", side_effect=[before, after]), \
         patch("tollbooth.authority.tenant_provisioner.get_operator_config_value",
               new=AsyncMock(side_effect=lambda _v, npub, _k: stamps[npub])):
        report = await at._audit_bootstrap_configs()
    return report, sent


def _cov(thin: int, stale: int, fresh: int) -> dict:
    return {THIN.hex(): tuple(R[:thin]), STALE.hex(): tuple(R[:stale]), FRESH.hex(): tuple(R[:fresh])}


@pytest.mark.asyncio
async def test_a_thin_or_stale_config_is_republished_and_a_fresh_well_held_one_is_left():
    report, sent = await _audit(_cov(1, 4, 4), _cov(4, 4, 4), AsyncMock(return_value=True))

    assert sorted(c.args[0] for c in sent.call_args_list) == sorted([NPUBS["thin"], NPUBS["stale"]])
    rows = {r["operator"]: r for r in report["operators"]}
    assert rows[f"{NPUBS['thin'][:16]}..."]["reason"] == "thin"
    assert rows[f"{NPUBS['thin'][:16]}..."]["holders_before"] == 1
    assert rows[f"{NPUBS['thin'][:16]}..."]["holders_after"] == 4
    assert rows[f"{NPUBS['stale'][:16]}..."]["reason"] == "stale"
    assert rows[f"{NPUBS['fresh'][:16]}..."]["republished"] is False
    assert report["still_thin"] == []


@pytest.mark.asyncio
async def test_an_operator_still_thin_after_republishing_is_named():
    report, _ = await _audit(_cov(0, 4, 4), _cov(1, 4, 4), AsyncMock(return_value=False))
    assert report["still_thin"] == [f"{NPUBS['thin'][:16]}..."]


@pytest.mark.asyncio
async def test_the_audit_finishes_its_publishing_before_it_answers():
    """The whole point: nothing is left for a frozen process to (not) finish."""
    finished: list[str] = []

    async def slow_resend(npub):
        await asyncio.sleep(0.05)
        finished.append(npub)
        return True

    await _audit(_cov(1, 4, 4), _cov(4, 4, 4), slow_resend)
    assert sorted(finished) == sorted([NPUBS["thin"], NPUBS["stale"]])


@pytest.mark.asyncio
async def test_the_report_carries_no_config_values():
    report, _ = await _audit(_cov(1, 4, 4), _cov(4, 4, 4), AsyncMock(return_value=True))
    text = json.dumps(report)
    assert "postgres" not in text and "neon" not in text.lower()
    assert all(len(r["operator"]) == 19 for r in report["operators"]), "npub prefixes only"


@pytest.mark.asyncio
async def test_nothing_due_means_one_measurement_and_no_publishing():
    runtime = MagicMock()
    runtime.vault = AsyncMock(return_value=_vault())
    with patch.object(at, "_get_runtime", return_value=runtime), \
         patch.object(at, "_get_nostr_signer", return_value=MagicMock(pubkey_hex=AUTH.public_key.hex())), \
         patch.object(at, "_resend_bootstrap_dm", new=AsyncMock()) as sent, \
         patch("tollbooth.relay_registry.get_relays", return_value=R), \
         patch("tollbooth.bootstrap_relay.config_coverage", return_value=_cov(4, 4, 4)) as cov, \
         patch("tollbooth.authority.tenant_provisioner.get_operator_config_value",
               new=AsyncMock(return_value=TODAY)):
        await at._audit_bootstrap_configs()
    sent.assert_not_called()
    assert cov.call_count == 1


# ---------------------------------------------------------------------------
# The resend itself (unchanged path, kept from the old refresh tests)
# ---------------------------------------------------------------------------


class _PublishResult:
    def __init__(self, accepted: int, rejected: int) -> None:
        self.accepted, self.rejected = accepted, rejected

    def __bool__(self) -> bool:
        return self.accepted > 0


@pytest.mark.asyncio
async def test_resend_logs_accepted_and_rejected_counts(caplog):
    runtime = MagicMock()
    runtime.vault = AsyncMock(return_value=MagicMock())
    signer = MagicMock()
    signer.nsec = "a" * 64
    with patch.object(at, "_get_runtime", return_value=runtime), \
         patch.object(at, "_get_nostr_signer", return_value=signer), \
         patch("tollbooth.authority.tenant_provisioner.get_all_operator_config",
               new=AsyncMock(return_value={"neon_database_url": "postgres://x", "schema": "op_x"})), \
         patch("tollbooth.authority.tenant_provisioner.store_operator_config", new=AsyncMock()), \
         patch.object(at.asyncio, "to_thread", new=AsyncMock(return_value=_PublishResult(3, 2))), \
         caplog.at_level(logging.INFO, logger=at.logger.name):
        ok = await at._resend_bootstrap_dm(NPUBS["thin"])
    assert ok is True
    assert "accepted=3 rejected=2" in " ".join(r.message for r in caplog.records)
