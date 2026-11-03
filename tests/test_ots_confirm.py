"""OTS snapshots run on a young container, confirm in Bitcoin, and hash plaintext.

Three defects, one feature: the background notarizer never fired in a
container booted under an hour ago; nothing ever collected the Bitcoin
attestation, so no snapshot reached ``confirmed``; and the tree hashed the
at-rest ciphertext, so no patron could recompute their own leaf.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.ots import OTSParseError, walk_timestamp
from tollbooth.runtime import OperatorRuntime
from tollbooth.tools.notarization import upgrade_pending_anchors

# Recorded fixtures, cross-checked against python-opentimestamps 0.4.5.
# A real pending receipt from https://a.pool.opentimestamps.org for DIGEST:
DIGEST = bytes.fromhex("966ba012779388e60dc7667eb379acf5e7ce82d1cffab932f1fa63df663948b8")
PENDING_RECEIPT = bytes.fromhex(
    "f008cc915afef5b53df008f0100e8f9f2287e00129790b2f49c00df30a08f1205b2033a10b34"
    "1ef8c796507841ba26f84d65b590cecb0c70d565c56f4a6e17a508f020ee14cc6e994a4e6f37"
    "4473d69fe93ef0e526275125e2046bc5876e07504ad2de08f120b744e4c7e3e14eda088dd68c"
    "a63789c3be20349f0b82190b74576503d4a429cb08f1046aea1e9af008628e1b6de0915aa600"
    "83dfe30d2ef90c8e2e2d68747470733a2f2f616c6963652e6274632e63616c656e6461722e6f"
    "70656e74696d657374616d70732e6f7267"
)
PENDING_COMMITMENT = bytes.fromhex(
    "6aea1e9ab43d70a533cf877177bf416d06f756581aba41220961e7da74188c347d61bacf628e1b6de0915aa6"
)
PENDING_URI = "https://alice.btc.calendar.opentimestamps.org"
# The OpenTimestamps project's confirmed example (hello-world.txt.ots):
CONFIRMED_MSG = bytes.fromhex("03ba204e50d126e4674c005e04d82e84c21366780af1f43bd54a37816b6ab340")
CONFIRMED_TIMESTAMP = bytes.fromhex(
    "03f1c8010100000001e482f9d32ecc3ba657b69d898010857b54457a90497982ff56f97c4ec5"
    "8e6f98010000006b483045022100b253add1d1cf90844338a475a04ff13fc9e7bd242b07762d"
    "ea07f5608b2de367022000b268ca9c3342b3769cdd062891317cdcef87aac310b6855e9d9389"
    "8ebbe8ec0121020d8e4d107d2b339b0050efdd4b4a09245aa056048f125396374ea6a2ab0709"
    "c6ffffffff026533e605000000001976a9140bf057d40fbba6744862515f5b55a2310de5772f"
    "88aca0860100000000001976a914f00688ac000000000808f120a987f716c533913c314c78e3"
    "5d35884cac943fa42cac49d2b2c69f4003f85f880808f120dec55b3487e1e3f722a49b55a778"
    "3215862785f4a3acb392846019f71dc64a9d0808f120b2ca18f485e080478e025dab3d464b41"
    "6c0e1ecb6629c9aefce8c8214d0424320808f02011b0e90661196ff4b0813c3eda141bab5e91"
    "604837bdf7a0c9df37db0e3a11980808f020c34bc1a4a1093ffd148c016b1e664742914e939e"
    "fabe4d3d356515914b26d9e20808f020c3e6e7c38c69f6af24c2be34ebac48257ede61ec0a21"
    "b9535e4443277be306460808f1200798bf8606e00024e5d5d54bf0c960f629dfb9dad6915745"
    "5b6f2652c0e8de810808f0203f9ada6d60baa244006bb0aad51448ad2fafb9d4b6487a0999cf"
    "f26b91f0f5360808f120c703019e959a8dd3faef7489bb328ba485574758e7091f01464eb658"
    "72c975c80808f020cbfefff513ff84b915e3fed6f9d799676630f8364ea2a6c7557fad94a5b5"
    "d7880808f1200be23709859913babd4460bbddf8ed213e7c8773a4b1face30f8acfdf093b705"
    "0808000588960d73d7190103f7ef15"
)
CONFIRMED_HEIGHT = 358391
BITCOIN_TAG = bytes.fromhex("0588960d73d71901")


def _upgrade_reaching_bitcoin(height: int) -> bytes:
    """A calendar's upgrade: sha256, then a Bitcoin attestation at ``height``."""
    payload = bytes([height & 0x7F | 0x80, height >> 7 & 0x7F | 0x80, height >> 14])
    return bytes([0x08, 0x00]) + BITCOIN_TAG + bytes([len(payload)]) + payload


# ── walk_timestamp ────────────────────────────────────────────────────

def test_a_pending_receipt_names_the_calendar_and_its_commitment():
    assert walk_timestamp(PENDING_RECEIPT, DIGEST) == [
        ("pending", PENDING_COMMITMENT, PENDING_URI),
    ]


def test_a_confirmed_timestamp_gives_its_block_height():
    found = [v for k, _m, v in walk_timestamp(CONFIRMED_TIMESTAMP, CONFIRMED_MSG) if k == "bitcoin"]
    assert found == [CONFIRMED_HEIGHT]


def test_the_synthetic_upgrade_reads_as_bitcoin():
    assert [v for k, _m, v in walk_timestamp(_upgrade_reaching_bitcoin(921_000), b"x") if k == "bitcoin"] == [921_000]


@pytest.mark.parametrize("data", [b"", PENDING_RECEIPT[:20], bytes([0x99]), bytes([0xF0, 0xFF, 0xFF])])
def test_a_malformed_timestamp_is_refused(data):
    with pytest.raises(OTSParseError):
        walk_timestamp(data, DIGEST)


# ── upgrade_pending_anchors ───────────────────────────────────────────

def _vault_with_anchor(receipts):
    vault = MagicMock()
    vault.unconfirmed_anchors = AsyncMock(return_value=[{
        "id": 7, "root_hash": DIGEST.hex(),
        "ots_receipts_json": json.dumps(receipts), "created_at": "2026-10-06 01:54:07+00",
    }])
    vault.update_anchor_receipts = AsyncMock()
    vault.update_anchor_status = AsyncMock()
    return vault


def _receipt():
    return {"calendar": "https://a.pool.opentimestamps.org",
            "receipt_b64": base64.b64encode(PENDING_RECEIPT).decode()}


@pytest.mark.asyncio
async def test_a_calendar_that_reached_bitcoin_confirms_the_anchor():
    vault = _vault_with_anchor([_receipt()])
    upgrade = AsyncMock(return_value=_upgrade_reaching_bitcoin(921_000))
    with patch("tollbooth.ots.OTSCalendarClient.upgrade_receipt", upgrade):
        r = await upgrade_pending_anchors(vault)
    assert r == {"checked": 1, "confirmed": 1}
    upgrade.assert_awaited_once_with(PENDING_COMMITMENT.hex(), PENDING_URI)
    stored = json.loads(vault.update_anchor_receipts.await_args.args[1])
    assert stored[0]["bitcoin_height"] == 921_000 and stored[0]["upgraded_b64"]
    assert vault.update_anchor_status.await_args.args[:2] == ("7", "confirmed")


@pytest.mark.asyncio
async def test_a_calendar_still_pending_leaves_the_anchor_submitted():
    vault = _vault_with_anchor([_receipt()])
    with patch("tollbooth.ots.OTSCalendarClient.upgrade_receipt", AsyncMock(return_value=None)):
        r = await upgrade_pending_anchors(vault)
    assert r == {"checked": 1, "confirmed": 0}
    vault.update_anchor_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_anchors_older_than_two_hours_are_asked_about():
    vault = MagicMock()
    vault.unconfirmed_anchors = AsyncMock(return_value=[])
    now = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
    await upgrade_pending_anchors(vault, now=now)
    cutoff = datetime.fromisoformat(vault.unconfirmed_anchors.await_args.kwargs["older_than"])
    assert cutoff == now - timedelta(hours=2)


# ── the background trigger ────────────────────────────────────────────

def _runtime(vault):
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    rt._ots_enabled = True
    rt.vault = AsyncMock(return_value=vault)
    return rt


def _notarizing_vault():
    vault = MagicMock()
    vault.unconfirmed_anchors = AsyncMock(return_value=[])
    vault.list_anchors = AsyncMock(return_value=[{"created_at": "2026-10-06 01:54:07.412774+00"}])
    vault.fetch_all_balances = AsyncMock(return_value=[("npub1a", "{}"), ("npub1b", "{}")])
    vault.store_anchor = AsyncMock(return_value="6")
    return vault


async def _settle(rt):
    await asyncio.wait_for(rt._ots_task, timeout=5)


@pytest.mark.asyncio
async def test_a_container_booted_minutes_ago_still_notarizes():
    vault = _notarizing_vault()
    rt = _runtime(vault)
    with patch("time.monotonic", return_value=120.0), \
         patch("tollbooth.ots.OTSCalendarClient.submit_digest", AsyncMock(return_value=[{"calendar": "c", "receipt_b64": ""}])):
        rt.fire_and_forget_notarize_if_stale()
        await _settle(rt)
    vault.store_anchor.assert_awaited_once()
    assert rt.notarization_status()["last_notarized_at"]
    assert "last_error" not in rt.notarization_status()


@pytest.mark.asyncio
async def test_a_second_call_within_the_hour_does_not_check_again():
    vault = _notarizing_vault()
    rt = _runtime(vault)
    with patch("time.monotonic", return_value=120.0), \
         patch("tollbooth.ots.OTSCalendarClient.submit_digest", AsyncMock(return_value=[])):
        rt.fire_and_forget_notarize_if_stale()
        await _settle(rt)
        first = rt._ots_task
        rt.fire_and_forget_notarize_if_stale()
    assert rt._ots_task is first


@pytest.mark.asyncio
async def test_a_failure_is_recorded_by_stage_and_type_only():
    vault = _notarizing_vault()
    vault.fetch_all_balances = AsyncMock(side_effect=RuntimeError("postgres://user:secret@host/db"))
    vault.list_anchors = AsyncMock(side_effect=RuntimeError("postgres://user:secret@host/db"))
    rt = _runtime(vault)
    rt.fire_and_forget_notarize_if_stale()
    await _settle(rt)
    status = rt.notarization_status()
    assert status["last_error"] == {"stage": "list_anchors", "error_type": "RuntimeError"}
    assert "secret" not in json.dumps(status)


# ── the tree is over plaintext ────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_tree_hashes_the_ledger_not_its_ciphertext():
    from tollbooth.vaults import NeonVault

    v = NeonVault(database_url="postgresql://u:p@ep-x.neon.tech/db", encryption_nsec_hex="a" * 64)
    ledger = '{"balance_api_sats": 5}'
    v._execute = AsyncMock(return_value={"rows": [{"npub": "npub1a", "ledger_json": v._encrypt(ledger)}]})
    assert await v.fetch_all_balances() == [("npub1a", ledger)]


@pytest.mark.asyncio
async def test_unconfirmed_anchors_are_read_oldest_first():
    from tollbooth.vaults import NeonVault

    v = NeonVault(database_url="postgresql://u:p@ep-x.neon.tech/db", encryption_nsec_hex="a" * 64)
    v._execute = AsyncMock(return_value={"rows": []})
    await v.unconfirmed_anchors(older_than="2026-11-03T10:00:00+00:00", limit=5)
    sql, params = v._execute.await_args.args
    assert "status = 'submitted'" in sql and "ORDER BY created_at ASC" in sql
    assert params == ["2026-11-03T10:00:00+00:00", 5]
