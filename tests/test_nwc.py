"""tollbooth.nwc — the NIP-47 client, against a wallet service played in-process.

``FakeRelay`` stands in for one relay socket with a wallet listening behind it.
The wallet holds a real key: it verifies the request's signature, opens it with
its own private key, and signs its answer, so both directions of the exchange
run through the SDK's real signing and encryption.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from typing import Any
from unittest.mock import patch

import pytest
from pynostr.bech32 import Encoding, bech32_encode
from pynostr.event import Event
from pynostr.key import PrivateKey

from tollbooth import nip04, nip44, nwc
from tollbooth.bolt11 import Bolt11Error, decode

WALLET = PrivateKey()
CLIENT = PrivateKey()
RELAY_A = "wss://relay-a.example.com"
RELAY_B = "wss://relay-b.example.com"
PREIMAGE = "11" * 32
PAYMENT_HASH = hashlib.sha256(bytes.fromhex(PREIMAGE)).hexdigest()


def _uri(*relays: str) -> str:
    query = "&".join(f"relay={r}" for r in relays or (RELAY_A,))
    return f"nostr+walletconnect://{WALLET.public_key.hex()}?{query}&secret={CLIENT.hex()}"


def _groups(value: int, count: int) -> list[int]:
    return [(value >> (5 * i)) & 31 for i in reversed(range(count))]


def make_invoice(
    amount_sats: int | None,
    *,
    payment_hash: str = PAYMENT_HASH,
    created_at: int | None = None,
    expiry: int = 3600,
) -> str:
    """A BOLT11 invoice with a blank signature — enough for a reader, not a node."""
    hrp = "lnbc" + (f"{amount_sats * 10}n" if amount_sats is not None else "")
    data = _groups(created_at if created_at is not None else int(time.time()), 7)
    data += [1, 1, 20] + _groups(int(payment_hash, 16) << 4, 52)
    data += [6, 0, 4] + _groups(expiry, 4)
    data += [0] * 104
    return str(bech32_encode(hrp, data, Encoding.BECH32))


class FakeRelay:
    """One relay socket, with a wallet service answering behind it."""

    def __init__(
        self,
        *,
        encryption: str | None = "nip44_v2 nip04",
        methods: str = "pay_invoice lookup_invoice get_info",
        answer: dict[str, Any] | None = None,
        publish_info: bool = True,
        refuse_subscription: bool = False,
        reject_event: bool = False,
        silent: bool = False,
        signer: PrivateKey = WALLET,
        decoy: str | None = None,
    ) -> None:
        self.decoy = decoy
        self.encryption = encryption
        self.methods = methods
        self.answer = answer
        self.publish_info = publish_info
        self.refuse_subscription = refuse_subscription
        self.reject_event = reject_event
        self.silent = silent
        self.signer = signer
        self.outbox: deque[str] = deque()
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    # -- websocket-client surface ------------------------------------------

    def settimeout(self, _seconds: float) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def recv(self) -> str:
        if not self.outbox:
            raise TimeoutError("timed out")
        return self.outbox.popleft()

    def send(self, raw: str) -> None:
        msg = json.loads(raw)
        if msg[0] == "REQ":
            self._on_req(msg[1], msg[2])
        elif msg[0] == "EVENT":
            self._on_event(msg[1])

    # -- relay + wallet behaviour ------------------------------------------

    def _push(self, *msg: Any) -> None:
        self.outbox.append(json.dumps(list(msg)))

    def _on_req(self, sub: str, flt: dict[str, Any]) -> None:
        if flt["kinds"] == [nwc.KIND_INFO]:
            if self.publish_info:
                tags = [["encryption", self.encryption]] if self.encryption else []
                info = Event(
                    pubkey=WALLET.public_key.hex(), kind=nwc.KIND_INFO,
                    content=self.methods, tags=tags,
                )
                info.sign(WALLET.hex())
                self._push("EVENT", sub, info.to_dict())
            self._push("EOSE", sub)
            return
        self.response_sub = sub
        self.response_filter = flt
        if self.refuse_subscription:
            self._push("CLOSED", sub, "auth-required: sign in first")
        else:
            self._push("EOSE", sub)

    def _on_event(self, event: dict[str, Any]) -> None:
        if self.reject_event:
            self._push("OK", event["id"], False, "blocked: not on the allow list")
            return
        assert Event.from_dict(event).verify(), "the request must be signed"
        assert event["kind"] == nwc.KIND_REQUEST
        assert event["pubkey"] == CLIENT.public_key.hex()
        scheme = next(t[1] for t in event["tags"] if t[0] == "encryption")
        opener = nip44.decrypt if scheme == "nip44_v2" else nip04.decrypt
        sealer = nip44.encrypt if scheme == "nip44_v2" else nip04.encrypt
        request = json.loads(opener(event["content"], WALLET.hex(), event["pubkey"]))
        self.requests.append({"scheme": scheme, "tags": event["tags"], **request})
        self._push("OK", event["id"], True, "")
        if self.silent:
            return
        answer = self.answer if self.answer is not None else {
            "result_type": request["method"], "error": None, "result": {},
        }
        if self.decoy:
            self._push("EVENT", self.response_sub, self._decoy(event))
        response = Event(
            pubkey=self.signer.public_key.hex(), kind=nwc.KIND_RESPONSE,
            content=sealer(json.dumps(answer), self.signer.hex(), event["pubkey"]),
            tags=[["p", event["pubkey"]], ["e", event["id"]]],
        )
        response.sign(self.signer.hex())
        self._push("EVENT", self.response_sub, response.to_dict())


    def _decoy(self, request: dict[str, Any]) -> dict[str, Any]:
        """An answer that arrives first and is not the wallet's answer to this request."""
        if self.decoy == "answer_to_another_request":
            # Genuinely the wallet's, about a request this client never made.
            stale = Event(
                pubkey=WALLET.public_key.hex(), kind=nwc.KIND_RESPONSE,
                content=nip44.encrypt(
                    json.dumps(_refusal("QUOTA_EXCEEDED", "an older request")),
                    WALLET.hex(), request["pubkey"],
                ),
                tags=[["p", request["pubkey"]], ["e", "ab" * 32]],
            )
            stale.sign(WALLET.hex())
            return dict(stale.to_dict())
        stranger = PrivateKey()
        forged = Event(
            pubkey=stranger.public_key.hex(), kind=nwc.KIND_RESPONSE,
            content=nip44.encrypt(json.dumps(_paid("22" * 32)), stranger.hex(), request["pubkey"]),
            tags=[["p", request["pubkey"]], ["e", request["id"]]],
        )
        forged.sign(stranger.hex())
        served = forged.to_dict()
        if self.decoy == "wallet_name_bad_signature":
            # Claims to be the wallet; the signature is still the stranger's.
            served["pubkey"] = WALLET.public_key.hex()
        return dict(served)


def _paid(preimage: str = PREIMAGE, **extra: Any) -> dict[str, Any]:
    return {"result_type": "pay_invoice", "error": None, "result": {"preimage": preimage, **extra}}


def _refusal(code: str, message: str, method: str = "pay_invoice") -> dict[str, Any]:
    return {"result_type": method, "error": {"code": code, "message": message}, "result": None}


def _dial(*relays: FakeRelay | Exception):
    """Patch the socket factory to hand out these relays in order."""
    return patch("websocket.create_connection", side_effect=list(relays))


# ---------------------------------------------------------------------------
# Connection string
# ---------------------------------------------------------------------------


class TestConnectionString:
    def test_the_spec_example_parses(self) -> None:
        conn = nwc.parse_connection_uri(
            "nostr+walletconnect://b889ff5b1513b641e2a139f661a661364979c5beee91842f8f0ef42ab558e9d4"
            "?relay=wss%3A%2F%2Frelay.damus.io"
            "&secret=71a8c14c1407c113601079c4302dab36460f0ccd0ad506f1f2dc73b5100e4f3c"
        )
        assert conn.wallet_pubkey.startswith("b889ff5b")
        assert conn.relays == ("wss://relay.damus.io",)
        assert conn.lud16 is None

    def test_several_relays_and_a_lightning_address(self) -> None:
        conn = nwc.parse_connection_uri(_uri(RELAY_A, RELAY_B, RELAY_A) + "&lud16=me%40example.com")
        assert conn.relays == (RELAY_A, RELAY_B)
        assert conn.lud16 == "me@example.com"
        assert conn.client_pubkey == CLIENT.public_key.hex()

    def test_the_secret_is_absent_from_repr(self) -> None:
        conn = nwc.parse_connection_uri(_uri())
        assert CLIENT.hex() not in repr(conn)
        assert CLIENT.hex() not in str(conn)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "https://example.com",
            f"nostr+walletconnect://nothex?relay={RELAY_A}&secret={CLIENT.hex()}",
            f"nostr+walletconnect://{WALLET.public_key.hex()}?relay={RELAY_A}",
            f"nostr+walletconnect://{WALLET.public_key.hex()}?relay={RELAY_A}&secret=short",
            f"nostr+walletconnect://{WALLET.public_key.hex()}?secret={CLIENT.hex()}",
            f"nostr+walletconnect://{WALLET.public_key.hex()}?relay=ws://relay.example.com&secret={CLIENT.hex()}",
            f"nostr+walletconnect://{WALLET.public_key.hex()}?relay=wss://localhost&secret={CLIENT.hex()}",
            f"nostr+walletconnect://{WALLET.public_key.hex()}?relay=wss://10.0.0.8&secret={CLIENT.hex()}",
            f"nostr+walletconnect://{WALLET.public_key.hex()}?relay=wss://169.254.169.254&secret={CLIENT.hex()}",
            f"nostr+walletconnect://{WALLET.public_key.hex()}?relay=wss://[::1]&secret={CLIENT.hex()}",
            _uri(*[f"wss://r{i}.example.com" for i in range(6)]),
        ],
    )
    def test_an_unusable_string_is_refused_without_quoting_it(self, bad: str) -> None:
        with pytest.raises(ValueError) as caught:
            nwc.parse_connection_uri(bad)
        assert CLIENT.hex() not in str(caught.value)


# ---------------------------------------------------------------------------
# Paying
# ---------------------------------------------------------------------------


@pytest.fixture
def conn() -> nwc.NwcConnection:
    return nwc.parse_connection_uri(_uri(RELAY_A, RELAY_B))


class TestPayInvoice:
    async def test_a_payment_returns_the_preimage_that_proves_it(self, conn) -> None:
        relay = FakeRelay(answer=_paid(fees_paid=1000))
        invoice = make_invoice(21)
        with _dial(relay):
            paid = await nwc.pay_invoice(conn, invoice, amount_msat=21_000)
        assert paid == nwc.Payment(PAYMENT_HASH, PREIMAGE, 1000)
        assert relay.requests == [{
            "scheme": "nip44_v2",
            "tags": relay.requests[0]["tags"],
            "method": "pay_invoice",
            "params": {"invoice": invoice},
        }]
        assert relay.closed

    async def test_the_request_names_the_wallet_and_expires(self, conn) -> None:
        relay = FakeRelay(answer=_paid())
        before = int(time.time())
        with _dial(relay):
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000, timeout=30)
        tags = {t[0]: t[1] for t in relay.requests[0]["tags"]}
        assert tags["p"] == WALLET.public_key.hex()
        assert before + 30 <= int(tags["expiration"]) <= int(time.time()) + 30
        assert relay.response_filter == {
            "kinds": [nwc.KIND_RESPONSE],
            "authors": [WALLET.public_key.hex()],
            "#e": relay.response_filter["#e"],
        }

    async def test_a_wallet_with_no_encryption_tag_is_spoken_to_in_nip04(self, conn) -> None:
        relay = FakeRelay(encryption=None, answer=_paid())
        with _dial(relay):
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert relay.requests[0]["scheme"] == "nip04"

    async def test_an_invoice_for_more_than_agreed_is_never_sent(self, conn) -> None:
        with _dial() as dial, pytest.raises(nwc.NwcAmountMismatch, match="22000 msat"):
            await nwc.pay_invoice(conn, make_invoice(22), amount_msat=21_000)
        dial.assert_not_called()

    async def test_an_invoice_for_less_than_agreed_is_never_sent(self, conn) -> None:
        with _dial() as dial, pytest.raises(nwc.NwcAmountMismatch):
            await nwc.pay_invoice(conn, make_invoice(20), amount_msat=21_000)
        dial.assert_not_called()

    async def test_an_invoice_naming_no_amount_is_never_sent(self, conn) -> None:
        with _dial() as dial, pytest.raises(nwc.NwcAmountMismatch):
            await nwc.pay_invoice(conn, make_invoice(None), amount_msat=21_000)
        dial.assert_not_called()

    async def test_an_expired_invoice_is_never_sent(self, conn) -> None:
        stale = make_invoice(21, created_at=int(time.time()) - 7200, expiry=3600)
        with _dial() as dial, pytest.raises(nwc.NwcError, match="expired"):
            await nwc.pay_invoice(conn, stale, amount_msat=21_000)
        dial.assert_not_called()

    async def test_text_that_is_not_an_invoice_is_never_sent(self, conn) -> None:
        with _dial() as dial, pytest.raises(Bolt11Error):
            await nwc.pay_invoice(conn, "lnbc-not-really", amount_msat=21_000)
        dial.assert_not_called()

    async def test_the_wallets_budget_refusal_carries_its_code(self, conn) -> None:
        relay = FakeRelay(answer=_refusal("QUOTA_EXCEEDED", "budget of 1000 sats is spent"))
        with _dial(relay), pytest.raises(nwc.NwcError) as caught:
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert caught.value.code == "QUOTA_EXCEEDED"
        assert "budget of 1000 sats" in caught.value.message

    async def test_a_long_wallet_message_is_cut_short(self, conn) -> None:
        relay = FakeRelay(answer=_refusal("OTHER", "x" * 5000))
        with _dial(relay), pytest.raises(nwc.NwcError) as caught:
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert len(caught.value.message) == 300

    async def test_a_preimage_that_does_not_match_the_invoice_is_not_a_payment(self, conn) -> None:
        relay = FakeRelay(answer=_paid(preimage="22" * 32))
        with _dial(relay), pytest.raises(nwc.NwcOutcomeUnknown, match="preimage"):
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)

    async def test_a_connection_that_may_not_pay_is_told_so_before_asking(self, conn) -> None:
        relay = FakeRelay(methods="get_info lookup_invoice")
        with _dial(relay), pytest.raises(nwc.NwcError) as caught:
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert caught.value.code == "NOT_IMPLEMENTED"
        assert relay.requests == []


class TestWhichRelayCarriesIt:
    async def test_a_relay_that_cannot_be_reached_is_skipped(self, conn) -> None:
        second = FakeRelay(answer=_paid())
        with _dial(OSError("refused"), second):
            paid = await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert paid.preimage == PREIMAGE

    async def test_a_relay_that_refuses_the_subscription_is_skipped(self, conn) -> None:
        first, second = FakeRelay(refuse_subscription=True), FakeRelay(answer=_paid())
        with _dial(first, second):
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert first.requests == [] and len(second.requests) == 1
        assert first.closed

    async def test_a_relay_that_rejects_the_request_is_skipped(self, conn) -> None:
        first, second = FakeRelay(reject_event=True), FakeRelay(answer=_paid())
        with _dial(first, second):
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert len(second.requests) == 1

    async def test_a_relay_holding_no_info_event_is_skipped(self, conn) -> None:
        first, second = FakeRelay(publish_info=False), FakeRelay(answer=_paid())
        with _dial(first, second):
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert first.requests == []

    async def test_when_no_relay_carries_it_nothing_was_sent(self, conn) -> None:
        with _dial(OSError("refused"), FakeRelay(reject_event=True)), pytest.raises(
            nwc.NwcNotSent
        ) as caught:
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert "relay-a.example.com" in str(caught.value)
        assert "relay-b.example.com" in str(caught.value)

    async def test_silence_after_sending_is_unknown_and_is_not_sent_again(self, conn) -> None:
        first, second = FakeRelay(silent=True), FakeRelay(answer=_paid())
        with _dial(first, second), pytest.raises(nwc.NwcOutcomeUnknown):
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert len(first.requests) == 1
        assert second.requests == [], "a second request could pay twice"

    async def test_an_answer_signed_by_someone_else_is_ignored(self, conn) -> None:
        impostor = FakeRelay(answer=_paid(), signer=PrivateKey())
        with _dial(impostor), pytest.raises(nwc.NwcOutcomeUnknown):
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)

    @pytest.mark.parametrize(
        "decoy", ["stranger", "wallet_name_bad_signature", "answer_to_another_request"]
    )
    async def test_a_forged_answer_is_passed_over_for_the_wallets_own(self, conn, decoy) -> None:
        relay = FakeRelay(answer=_paid(), decoy=decoy)
        with _dial(relay):
            paid = await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
        assert paid.preimage == PREIMAGE

    async def test_an_answer_to_a_different_method_is_unknown(self, conn) -> None:
        relay = FakeRelay(answer={"result_type": "get_info", "error": None, "result": {}})
        with _dial(relay), pytest.raises(nwc.NwcOutcomeUnknown, match="different question"):
            await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)


# ---------------------------------------------------------------------------
# Looking up, and info
# ---------------------------------------------------------------------------


class TestLookupInvoice:
    async def test_a_settled_payment_is_found_by_its_hash(self, conn) -> None:
        relay = FakeRelay(answer={
            "result_type": "lookup_invoice", "error": None,
            "result": {"state": "settled", "preimage": PREIMAGE, "fees_paid": 2000,
                       "settled_at": 1_800_000_000},
        })
        with _dial(relay):
            seen = await nwc.lookup_invoice(conn, payment_hash=PAYMENT_HASH.upper())
        assert seen == nwc.InvoiceStatus(
            PAYMENT_HASH, "settled", True, PREIMAGE, 2000, 1_800_000_000
        )
        assert relay.requests[0]["params"] == {"payment_hash": PAYMENT_HASH}

    async def test_a_pending_payment_is_not_settled(self, conn) -> None:
        relay = FakeRelay(answer={
            "result_type": "lookup_invoice", "error": None, "result": {"state": "pending"},
        })
        with _dial(relay):
            seen = await nwc.lookup_invoice(conn, payment_hash=PAYMENT_HASH)
        assert seen.settled is False and seen.preimage is None

    async def test_a_preimage_for_another_payment_does_not_settle_this_one(self, conn) -> None:
        relay = FakeRelay(answer={
            "result_type": "lookup_invoice", "error": None,
            "result": {"state": "pending", "preimage": "22" * 32},
        })
        with _dial(relay):
            seen = await nwc.lookup_invoice(conn, payment_hash=PAYMENT_HASH)
        assert seen.settled is False and seen.preimage is None

    async def test_a_payment_the_wallet_never_saw_is_not_found(self, conn) -> None:
        relay = FakeRelay(answer=_refusal("NOT_FOUND", "no such invoice", "lookup_invoice"))
        with _dial(relay), pytest.raises(nwc.NwcError) as caught:
            await nwc.lookup_invoice(conn, payment_hash=PAYMENT_HASH)
        assert caught.value.code == "NOT_FOUND"

    async def test_a_malformed_hash_is_refused_before_asking(self, conn) -> None:
        with _dial() as dial, pytest.raises(ValueError, match="64 hex"):
            await nwc.lookup_invoice(conn, payment_hash="abc")
        dial.assert_not_called()


class TestGetInfo:
    async def test_it_reports_the_wallet_and_what_the_connection_may_do(self, conn) -> None:
        relay = FakeRelay(answer={
            "result_type": "get_info", "error": None,
            "result": {"alias": "Hub", "network": "mainnet",
                       "methods": ["pay_invoice", "get_info", 7]},
        })
        with _dial(relay):
            info = await nwc.get_info(conn)
        assert info == nwc.WalletInfo("Hub", "mainnet", ("pay_invoice", "get_info"))


# ---------------------------------------------------------------------------
# The secret stays put
# ---------------------------------------------------------------------------


class TestTheSecretStaysPut:
    async def test_no_log_line_or_error_carries_the_secret(self, conn, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        outcomes: list[BaseException] = []
        scripts = [
            (FakeRelay(answer=_paid()),),
            (FakeRelay(answer=_refusal("QUOTA_EXCEEDED", "over budget")),),
            (FakeRelay(silent=True),),
            (OSError("refused"), FakeRelay(reject_event=True)),
        ]
        for script in scripts:
            with _dial(*script):
                try:
                    await nwc.pay_invoice(conn, make_invoice(21), amount_msat=21_000)
                except Exception as exc:  # noqa: BLE001
                    outcomes.append(exc)
        assert len(outcomes) == 3
        spoken = caplog.text + "".join(f"{e!r}{e}{e.__cause__!r}" for e in outcomes)
        assert CLIENT.hex() not in spoken
        assert "secret=" not in spoken


def test_the_fixture_invoice_is_what_it_claims() -> None:
    """Guard the helper: the spec-checked reader agrees with what it wrote."""
    asked = decode(make_invoice(21, expiry=600))
    assert (asked.amount_msat, asked.payment_hash, asked.expiry_seconds) == (
        21_000, PAYMENT_HASH, 600,
    )
