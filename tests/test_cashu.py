"""tollbooth.cashu against the Cashu specification's published test vectors.

Vectors: https://github.com/cashubtc/nuts/tree/main/tests (00, 11, 12, 20).
"""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from tollbooth import cashu

# --- NUT-00 ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "point"),
    [
        ("00" * 32, "024cce997d3b518f739663b757deaec95bcd9473c30a14ac2fd04023a739d1a725"),
        ("00" * 31 + "01", "022e7158e11c9506f1aa4248bf531298daa7febd6194f003edcd9b93ade6253acf"),
        ("00" * 31 + "02", "026cdbe15362df59cd1dd3c9c11de8aedac2106eca69236ecd9fbe117af897be4f"),
    ],
)
def test_hash_to_curve_vectors(message, point):
    assert cashu.hash_to_curve(bytes.fromhex(message)).format().hex() == point


def _blind_bytes(x_hex: str, r_hex: str) -> str:
    y = cashu.hash_to_curve(bytes.fromhex(x_hex))
    from coincurve import PrivateKey, PublicKey

    return PublicKey.combine_keys([y, PrivateKey(bytes.fromhex(r_hex)).public_key]).format().hex()


@pytest.mark.parametrize(
    ("x", "r", "b_"),
    [
        ("d341ee4871f1f889041e63cf0d3823c713eea6aff01e80f1719f08f9e5be98f6",
         "99fce58439fc37412ab3468b73db0569322588f62fb3a49182d67e23d877824a",
         "033b1a9737a40cc3fd9b6af4b723632b76a67a36782596304612a6c2bfb5197e6d"),
        ("f1aaf16c2239746f369572c0784d9dd3d032d952c2d992175873fb58fae31a60",
         "f78476ea7cc9ade20f9e05e58a804cf19533f03ea805ece5fee88c8e2874ba50",
         "029bdf2d716ee366eddf599ba252786c1033f47e230248a4612a5670ab931f1763"),
    ],
)
def test_blinded_message_vectors(x, r, b_):
    assert _blind_bytes(x, r) == b_


@pytest.mark.parametrize(
    ("k", "c_"),
    [
        ("00" * 31 + "01", "02a9acc1e48c25eeeb9289b5031cc57da9fe72f3fe2861d264bdc074209b107ba2"),
        ("7f" * 32, "0398bc70ce8184d27ba89834d19f5199c84443c31131e48d3c1214db24247d005d"),
    ],
)
def test_blinded_signature_vectors(k, c_):
    b_ = "02a9acc1e48c25eeeb9289b5031cc57da9fe72f3fe2861d264bdc074209b107ba2"
    assert cashu.mint_sign(bytes.fromhex(k), b_) == c_


def test_blind_sign_unblind_round_trip():
    k = bytes.fromhex("7f" * 32)
    from coincurve import PrivateKey

    mint_key = PrivateKey(k).public_key.format().hex()
    secret = cashu.p2pk_secret("02" + "11" * 32)
    b_, r = cashu.blind(secret)
    c = cashu.unblind(cashu.mint_sign(k, b_), r, mint_key)
    assert cashu.mint_verify(k, secret, c)
    assert not cashu.mint_verify(bytes.fromhex("01" * 32), secret, c)


# --- NUT-12 ------------------------------------------------------------------


def test_hash_e_vector():
    one = cashu._point("02" + "00" * 31 + "01")
    c_ = cashu._point("02a9acc1e48c25eeeb9289b5031cc57da9fe72f3fe2861d264bdc074209b107ba2")
    assert cashu.hash_e(one, one, one, c_).hex() == "a4dc034b74338c28c6bc3ea49731f2a24440fc7c4affc08b31a93fc9fbe6401e"


def test_dleq_on_blind_signature_vector():
    a = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    b_ = "02a9acc1e48c25eeeb9289b5031cc57da9fe72f3fe2861d264bdc074209b107ba2"
    e = "9818e061ee51d5c8edc3342369a554998ff7b4381c8652d724cdf46429be73d9"
    s = "9818e061ee51d5c8edc3342369a554998ff7b4381c8652d724cdf46429be73da"
    assert cashu.verify_dleq(b_, b_, e, s, a)
    assert not cashu.verify_dleq(b_, b_, e, s[:-1] + "b", a)


def test_dleq_on_deterministic_vector():
    a = "02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"
    b_ = "02a9acc1e48c25eeeb9289b5031cc57da9fe72f3fe2861d264bdc074209b107ba2"
    c_ = "0244eccfc7a348274458bb38044c7f3c389b3c2086c7ec18b5812d2877ab937787"
    e = "2a16ffee280aff3c429045607f9b8e0bf8b35910c44c1b20b9dfaf01b263d7b3"
    s = "9df27731238334718d120d4f74611a7c668233f988e687ac3fb188f0a34a2dab"
    assert cashu.verify_dleq(b_, c_, e, s, a)


# --- NUT-11 ------------------------------------------------------------------

SWAP_SECRET = (
    '["P2PK",{"nonce":"c7f280eb55c1e8564e03db06973e94bc9b666d9e1ca42ad278408fe625950303",'
    '"data":"030d8acedfe072c9fa449a1efe0817157403fbec460d8e79f957966056e5dd76c1","tags":[["sigflag","SIG_ALL"]]}]'
)


def test_sig_all_swap_message_vector():
    inputs = [{"amount": 2, "id": "00bfa73302d12ffd", "secret": SWAP_SECRET,
               "C": "02c97ee3d1db41cf0a3ddb601724be8711a032950811bf326f8219c50c4808d3cd"}]
    outputs = [{"amount": 2, "id": "00bfa73302d12ffd",
                "B_": "038ec853d65ae1b79b5cdbc2774150b2cb288d6d26e12958a16fb33c32d9a86c39"}]
    msg = cashu.swap_message(inputs, outputs)
    assert hashlib.sha256(msg.encode()).hexdigest() == "de7f9e3ca0fcc5ed3258fcf83dbf1be7fa78a5ed6da7bf2aa60d61e9dc6eb09a"
    sig = ("ce017ca25b1b97df2f72e4b49f69ac26a240ce14b3690a8fe619d41ccc42d3c1"
           "282e073f85acd36dc50011638906f35b56615f24e4d03e8effe8257f6a808538")
    assert cashu.schnorr_verify("030d8acedfe072c9fa449a1efe0817157403fbec460d8e79f957966056e5dd76c1", msg, sig)


def test_sig_all_multisig_melt_message_vector():
    secret = (
        '["P2PK",{"nonce":"68d7822538740e4f9c9ebf5183ef6c4501c7a9bca4e509ce2e41e1d62e7b8a99",'
        '"data":"0394e841bd59aeadce16380df6174cb29c9fea83b0b65b226575e6d73cc5a1bd59",'
        '"tags":[["pubkeys","033d892d7ad2a7d53708b7a5a2af101cbcef69522bd368eacf55fcb4f1b0494058"],'
        '["n_sigs","2"],["sigflag","SIG_ALL"]]}]'
    )
    inputs = [{"amount": 2, "id": "00bfa73302d12ffd", "secret": secret,
               "C": "03a70c42ec9d7192422c7f7a3ad017deda309fb4a2453fcf9357795ea706cc87a9"}]
    outputs = [{"amount": 0, "id": "00bfa73302d12ffd",
                "B_": "038ec853d65ae1b79b5cdbc2774150b2cb288d6d26e12958a16fb33c32d9a86c39"}]
    msg = cashu.melt_message(inputs, outputs, "Db3qEMVwFN2tf_1JxbZp29aL5cVXpSMIwpYfyOVF")
    sigs = [
        "ed739970d003f703da2f101a51767b63858f4894468cc334be04aa3befab1617a81e3eef093441afb499974152d279e59d9582a31dc68adbc17ffc22a2516086",
        "f9efe1c70eb61e7ad8bd615c50ff850410a4135ea73ba5fd8e12a734743ad045e575e9e76ea5c52c8e7908d3ad5c0eaae93337e5c11109e52848dc328d6757a2",
    ]
    keys = ["0394e841bd59aeadce16380df6174cb29c9fea83b0b65b226575e6d73cc5a1bd59",
            "033d892d7ad2a7d53708b7a5a2af101cbcef69522bd368eacf55fcb4f1b0494058"]
    assert sorted(any(cashu.schnorr_verify(k, msg, s) for k in keys) for s in sigs) == [True, True]


def test_single_sig_melt_message_vector():
    secret = (
        '["P2PK",{"nonce":"bbf9edf441d17097e39f5095a3313ba24d3055ab8a32f758ff41c10d45c4f3de",'
        '"data":"029116d32e7da635c8feeb9f1f4559eb3d9b42d400f9d22a64834d89cde0eb6835","tags":[["sigflag","SIG_ALL"]]}]'
    )
    inputs = [{"secret": secret, "C": "02a9d461ff36448469dccf828fa143833ae71c689886ac51b62c8d61ddaa10028b"}]
    outputs = [{"amount": 0, "B_": "038ec853d65ae1b79b5cdbc2774150b2cb288d6d26e12958a16fb33c32d9a86c39"}]
    msg = cashu.melt_message(inputs, outputs, "cF8911fzT88aEi1d-6boZZkq5lYxbUSVs-HbJxK0")
    assert hashlib.sha256(msg.encode()).hexdigest() == "9efa1067cc7dc870f4074f695115829c3cd817a6866c3b84e9814adf3c3cf262"


def test_p2pk_secret_shape_and_guards():
    key = "02" + "ab" * 32
    s = cashu.p2pk_secret(key, pubkeys=["02" + "cd" * 32, "02" + "ef" * 32], n_sigs=2,
                          locktime=1700000000, refund=[key])
    kind, body = json.loads(s)
    assert kind == "P2PK" and body["data"] == key and len(body["nonce"]) == 64
    assert ["n_sigs", "2"] in body["tags"] and ["sigflag", "SIG_ALL"] in body["tags"]
    assert cashu.secret_lock(s) == cashu.secret_lock(
        cashu.p2pk_secret(key, pubkeys=["02" + "cd" * 32, "02" + "ef" * 32], n_sigs=2, locktime=1700000000, refund=[key]))
    with pytest.raises(ValueError, match="refund"):
        cashu.p2pk_secret(key, locktime=1)
    with pytest.raises(ValueError, match="n_sigs"):
        cashu.p2pk_secret(key, n_sigs=2)
    with pytest.raises(ValueError, match="sigflag"):
        cashu.p2pk_secret(key, sigflag="ANY")


def test_schnorr_sign_verify_and_lock_key():
    sk = bytes.fromhex("01" * 32)
    from coincurve import PrivateKey

    xonly = PrivateKey(sk).public_key.format()[1:].hex()
    sig = cashu.schnorr_sign(sk, "hello")
    assert cashu.schnorr_verify(xonly, "hello", sig)
    assert cashu.schnorr_verify(cashu.lock_key(xonly), "hello", sig)
    assert not cashu.schnorr_verify(xonly, "hellO", sig)
    assert not cashu.schnorr_verify(xonly, "hello", "00" * 64)
    with pytest.raises(ValueError):
        cashu.lock_key("ab" * 33)


# --- NUT-20 ------------------------------------------------------------------


def test_mint_quote_message_vector():
    outputs = [
        {"amount": 1, "id": "009a1f293253e41e", "B_": "036d6caac248af96f6afa7f904f550253a0f3ef3f5aa2fe6838a95b216691468e2"},
        {"amount": 1, "id": "009a1f293253e41e", "B_": "021f8a566c205633d029094747d2e18f44e05993dda7a5f88f496078205f656e59"},
    ]
    msg = cashu.mint_quote_message("0192d3c0-7e8a-7c3d-8e9f-1a2b3c4d5e6f", outputs)
    assert hashlib.sha256(msg).hexdigest() == "c164fd384879f74ab6ea2e7cf13d90ed42e6df9d5de607eeb5c9cc7d36fb1c21"
    sig = ("4881093a332ff7c79f3e598ce5b249d64978b47165a0b19c18adf0ced0246228"
           "e61e702f0abaf1bf27b92be4336bdbabacfbe4c914076386b3c66fdcd0b3480e")
    assert cashu.schnorr_verify("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", msg, sig)


# --- Amounts, outputs, proofs -------------------------------------------------


def test_amounts_and_fees():
    assert cashu.amount_split(999) == [1, 2, 4, 32, 64, 128, 256, 512]
    assert sum(cashu.amount_split(12345)) == 12345
    assert cashu.amount_split(0) == []
    assert cashu.input_fee([{}] * 6, 100) == 1
    assert cashu.input_fee([{}] * 11, 100) == 2
    assert cashu.blanks_needed(0) == 0 and cashu.blanks_needed(19) == 5


def test_proofs_from_requires_dleq():
    from coincurve import PrivateKey

    k = PrivateKey()
    keys = {8: k.public_key.format().hex()}
    pending = cashu.outputs_for([8], ["aa" * 32], "00ff")
    promise = {"amount": 8, "id": "00ff", "C_": cashu.mint_sign(k.secret, pending.outputs[0]["B_"])}
    with pytest.raises(cashu.CashuError, match="DLEQ"):
        cashu.proofs_from([promise], pending, keys)


def test_with_witness_puts_every_signature_on_the_first_input():
    proofs = [{"secret": "a", "C": "b", "witness": "x"}, {"secret": "c", "C": "d", "witness": "y"}]
    out = cashu.with_witness(proofs, ["s1", "s2"])
    assert json.loads(out[0]["witness"]) == {"signatures": ["s1", "s2"]}
    assert "witness" not in out[1]


# --- The mint client ------------------------------------------------------------


@pytest.mark.parametrize("url", ["http://mint.example", "https://localhost", "https://10.0.0.1", "https://mint.local", "ftp://x"])
def test_mint_url_must_be_public_https(url):
    with pytest.raises(ValueError):
        cashu.checked_mint_url(url)


def test_mint_url_normalized():
    assert cashu.checked_mint_url(" https://testnut.cashu.space/ ") == "https://testnut.cashu.space"


def test_missing_escrow_nuts():
    info = {"nuts": {"7": {"supported": True}, "8": {"supported": True}, "11": {"supported": True},
                     "12": {"supported": True}, "20": {"supported": False}}}
    assert cashu.missing_escrow_nuts(info) == ["20"]
    assert cashu.missing_escrow_nuts({}) == list(cashu.ESCROW_NUTS)


async def test_mint_client_reports_mint_refusal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "Token already spent", "code": 11001})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        mint = cashu.MintClient("https://mint.example", client=client)
        with pytest.raises(cashu.CashuError) as err:
            await mint.swap([], [])
    assert err.value.code == 11001 and "already spent" in str(err.value)


async def test_active_keyset_reads_fee_and_keys():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/keysets":
            return httpx.Response(200, json={"keysets": [
                {"id": "00usd", "unit": "usd", "active": True, "input_fee_ppk": 0},
                {"id": "00sat", "unit": "sat", "active": True, "input_fee_ppk": 100}]})
        return httpx.Response(200, json={"keysets": [{"id": "00sat", "keys": {"1": "02" + "aa" * 32}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ks = await cashu.MintClient("https://mint.example", client=client).active_keyset()
    assert ks == {"id": "00sat", "input_fee_ppk": 100, "keys": {1: "02" + "aa" * 32}}
