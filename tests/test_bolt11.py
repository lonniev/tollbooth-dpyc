"""bolt11.decode against the BOLT11 specification's own examples.

Vectors are copied from lightning/bolts 11-payment-encoding.md ("Examples" and
"Examples of Invalid Invoices"), so the reader is checked against invoices it
did not write.
"""

from __future__ import annotations

import pytest

from tollbooth.bolt11 import Bolt11Error, decode

SPEC_HASH = "0001020304050607080900010203040506070809000102030405060708090102"

DONATION = (
    "lnbc1pvjluezsp5zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zygspp5qqqsyqcyq5rqwzqf"
    "qqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqypqdpl2pkx2ctnv5sxxmmwwd5kgetjypeh2ursdae8g6twvus8g6r"
    "fwvs8qun0dfjkxaq9qrsgq357wnc5r2ueh7ck6q93dj32dlqnls087fxdwk8qakdyafkq3yap9us6v52vjjsrv"
    "ywa6rt52cm9r9zqt8r2t7mlcwspyetp5h2tztugp9lfyql"
)
COFFEE = (
    "lnbc2500u1pvjluezsp5zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zygspp5qqqsyqcyq5r"
    "qwzqfqqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqypqdq5xysxxatsyp3k7enxv4jsxqzpu9qrsgquk0rl77nj30"
    "yxdy8j9vdx85fkpmdla2087ne0xh8nhedh8w27kyke0lp53ut353s06fv3qfegext0eh0ymjpf39tuven09sam"
    "30g4vgpfna3rh"
)
TESTNET = (
    "lntb20m1pvjluezsp5zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zygshp58yjmdan79s6qq"
    "dhdzgynm4zwqd5d7xmw5fk98klysy043l2ahrqspp5qqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqqqsyqcyq5rq"
    "wzqfqypqfpp3x9et2e20v6pu37c5d9vax37wxq72un989qrsgqdj545axuxtnfemtpwkc45hx9d2ft7x04mt8q"
    "7y6t0k2dge9e7h8kpy9p34ytyslj3yu569aalz2xdk8xkd7ltxqld94u8h2esmsmacgpghe9k8"
)
PICO = (
    "lnbc9678785340p1pwmna7lpp5gc3xfm08u9qy06djf8dfflhugl6p7lgza6dsjxq454gxhj9t7a0sd8dgfkx7"
    "cmtwd68yetpd5s9xar0wfjn5gpc8qhrsdfq24f5ggrxdaezqsnvda3kkum5wfjkzmfqf3jkgem9wgsyuctwdus"
    "9xgrcyqcjcgpzgfskx6eqf9hzqnteypzxz7fzypfhg6trddjhygrcyqezcgpzfysywmm5ypxxjemgw3hxjmn8y"
    "ptk7untd9hxwg3q2d6xjcmtv4ezq7pqxgsxzmnyyqcjqmt0wfjjq6t5v4khxsp5zyg3zyg3zyg3zyg3zyg3zyg"
    "3zyg3zyg3zyg3zyg3zyg3zyg3zygsxqyjw5qcqp2rzjq0gxwkzc8w6323m55m4jyxcjwmy7stt9hwkwe2qxmy8"
    "zpsgg7jcuwz87fcqqeuqqqyqqqqlgqqqqn3qq9q9qrsgqrvgkpnmps664wgkp43l22qsgdw4ve24aca4nymnxd"
    "dlnp8vh9v2sdxlu5ywdxefsfvm0fq3sesf08uf6q9a2ke0hc9j6z6wlxg5z5kqpu2v9wz"
)
COFFEE_BEANS = (
    "lnbc25m1pvjluezpp5qqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqypqdq5vdhkven9v5sxy"
    "etpdeessp5zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zygs9q5sqqqqqqqqqqqqqqqqsgq2"
    "a25dxl5hrntdtn6zvydt7d66hyzsyhqs4wdynavys42xgl6sgx9c4g7me86a27t07mdtfry458rtjr0v92cnms"
    "wpsjscgt2vcse3sgpz3uapa"
)


class TestSpecExamples:
    def test_an_invoice_for_any_amount_has_no_amount(self) -> None:
        asked = decode(DONATION)
        assert asked.amount_msat is None
        assert asked.payment_hash == SPEC_HASH
        assert asked.network == "bc"
        assert asked.created_at == 1496314658
        assert asked.expiry_seconds == 3600

    def test_micro_bitcoin_amount_and_a_one_minute_expiry(self) -> None:
        asked = decode(COFFEE)
        assert asked.amount_msat == 250_000_000
        assert asked.payment_hash == SPEC_HASH
        assert asked.expiry_seconds == 60
        assert asked.expires_at == 1496314658 + 60

    def test_testnet_invoice_with_the_hash_after_other_fields(self) -> None:
        asked = decode(TESTNET)
        assert asked.network == "tb"
        assert asked.amount_msat == 2_000_000_000
        assert asked.payment_hash == SPEC_HASH

    def test_pico_bitcoin_amount_and_a_one_week_expiry(self) -> None:
        asked = decode(PICO)
        assert asked.amount_msat == 967_878_534
        assert asked.payment_hash == (
            "462264ede7e14047e9b249da94fefc47f41f7d02ee9b091815a5506bc8abf75f"
        )
        assert asked.expiry_seconds == 604_800

    def test_all_upper_case_reads_the_same(self) -> None:
        assert decode(COFFEE_BEANS.upper()) == decode(COFFEE_BEANS)

    def test_a_lightning_uri_prefix_is_accepted(self) -> None:
        assert decode("lightning:" + COFFEE) == decode(COFFEE)


class TestRefusals:
    def test_a_changed_character_fails_the_checksum(self) -> None:
        with pytest.raises(Bolt11Error, match="checksum"):
            decode(COFFEE[:-1] + ("q" if COFFEE[-1] != "q" else "p"))

    def test_a_changed_amount_fails_the_checksum(self) -> None:
        with pytest.raises(Bolt11Error, match="checksum"):
            decode(COFFEE.replace("lnbc2500u", "lnbc2600u", 1))

    def test_mixed_case_is_refused(self) -> None:
        with pytest.raises(Bolt11Error):
            decode("LNBC2500u" + COFFEE[9:])

    def test_sub_millisatoshi_precision_is_refused(self) -> None:
        # Spec: "Invalid sub-millisatoshi precision."
        with pytest.raises(Bolt11Error, match="finer than a millisatoshi"):
            decode(
                "lnbc2500000001p1pvjluezpp5qqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqy"
                "pqdq5xysxxatsyp3k7enxv4jsxqzpusp5zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg"
                "3zyg3zygs9qrsgq0lzc236j96a95uv0m3umg28gclm5lqxtqqwk32uuk4k6673k6n5kfvx3d2h8s"
                "295fad45fdhmusm8sjudfhlf6dcsxmfvkeywmjdkxcp99202x"
            )

    def test_an_invalid_multiplier_is_refused(self) -> None:
        # Spec: "Invalid multiplier"
        with pytest.raises(Bolt11Error, match="prefix"):
            decode(
                "lnbc2500x1pvjluezpp5qqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqypqdq5x"
                "ysxxatsyp3k7enxv4jsxqzpusp5zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3z"
                "ygs9qrsgqrrzc4cvfue4zp3hggxp47ag7xnrlr8vgcmkjxk3j5jqethnumgkpqp23z9jclu3v0a7"
                "e0aruz366e9wqdykw6dxhdzcjjhldxq0w6wgqcnu43j"
            )

    @pytest.mark.parametrize("junk", ["", "   ", "lnbc", "not an invoice", "npub1" + "q" * 58])
    def test_text_that_is_not_an_invoice_is_refused(self, junk: str) -> None:
        with pytest.raises(Bolt11Error):
            decode(junk)

    def test_an_enormous_string_is_refused_without_parsing(self) -> None:
        with pytest.raises(Bolt11Error, match="too long"):
            decode("lnbc1" + "q" * 10_000)



# --- the paid node ------------------------------------------------------------

#: The spec's examples are signed by this node ("…to me @03e7156ae33b…").
SPEC_PAYEE = "03e7156ae33b0a208d0744199163177e909e80176e55d97a2f221ede0f934dd9ad"


@pytest.mark.parametrize("invoice", [DONATION, COFFEE, TESTNET])
def test_the_payee_is_recovered_from_the_signature(invoice):
    assert decode(invoice).payee == SPEC_PAYEE


def test_a_signature_that_recovers_nothing_names_no_payee():
    from tollbooth.bolt11 import _signer

    assert _signer("lnbc", [0] * 200) == ""
