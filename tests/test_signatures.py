"""Signatures over a statement, by signed request or by signed reply, checkable from the record alone.

Pins:
  - the statement is normalized before hashing (trimmed, Unix line endings);
  - over HTTP, a tool keeps the request an agent signed as a signature record,
    and the record verifies only for that signer, that body and that hash;
  - a DM reply plus the Operator's witness verifies only for that signer,
    that reply, that statement and that Operator;
  - a collected approval reply carries the whole signed layer.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastmcp import FastMCP
from fastmcp.utilities.tests import run_server_async
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tests.test_holder_auth import _free_port
from tollbooth.holder_auth import SignedRequestProvider
from tollbooth.identity_proof import sign_signature_witness
from tollbooth.mcp_client import service_client
from tollbooth.tools.proof import ApprovalReply
from tollbooth.tools.signature import (
    normalized_statement,
    signed_request_record,
    statement_hash,
    verify_signature_record,
)

STATEMENT = "I agree to the cover page of job 7.\r\nTerms v0.\n"
DIGEST = statement_hash(STATEMENT)


def test_the_statement_is_normalized_before_it_is_hashed():
    assert normalized_statement("  a\r\nb \n") == "a\nb"
    assert statement_hash(" I agree.\r\n") == statement_hash("I agree.")
    assert statement_hash("I agree.") != statement_hash("I agree!")


def test_a_reply_record_always_has_its_whole_signed_layer_field():
    assert ApprovalReply(payload={}, patron_sig="", patron_event_id="", popped=0).patron_event == {}


# ---------------------------------------------------------------------------
# A signed request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_agent_signs_by_calling_a_tool_with_its_own_key():
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    mcp = FastMCP("signing", auth=SignedRequestProvider())

    @mcp.tool
    def sign(digest: str) -> dict[str, Any]:
        return {"record": signed_request_record(digest)}

    alice, bob = PrivateKey(), PrivateKey()
    async with run_server_async(mcp, port=port):
        async with service_client(url, alice.bech32()) as client:
            record = (await client.call_tool("sign", {"digest": DIGEST})).structured_content["record"]
            elsewhere = (await client.call_tool("sign", {"digest": "0" * 64})).structured_content["record"]
        async with service_client(url, None) as client:
            unsigned = (await client.call_tool("sign", {"digest": DIGEST})).structured_content["record"]

    alice_npub, bob_npub = alice.public_key.bech32(), bob.public_key.bech32()
    assert record["method"] == "signed_request" and record["signer"] == alice_npub
    assert DIGEST in record["body"] and record["event"]["kind"] == 27235
    assert verify_signature_record(record, statement_hash=DIGEST, signer_npub=alice_npub) is None
    # A body that does not name the statement, and an unsigned request, are no signature at all.
    assert elsewhere is not None and verify_signature_record(
        elsewhere, statement_hash=DIGEST, signer_npub=alice_npub) is not None
    assert unsigned is None
    # Only alice signed it, only this body, only this statement.
    assert "not this signer" in verify_signature_record(record, statement_hash=DIGEST, signer_npub=bob_npub)
    tampered = {**record, "body": record["body"].replace(DIGEST, statement_hash("something else"))}
    assert verify_signature_record(tampered, statement_hash=DIGEST, signer_npub=alice_npub) is not None
    assert verify_signature_record(record, statement_hash=statement_hash("other"), signer_npub=alice_npub)


# ---------------------------------------------------------------------------
# A signed reply, witnessed
# ---------------------------------------------------------------------------


def _reply(signer: PrivateKey, operator: PrivateKey) -> dict[str, Any]:
    event = Event(pubkey=signer.public_key.hex(), kind=4, content="ciphertext?iv=x",
                  tags=[["p", operator.public_key.hex()]], created_at=1_800_000_000)
    event.sign(signer.hex())
    return event.to_dict()


def _record(signer: PrivateKey, operator: PrivateKey, digest: str = DIGEST) -> dict[str, Any]:
    reply = _reply(signer, operator)
    witness = sign_signature_witness(
        operator.bech32(), signer_npub=signer.public_key.bech32(), statement_hash=digest,
        reply_event_id=reply["id"], challenge_hash="cd" * 32,
    )
    return {"method": "dm_reply", "signer": signer.public_key.bech32(), "statement_hash": digest,
            "signed_at": reply["created_at"], "reply_event": reply, "witness": json.loads(witness)}


def test_a_witnessed_reply_is_a_signature_for_that_signer_statement_and_operator():
    alice, bob, operator, impostor = PrivateKey(), PrivateKey(), PrivateKey(), PrivateKey()
    record = _record(alice, operator)
    ok = verify_signature_record(record, statement_hash=DIGEST, signer_npub=alice.public_key.bech32(),
                                 operator_npub=operator.public_key.bech32())
    assert ok is None
    # The witness never passes as a signed request: it names no method and no payload.
    assert not [t for t in record["witness"]["tags"] if t[0] in ("method", "payload")]
    assert verify_signature_record(record, statement_hash=DIGEST, signer_npub=bob.public_key.bech32())
    assert "not this operator" in verify_signature_record(
        record, statement_hash=DIGEST, signer_npub=alice.public_key.bech32(),
        operator_npub=impostor.public_key.bech32())


def test_a_witness_cannot_be_moved_to_another_reply_or_statement():
    alice, operator = PrivateKey(), PrivateKey()
    record = _record(alice, operator)
    other_reply = _reply(alice, operator) | {}
    other = Event(pubkey=alice.public_key.hex(), kind=4, content="another", tags=[], created_at=1_800_000_001)
    other.sign(alice.hex())
    swapped = {**record, "reply_event": other.to_dict()}
    assert "does not bind" in verify_signature_record(
        swapped, statement_hash=DIGEST, signer_npub=alice.public_key.bech32())
    elsewhere = _record(alice, operator, digest=statement_hash("a different job"))
    assert verify_signature_record({**elsewhere, "statement_hash": DIGEST}, statement_hash=DIGEST,
                                   signer_npub=alice.public_key.bech32())
    forged = {**record, "reply_event": {**other_reply, "content": "edited after signing"}}
    assert "not this signer" in verify_signature_record(
        forged, statement_hash=DIGEST, signer_npub=alice.public_key.bech32())


def test_a_record_of_an_unknown_kind_is_refused():
    assert verify_signature_record({"method": "fax", "statement_hash": DIGEST}, statement_hash=DIGEST,
                                   signer_npub=PrivateKey().public_key.bech32()) == "unknown signature method"
