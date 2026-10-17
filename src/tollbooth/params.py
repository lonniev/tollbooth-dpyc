"""Tool parameters every DPYC operator shares, described once.

Annotate a tool parameter with one of these and its description reaches the
agent in the tool's input schema. One wording for the whole fleet, so a patron
proving an npub reads the same instructions at every operator.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

DpopToken = Annotated[
    str,
    Field(
        description=(
            "Proof that the caller holds npub. Either the grant envelope that "
            "receive_npub_proof returned, or the raw JSON of a kind-27235 Nostr "
            "event signed by npub: content \"\", a `u` tag holding this tool's "
            "exact name from tools/list, created_at within 60 s of now, and a "
            "random `nonce` tag. Leave it empty when the connection proves npub "
            "— a request signed with npub's key (NIP-98 `Authorization: Nostr`), "
            "or a connection that holds npub (see held_identities)."
        ),
    ),
]

PatronNpub = Annotated[
    str,
    Field(
        description=(
            "The patron's Nostr public key (npub1...). Tool fees are paid from "
            "this patron's balance; dpop_token proves the caller holds it."
        ),
    ),
]
