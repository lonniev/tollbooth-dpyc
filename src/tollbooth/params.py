"""Tool parameters every DPYC operator shares, described once.

Annotate a tool parameter with one of these and its description reaches the
agent in the tool's input schema. One wording for the whole fleet, so a patron
naming an npub reads the same instructions at every operator.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

PatronNpub = Annotated[
    str,
    Field(
        description=(
            "The patron's Nostr public key (npub1...). Tool fees are paid from "
            "this patron's balance. The connection proves the caller holds "
            "it: a request signed with npub's key (NIP-98 `Authorization: "
            "Nostr`), or an OAuth connection that holds npub (see "
            "held_identities)."
        ),
    ),
]
