"""Patron proof constraint — require the patron's own key on high-value tools.

When configured on a tool's pipeline, the request must be signed by the nsec
of the npub being debited (NIP-98). A holder connection, which proves an npub
by a grant earned once, does not satisfy it. This protects high-value
operations from unauthorized spend.

Low-fee tools don't need this overhead — operators configure it
selectively on tools where the stakes justify the friction.
"""

from __future__ import annotations

from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ConstraintSchema,
    ToolConstraint,
)


class PatronProofConstraint(ToolConstraint):
    """Require the request to be signed by the patron's own key."""

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        from tollbooth.holder_auth import signed_npub

        if signed_npub() == context.patron.npub:
            return ConstraintResult(
                allowed=True, reason="patron_proof", message="Patron identity verified.",
            )
        return ConstraintResult(
            allowed=False,
            reason="patron_proof_required",
            message=(
                "This tool requires the patron's own key: sign the "
                "request with the patron's nsec."
            ),
        )

    def describe(self) -> str:
        return "Patron's own key: each request signed with the patron's nsec"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "patron_proof"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatronProofConstraint:
        return cls()

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="patron_proof",
            category="Access",
            description=(
                "Require each request to be signed with the patron's own nsec. "
                "Use for high-value tools where unauthorized spend must be "
                "prevented: the operator verifies the request's signature "
                "matches the npub being debited."
            ),
            params=[],
        )
