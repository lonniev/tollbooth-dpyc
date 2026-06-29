"""Patron proof constraint — require Schnorr signature for high-value tools.

When configured on a tool's pipeline, the patron must include a signed
Nostr kind-27235 event proving they hold the nsec for the npub being
debited. This protects high-value operations from unauthorized spend.

Low-fee tools don't need this overhead — operators configure it
selectively on tools where the stakes justify the friction.
"""

from __future__ import annotations

from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ConstraintSchema,
    ParamSchema,
    ToolConstraint,
)


class PatronProofConstraint(ToolConstraint):
    """Require the patron to prove npub ownership via Schnorr signature.

    Parameters
    ----------
    window_seconds:
        Maximum age of the proof event (default 120 seconds).
        Longer windows reduce friction for multi-step workflows.
    """

    def __init__(self, window_seconds: int = 120) -> None:
        self.window_seconds = window_seconds

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        if not context.dpop_token:
            return ConstraintResult(
                allowed=False,
                reason="patron_proof_required",
                message=(
                    "This tool requires patron identity proof. "
                    "Sign a kind-27235 Nostr event with your nsec "
                    "and pass it as the dpop_token parameter."
                ),
            )

        from tollbooth.identity_proof import verify_proof

        if not verify_proof(
            context.dpop_token,
            context.patron.npub,
            context.env.tool_name,
            window_seconds=self.window_seconds,
        ):
            return ConstraintResult(
                allowed=False,
                reason="patron_proof_invalid",
                message=(
                    "Patron identity proof is invalid — the signature "
                    "does not match, the tool name is wrong, or the "
                    "proof has expired."
                ),
            )

        return ConstraintResult(
            allowed=True,
            reason="patron_proof",
            message="Patron identity verified.",
        )

    def describe(self) -> str:
        return f"Patron dpop_token: Schnorr signature required (window {self.window_seconds}s)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "patron_proof",
            "window_seconds": self.window_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatronProofConstraint:
        return cls(
            window_seconds=int(data.get("window_seconds", 120)),
        )

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="patron_proof",
            category="Access",
            description=(
                "Require the patron to prove npub ownership via Schnorr "
                "signature on each call. Use for high-value tools where "
                "unauthorized spend must be prevented. The patron signs "
                "a Nostr kind-27235 event with their nsec — the operator "
                "verifies the signature matches the npub being debited."
            ),
            params=[
                ParamSchema(
                    name="window_seconds",
                    type="int",
                    required=False,
                    default=120,
                    description="Maximum age of the proof event in seconds.",
                ),
            ],
        )
