"""Operator proof — backward-compatible re-exports from identity_proof.

All proof logic lives in :mod:`tollbooth.identity_proof`. This module
re-exports the operator-specific names so existing callers don't break.
"""

from tollbooth.identity_proof import (  # noqa: F401
    PROOF_EVENT_KIND as OPERATOR_PROOF_KIND,
    DEFAULT_WINDOW_SECONDS as PROOF_WINDOW_SECONDS,
    create_proof as create_operator_proof,
    verify_proof as verify_identity_proof,
)


def verify_operator_proof(
    proof_json: str,
    operator_npub: str,
    tool_name: str,
) -> bool:
    """Verify a kind-27235 operator proof. Delegates to verify_proof."""
    return verify_identity_proof(proof_json, operator_npub, tool_name)
