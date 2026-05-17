"""Authority curator onboarding state machine and credential templates.

Tracks the in-memory state of a single onboarding flow (claim → approval)
and provides the CredentialTemplate definitions used by the Nostr DM
challenge-response protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from tollbooth.credential_templates import CredentialTemplate, FieldSpec

# ---------------------------------------------------------------------------
# Credential templates for the Secure Courier exchange
# ---------------------------------------------------------------------------

AUTHORITY_CLAIM_TEMPLATE = CredentialTemplate(
    service="authority_claim",
    version=1,
    fields={"claim": FieldSpec(required=True, sensitive=False)},
    description="Authority curator claim challenge — candidate proves npub ownership",
)

AUTHORITY_APPROVAL_TEMPLATE = CredentialTemplate(
    service="authority_approval",
    version=1,
    fields={"approval": FieldSpec(required=True, sensitive=False)},
    description="Authority curator approval — parent Authority approves the candidate",
)

ONBOARDING_TEMPLATES: dict[str, CredentialTemplate] = {
    "authority_claim": AUTHORITY_CLAIM_TEMPLATE,
    "authority_approval": AUTHORITY_APPROVAL_TEMPLATE,
}

# ---------------------------------------------------------------------------
# Onboarding state
# ---------------------------------------------------------------------------

_DEFAULT_TTL_SECONDS = 600  # 10 minutes


@dataclass
class OnboardingChallenge:
    """Represents an in-progress onboarding attempt."""

    candidate_npub: str
    phase: str  # "claim" | "approval"
    parent_npub: str | None = None  # registered approver (Prime, NA, etc.)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = _DEFAULT_TTL_SECONDS

    @property
    def expired(self) -> bool:
        return time.time() > self.created_at + self.ttl_seconds


class OnboardingState:
    """Track which candidate is being onboarded and at what phase.

    Only one onboarding in progress at a time per Authority instance.
    """

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._active: OnboardingChallenge | None = None
        self._ttl = ttl_seconds

    def start_claim(self, candidate_npub: str) -> OnboardingChallenge:
        """Begin a new onboarding by entering the claim phase.

        Raises ``ValueError`` if an onboarding is already active
        (and not expired).
        """
        self._prune()
        if self._active is not None:
            raise ValueError(
                f"Onboarding already in progress for "
                f"{self._active.candidate_npub} (phase={self._active.phase}). "
                f"Complete or wait for expiry."
            )
        challenge = OnboardingChallenge(
            candidate_npub=candidate_npub,
            phase="claim",
            ttl_seconds=self._ttl,
        )
        self._active = challenge
        return challenge

    def promote_to_approval(self, parent_npub: str) -> OnboardingChallenge:
        """Promote the current claim to the approval phase.

        Resets the TTL so the parent Authority has a fresh window to
        respond. Raises ``ValueError`` if no active claim or wrong phase.
        """
        self._prune()
        if self._active is None:
            raise ValueError("No active onboarding to promote.")
        if self._active.phase != "claim":
            raise ValueError(
                f"Cannot promote: current phase is '{self._active.phase}', expected 'claim'."
            )
        self._active.phase = "approval"
        self._active.parent_npub = parent_npub
        self._active.created_at = time.time()  # fresh TTL
        return self._active

    def complete(self) -> None:
        """Mark the onboarding as complete and clear state."""
        self._active = None

    def get(self) -> OnboardingChallenge | None:
        """Return the active challenge, or None if none/expired."""
        self._prune()
        return self._active

    def _prune(self) -> None:
        if self._active is not None and self._active.expired:
            self._active = None
