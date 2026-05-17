"""Tollbooth Authority extension — shared protocol code for any Authority MCP.

An Authority MCP service is a thin actor-specific wrapper around the
shared protocol logic that lives here. Each Authority deployment imports
the modules it needs (onboarding state machine, Schnorr signer, replay
tracker, Neon tenant provisioning, etc.) so this code is defined exactly
once and consumed everywhere.

**v0.21.0 — Phase A (this release):** the six supporting modules are
promoted out of each Authority repo into this shared package:

- ``tollbooth.authority.onboarding`` — claim/approval state machine + DM templates
- ``tollbooth.authority.nostr_signing`` — Schnorr-native certificate signer
- ``tollbooth.authority.replay`` — anti-replay JTI tracker
- ``tollbooth.authority.tenant_provisioner`` — per-operator Neon schemas and LOGIN roles
- ``tollbooth.authority.role_migration`` — one-shot CLI to migrate legacy schemas
- ``tollbooth.authority.settings`` — pydantic-settings ``AuthoritySettings``

**Planned Phase B (v0.22.0):** promote the 10 Authority ``@tool``
definitions from each Authority repo's ``server.py`` into a
``register_authority_tools(mcp, runtime)`` function here. After that,
each Authority repo's ``server.py`` collapses to ~30 lines of
actor-specific configuration (name, instructions, identity).
"""

from __future__ import annotations

from tollbooth.authority.nostr_signing import AuthorityNostrSigner, NOSTR_CERT_KIND
from tollbooth.authority.onboarding import (
    AUTHORITY_APPROVAL_TEMPLATE,
    AUTHORITY_CLAIM_TEMPLATE,
    ONBOARDING_TEMPLATES,
    OnboardingChallenge,
    OnboardingState,
)
from tollbooth.authority.replay import ReplayTracker
from tollbooth.authority.settings import AuthoritySettings

__all__ = [
    "AUTHORITY_APPROVAL_TEMPLATE",
    "AUTHORITY_CLAIM_TEMPLATE",
    "AuthorityNostrSigner",
    "AuthoritySettings",
    "NOSTR_CERT_KIND",
    "ONBOARDING_TEMPLATES",
    "OnboardingChallenge",
    "OnboardingState",
    "ReplayTracker",
]
