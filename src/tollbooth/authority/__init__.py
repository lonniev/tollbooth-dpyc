"""Tollbooth Authority extension — full mixin for any Authority MCP.

An Authority MCP service is now a thin actor-specific wrapper around the
shared protocol logic that lives here. Each Authority deployment imports
``register_authority_tools`` (analogous to ``register_standard_tools``)
to mount the full set of Authority @tool definitions on its FastMCP
instance, then provides only the actor-specific bits: the FastMCP name,
instructions, identity (nsec via env), Neon URL (via env), and optionally
a custom credential template.

Minimal Authority ``server.py``::

    from fastmcp import FastMCP
    from tollbooth.authority import (
        AUTHORITY_TOOL_REGISTRY,
        OPERATOR_CREDENTIAL_TEMPLATE,
        register_authority_tools,
    )
    from tollbooth.runtime import OperatorRuntime, register_standard_tools
    from tollbooth.tool_identity import STANDARD_IDENTITIES

    mcp = FastMCP("tollbooth-authority-mine", instructions="…")

    runtime = OperatorRuntime(
        tool_registry={**STANDARD_IDENTITIES, **AUTHORITY_TOOL_REGISTRY},
        purchase_mode="direct",
        service_name="My Authority",
        ots_enabled=True,
        operator_credential_template=OPERATOR_CREDENTIAL_TEMPLATE,
    )

    register_standard_tools(mcp, "authority", runtime,
                            service_name="…", service_version="…")
    register_authority_tools(mcp, runtime)

Everything else — onboarding state machine, Schnorr signer, replay
tracker, Neon tenant provisioning, the 10 Authority @tool definitions —
lives in this package and is shared across every Authority deployment.
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
from tollbooth.authority.tools import (
    AUTHORITY_DOMAIN_TOOLS,
    AUTHORITY_TOOL_REGISTRY,
    OPERATOR_CREDENTIAL_TEMPLATE,
    register_authority_tools,
)

__all__ = [
    "AUTHORITY_APPROVAL_TEMPLATE",
    "AUTHORITY_CLAIM_TEMPLATE",
    "AUTHORITY_DOMAIN_TOOLS",
    "AUTHORITY_TOOL_REGISTRY",
    "AuthorityNostrSigner",
    "AuthoritySettings",
    "NOSTR_CERT_KIND",
    "ONBOARDING_TEMPLATES",
    "OPERATOR_CREDENTIAL_TEMPLATE",
    "OnboardingChallenge",
    "OnboardingState",
    "ReplayTracker",
    "register_authority_tools",
]
