"""Supporting types for DPYC actor protocols.

Defines the ActorRole enum, ToolPath enum, and ToolPathInfo frozen
dataclass used by OperatorProtocol, AuthorityProtocol, and OracleProtocol
to describe their tool surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActorRole(str, Enum):
    """DPYC ecosystem actor types."""

    OPERATOR = "operator"
    AUTHORITY = "authority"
    ORACLE = "oracle"


class ToolPath(str, Enum):
    """Categorizes a tool's latency and dependency profile.

    hot — local computation only, no outbound HTTP.
    cold — requires external I/O (BTCPay, GitHub, etc.).
    delegation — forwards to another actor via MCP-to-MCP HTTP.
    """

    HOT = "hot"
    COLD = "cold"
    DELEGATION = "delegation"


@dataclass(frozen=True)
class ToolPathInfo:
    """Metadata for a single tool exposed by a DPYC actor.

    Parameters
    ----------
    tool_name : str
        The tool's registration name (without slug prefix).
    path : ToolPath
        Latency/dependency category.
    delegates_to : ActorRole | None
        If this tool delegates, which actor handles it.
    requires_auth : bool
        Whether the caller must be authenticated.
    cost_tier : str
        Billing tier: FREE, READ, WRITE, or HEAVY.
    agent_hint : str
        Short routing guidance surfaced in MCP tool descriptions.
    supersedes : str
        Migration hint for AI conversations that learned old workflows.
    """

    tool_name: str
    path: ToolPath
    delegates_to: ActorRole | None = None
    requires_auth: bool = True
    cost_tier: str = "FREE"
    agent_hint: str = ""
    supersedes: str = ""


@dataclass(frozen=True)
class ObsoletePractice:
    """A pattern that agents should stop attempting.

    MCP operators should surface these in ``session_status`` or a
    dedicated ``get_practices`` tool so that AI agents receive a
    proactive "unlearn this" signal at session start, rather than
    discovering deprecated patterns through trial-and-error.

    Parameters
    ----------
    pattern : str
        The obsolete tool call or workflow (e.g. ``"activate_session(passphrase)"``).
    replaced_by : str
        The current correct approach.
    reason : str
        Why the old pattern was removed.
    deprecated_since : str
        ISO date when the pattern became obsolete.
    """

    pattern: str
    replaced_by: str
    reason: str
    deprecated_since: str
