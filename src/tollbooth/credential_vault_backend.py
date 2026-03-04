"""Abstract persistence interface for credential storage.

Defines the CredentialVaultBackend Protocol that NostrCredentialExchange
uses to persist credentials across sessions.  Concrete implementations
(e.g., TheBrainVault, NeonVault) live elsewhere.

Separate from VaultBackend (ledger/commerce state).  Credential vaults
store opaque encrypted blobs keyed by (service, npub).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CredentialVaultBackend(Protocol):
    """Async persistence backend for encrypted credential blobs.

    Implementations store and retrieve opaque strings (encrypted by the
    caller before storage).  The vault never sees plaintext credentials.
    """

    async def store_credentials(
        self, service: str, npub: str, encrypted_blob: str,
    ) -> None:
        """Store an encrypted credential blob.

        Overwrites any existing blob for the same (service, npub) pair.
        """
        ...

    async def fetch_credentials(
        self, service: str, npub: str,
    ) -> str | None:
        """Fetch an encrypted credential blob.

        Returns None if no credentials are stored for this pair.
        """
        ...

    async def delete_credentials(
        self, service: str, npub: str,
    ) -> bool:
        """Delete stored credentials.

        Returns True if credentials were found and deleted, False if
        no credentials existed for this pair.
        """
        ...


@runtime_checkable
class SessionBindingBackend(Protocol):
    """Async persistence for caller_id → npub session bindings.

    Allows silent session restoration on serverless cold starts.
    The binding maps a transport-layer caller ID (e.g. Horizon user_id)
    to the DPYC npub that was last used for credential receipt.

    Separate from CredentialVaultBackend so existing vault implementations
    remain compatible without changes.
    """

    async def store_session_binding(
        self, caller_id: str, service: str, npub: str,
    ) -> None:
        """Persist a session binding (upserts on conflict)."""
        ...

    async def fetch_session_binding(
        self, caller_id: str, service: str,
    ) -> str | None:
        """Look up the npub for a caller+service pair.  Returns None if missing."""
        ...

    async def delete_session_binding(
        self, caller_id: str, service: str,
    ) -> bool:
        """Remove a session binding.  Returns True if found and deleted."""
        ...
