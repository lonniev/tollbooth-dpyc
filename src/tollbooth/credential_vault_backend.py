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
