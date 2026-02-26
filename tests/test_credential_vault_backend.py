"""Tests for CredentialVaultBackend protocol compliance."""

from tollbooth.credential_vault_backend import CredentialVaultBackend


class InMemoryVault:
    """Minimal in-memory implementation for protocol testing."""

    def __init__(self):
        self._store: dict[str, str] = {}

    async def store_credentials(self, service: str, npub: str, encrypted_blob: str) -> None:
        self._store[f"{service}:{npub}"] = encrypted_blob

    async def fetch_credentials(self, service: str, npub: str) -> str | None:
        return self._store.get(f"{service}:{npub}")

    async def delete_credentials(self, service: str, npub: str) -> bool:
        key = f"{service}:{npub}"
        if key in self._store:
            del self._store[key]
            return True
        return False


class TestCredentialVaultBackend:
    """Protocol compliance tests."""

    def test_in_memory_is_instance(self):
        """InMemoryVault satisfies the CredentialVaultBackend protocol."""
        vault = InMemoryVault()
        assert isinstance(vault, CredentialVaultBackend)

    def test_dict_is_not_instance(self):
        """A plain dict does NOT satisfy the protocol."""
        assert not isinstance({}, CredentialVaultBackend)
