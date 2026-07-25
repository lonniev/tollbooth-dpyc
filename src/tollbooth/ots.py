"""OpenTimestamps Bitcoin anchoring — Merkle tree + OTS calendar client.

Provides three components for anchoring ledger state to Bitcoin:

- ``MerkleTree``: SHA-256 Merkle tree over sorted (npub, ledger_json) entries
- ``InclusionProof``: Verifiable proof that a specific leaf is in the tree
- ``OTSCalendarClient``: Async HTTP client for OTS calendar servers

Zero new dependencies: stdlib ``hashlib`` for hashing, existing ``httpx``
for HTTP calls to OTS calendar servers.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default OTS calendar servers (free, no API keys, no rate limits)
DEFAULT_CALENDARS: list[str] = [
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
    "https://ots.btc.catallaxy.com",
]


def _sha256(data: bytes) -> bytes:
    """Compute SHA-256 digest."""
    return hashlib.sha256(data).digest()


@dataclass(frozen=True)
class InclusionProof:
    """Merkle inclusion proof for a single leaf.

    Proves that ``leaf_hash`` is included in a Merkle tree with
    ``root_hash`` by providing the sibling hashes along the path
    from leaf to root.
    """

    leaf_hash: str
    leaf_index: int
    tree_size: int
    root_hash: str
    siblings: list[dict[str, str]]  # [{"hash": hex, "position": "left"|"right"}, ...]

    def verify(self) -> bool:
        """Recompute leaf→root and check against stored root_hash."""
        current = bytes.fromhex(self.leaf_hash)
        for sib in self.siblings:
            sib_hash = bytes.fromhex(sib["hash"])
            if sib["position"] == "left":
                current = _sha256(sib_hash + current)
            else:
                current = _sha256(current + sib_hash)
        return current.hex() == self.root_hash

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "leaf_hash": self.leaf_hash,
            "leaf_index": self.leaf_index,
            "tree_size": self.tree_size,
            "root_hash": self.root_hash,
            "siblings": self.siblings,
        }


class MerkleTree:
    """SHA-256 Merkle tree over sorted (npub, ledger_json) entries.

    Leaf computation: ``SHA256(npub + ":" + SHA256(ledger_json))``
    Entries are sorted by npub for deterministic construction.
    Odd leaf count: duplicate last leaf.
    """

    def __init__(self, entries: list[tuple[str, str]]) -> None:
        """Build a Merkle tree from (npub, ledger_json) pairs.

        Args:
            entries: List of (npub, ledger_json) tuples. Sorted by npub
                internally for deterministic ordering.
        """
        sorted_entries = sorted(entries, key=lambda e: e[0])
        self._leaf_hashes: list[bytes] = []
        self._npub_to_index: dict[str, int] = {}

        for i, (npub, ledger_json) in enumerate(sorted_entries):
            inner = _sha256(ledger_json.encode("utf-8"))
            leaf = _sha256(npub.encode("utf-8") + b":" + inner)
            self._leaf_hashes.append(leaf)
            self._npub_to_index[npub] = i

        self._levels: list[list[bytes]] = []
        self._build()

    @classmethod
    def from_leaf_hashes(cls, leaf_data: list[dict[str, str]]) -> MerkleTree:
        """Rebuild a MerkleTree from stored leaf hashes.

        Args:
            leaf_data: List of ``{"npub": leaf_hash_hex}`` single-key dicts,
                one per leaf, in original sorted order.

        Returns:
            A MerkleTree with the same root as the original.
        """
        tree = cls.__new__(cls)
        tree._leaf_hashes = []
        tree._npub_to_index = {}
        tree._levels = []

        for i, entry in enumerate(leaf_data):
            for npub, hash_hex in entry.items():
                tree._leaf_hashes.append(bytes.fromhex(hash_hex))
                tree._npub_to_index[npub] = i

        tree._build()
        return tree

    def _build(self) -> None:
        """Build the Merkle tree levels from leaf hashes."""
        if not self._leaf_hashes:
            self._levels = []
            return

        level = list(self._leaf_hashes)
        self._levels = [level]

        while len(level) > 1:
            # Duplicate last hash if odd count
            if len(level) % 2 == 1:
                level = level + [level[-1]]

            next_level: list[bytes] = []
            for j in range(0, len(level), 2):
                combined = _sha256(level[j] + level[j + 1])
                next_level.append(combined)

            self._levels.append(next_level)
            level = next_level

    @property
    def root(self) -> bytes:
        """The Merkle root as raw bytes. Empty bytes if tree is empty."""
        if not self._levels:
            return b""
        return self._levels[-1][0]

    @property
    def root_hex(self) -> str:
        """The Merkle root as a hex string. Empty string if tree is empty."""
        if not self._levels:
            return ""
        return self._levels[-1][0].hex()

    @property
    def leaf_count(self) -> int:
        """Number of leaves in the tree."""
        return len(self._leaf_hashes)

    def get_leaf_hashes(self) -> list[dict[str, str]]:
        """Export leaf hashes for storage.

        Returns:
            List of ``{"npub": leaf_hash_hex}`` dicts in tree order.
        """
        result: list[dict[str, str]] = []
        # Invert the index to get npub by position
        index_to_npub = {v: k for k, v in self._npub_to_index.items()}
        for i, h in enumerate(self._leaf_hashes):
            npub = index_to_npub.get(i, f"unknown_{i}")
            result.append({npub: h.hex()})
        return result

    def get_proof(self, npub: str) -> InclusionProof | None:
        """Generate an inclusion proof for a specific npub.

        Returns None if the npub is not in the tree.
        """
        idx = self._npub_to_index.get(npub)
        if idx is None:
            return None

        siblings: list[dict[str, str]] = []
        current_idx = idx

        for level_idx in range(len(self._levels) - 1):
            level = self._levels[level_idx]
            # Pad level for odd count (same as build)
            padded = list(level)
            if len(padded) % 2 == 1:
                padded.append(padded[-1])

            if current_idx % 2 == 0:
                # Current is left child, sibling is right
                sib_idx = current_idx + 1
                siblings.append({
                    "hash": padded[sib_idx].hex(),
                    "position": "right",
                })
            else:
                # Current is right child, sibling is left
                sib_idx = current_idx - 1
                siblings.append({
                    "hash": padded[sib_idx].hex(),
                    "position": "left",
                })

            current_idx = current_idx // 2

        return InclusionProof(
            leaf_hash=self._leaf_hashes[idx].hex(),
            leaf_index=idx,
            tree_size=self.leaf_count,
            root_hash=self.root_hex,
            siblings=siblings,
        )


class OTSCalendarClient:
    """Async HTTP client for OpenTimestamps calendar servers.

    Submits SHA-256 digests and retrieves timestamp proofs via the
    OTS calendar HTTP protocol.
    """

    def __init__(
        self,
        calendars: list[str] | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._calendars = calendars or list(DEFAULT_CALENDARS)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def submit_digest(self, digest: bytes) -> list[dict[str, str]]:
        """Submit a SHA-256 digest to all configured OTS calendars.

        Posts the raw 32-byte digest to each calendar's ``/digest``
        endpoint. Returns a list of successful receipt records.

        Never raises — logs failures and returns partial results.

        Args:
            digest: Raw 32-byte SHA-256 digest.

        Returns:
            List of ``{"calendar": url, "receipt_b64": base64_receipt}``
            for each successful submission.
        """
        results: list[dict[str, str]] = []

        for calendar in self._calendars:
            url = f"{calendar.rstrip('/')}/digest"
            try:
                resp = await self._client.post(
                    url,
                    content=digest,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/vnd.opentimestamps.v1",
                    },
                )
                if resp.status_code == 200:
                    receipt_b64 = base64.b64encode(resp.content).decode("ascii")
                    results.append({
                        "calendar": calendar,
                        "receipt_b64": receipt_b64,
                    })
                    logger.info("OTS digest submitted to %s (receipt: %d bytes)", calendar, len(resp.content))
                else:
                    logger.warning(
                        "OTS calendar %s returned HTTP %d: %s",
                        calendar, resp.status_code, resp.text[:200],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("OTS calendar %s unreachable: %s", calendar, exc)

        return results

    async def upgrade_receipt(
        self,
        commitment_hex: str,
        calendar_url: str,
    ) -> bytes | None:
        """Attempt to upgrade an OTS receipt to a Bitcoin-confirmed proof.

        Args:
            commitment_hex: The hex-encoded commitment (Merkle root).
            calendar_url: The calendar server that issued the receipt.

        Returns:
            Upgraded proof bytes on success, None if not yet confirmed (404).
        """
        url = f"{calendar_url.rstrip('/')}/timestamp/{commitment_hex}"
        try:
            resp = await self._client.get(
                url,
                headers={"Accept": "application/vnd.opentimestamps.v1"},
            )
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                return None
            logger.warning(
                "OTS upgrade from %s returned HTTP %d",
                calendar_url, resp.status_code,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("OTS upgrade from %s failed: %s", calendar_url, exc)
            return None

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
