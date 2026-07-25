"""Tests for OpenTimestamps Bitcoin anchoring: MerkleTree, InclusionProof, OTSCalendarClient, anchor tools."""

from __future__ import annotations

import base64
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tollbooth.ots import (
    DEFAULT_CALENDARS,
    InclusionProof,
    MerkleTree,
    OTSCalendarClient,
    _sha256,
)
from tollbooth.tools.anchors import (
    anchor_ledger_tool,
    get_anchor_proof_tool,
    list_anchors_tool,
)

# ---------------------------------------------------------------------------
# MerkleTree tests
# ---------------------------------------------------------------------------

class TestMerkleTree:
    """Tests for MerkleTree construction, determinism, and proof generation."""

    def _make_entries(self, n: int) -> list[tuple[str, str]]:
        """Generate n deterministic (npub, ledger_json) pairs."""
        return [(f"npub1{'a' * i}{chr(97 + i)}", f'{{"balance": {i * 100}}}') for i in range(n)]

    def test_single_leaf(self):
        entries = self._make_entries(1)
        tree = MerkleTree(entries)
        assert tree.leaf_count == 1
        assert len(tree.root) == 32
        assert len(tree.root_hex) == 64

    def test_two_leaves(self):
        entries = self._make_entries(2)
        tree = MerkleTree(entries)
        assert tree.leaf_count == 2
        assert len(tree.root) == 32

    def test_three_leaves_odd_duplication(self):
        entries = self._make_entries(3)
        tree = MerkleTree(entries)
        assert tree.leaf_count == 3
        assert len(tree.root) == 32

    def test_four_leaves(self):
        entries = self._make_entries(4)
        tree = MerkleTree(entries)
        assert tree.leaf_count == 4

    def test_hundred_leaves(self):
        entries = [(f"npub1user{i:04d}", f'{{"balance": {i}}}') for i in range(100)]
        tree = MerkleTree(entries)
        assert tree.leaf_count == 100
        assert len(tree.root) == 32

    def test_empty_tree(self):
        tree = MerkleTree([])
        assert tree.leaf_count == 0
        assert tree.root == b""
        assert tree.root_hex == ""

    def test_deterministic_ordering(self):
        """Tree should produce the same root regardless of input order."""
        entries = self._make_entries(5)
        tree1 = MerkleTree(entries)
        tree2 = MerkleTree(list(reversed(entries)))
        assert tree1.root_hex == tree2.root_hex

    def test_leaf_hash_computation(self):
        """Verify leaf = SHA256(npub + ":" + SHA256(ledger_json))."""
        npub = "npub1test"
        ledger = '{"balance": 500}'
        tree = MerkleTree([(npub, ledger)])

        inner = hashlib.sha256(ledger.encode()).digest()
        expected = hashlib.sha256(npub.encode() + b":" + inner).digest()
        assert tree._leaf_hashes[0] == expected

    def test_get_leaf_hashes_roundtrip(self):
        """Build tree → export leaf hashes → rebuild → same root."""
        entries = self._make_entries(7)
        original = MerkleTree(entries)
        exported = original.get_leaf_hashes()

        rebuilt = MerkleTree.from_leaf_hashes(exported)
        assert rebuilt.root_hex == original.root_hex
        assert rebuilt.leaf_count == original.leaf_count

    def test_get_leaf_hashes_large_roundtrip(self):
        """Roundtrip with 100 leaves."""
        entries = [(f"npub1user{i:04d}", f'{{"balance": {i}}}') for i in range(100)]
        original = MerkleTree(entries)
        exported = original.get_leaf_hashes()
        rebuilt = MerkleTree.from_leaf_hashes(exported)
        assert rebuilt.root_hex == original.root_hex

    def test_different_data_different_root(self):
        """Different entries must produce different roots."""
        tree1 = MerkleTree([("npub1a", '{"x": 1}')])
        tree2 = MerkleTree([("npub1a", '{"x": 2}')])
        assert tree1.root_hex != tree2.root_hex


# ---------------------------------------------------------------------------
# InclusionProof tests
# ---------------------------------------------------------------------------

class TestInclusionProof:
    """Tests for InclusionProof verification."""

    def _make_entries(self, n: int) -> list[tuple[str, str]]:
        return [(f"npub1user{i:04d}", f'{{"balance": {i * 100}}}') for i in range(n)]

    def test_verify_all_leaves_two(self):
        entries = self._make_entries(2)
        tree = MerkleTree(entries)
        sorted_npubs = sorted(e[0] for e in entries)
        for npub in sorted_npubs:
            proof = tree.get_proof(npub)
            assert proof is not None
            assert proof.verify() is True

    def test_verify_all_leaves_four(self):
        entries = self._make_entries(4)
        tree = MerkleTree(entries)
        for npub, _ in entries:
            proof = tree.get_proof(npub)
            assert proof is not None
            assert proof.verify() is True

    def test_verify_all_leaves_odd(self):
        """Verify proofs for a 5-leaf tree (odd count)."""
        entries = self._make_entries(5)
        tree = MerkleTree(entries)
        for npub, _ in entries:
            proof = tree.get_proof(npub)
            assert proof is not None
            assert proof.verify() is True

    def test_verify_all_leaves_large(self):
        """Verify proofs for a 50-leaf tree."""
        entries = [(f"npub1user{i:04d}", f'{{"balance": {i}}}') for i in range(50)]
        tree = MerkleTree(entries)
        for npub, _ in entries:
            proof = tree.get_proof(npub)
            assert proof is not None, f"No proof for {npub}"
            assert proof.verify() is True, f"Proof failed for {npub}"

    def test_verify_single_leaf(self):
        """Single-leaf tree: proof has no siblings."""
        tree = MerkleTree([("npub1only", '{"x": 1}')])
        proof = tree.get_proof("npub1only")
        assert proof is not None
        assert proof.siblings == []
        assert proof.verify() is True

    def test_tampered_root_fails(self):
        entries = self._make_entries(4)
        tree = MerkleTree(entries)
        proof = tree.get_proof(entries[0][0])
        assert proof is not None
        # Tamper with root
        bad_proof = InclusionProof(
            leaf_hash=proof.leaf_hash,
            leaf_index=proof.leaf_index,
            tree_size=proof.tree_size,
            root_hash="00" * 32,
            siblings=proof.siblings,
        )
        assert bad_proof.verify() is False

    def test_tampered_sibling_fails(self):
        entries = self._make_entries(4)
        tree = MerkleTree(entries)
        proof = tree.get_proof(entries[0][0])
        assert proof is not None
        assert len(proof.siblings) > 0
        # Tamper with first sibling
        bad_siblings = list(proof.siblings)
        bad_siblings[0] = {"hash": "ff" * 32, "position": bad_siblings[0]["position"]}
        bad_proof = InclusionProof(
            leaf_hash=proof.leaf_hash,
            leaf_index=proof.leaf_index,
            tree_size=proof.tree_size,
            root_hash=proof.root_hash,
            siblings=bad_siblings,
        )
        assert bad_proof.verify() is False

    def test_proof_not_found(self):
        tree = MerkleTree([("npub1a", '{"x": 1}')])
        assert tree.get_proof("npub1nonexistent") is None

    def test_to_dict(self):
        tree = MerkleTree([("npub1a", '{"x": 1}'), ("npub1b", '{"x": 2}')])
        proof = tree.get_proof("npub1a")
        assert proof is not None
        d = proof.to_dict()
        assert "leaf_hash" in d
        assert "leaf_index" in d
        assert "tree_size" in d
        assert "root_hash" in d
        assert "siblings" in d
        assert d["tree_size"] == 2


# ---------------------------------------------------------------------------
# OTSCalendarClient tests
# ---------------------------------------------------------------------------

class TestOTSCalendarClient:
    """Tests for OTSCalendarClient HTTP interactions (mocked)."""

    @pytest.mark.asyncio
    async def test_submit_digest_success(self):
        """All calendars return 200 — all receipts collected."""
        fake_receipt = b"\x00\x01\x02\x03"
        client = OTSCalendarClient(calendars=["https://cal1.test", "https://cal2.test"])

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_receipt

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
            digest = _sha256(b"test data")
            results = await client.submit_digest(digest)

        assert len(results) == 2
        for r in results:
            assert "calendar" in r
            assert "receipt_b64" in r
            assert base64.b64decode(r["receipt_b64"]) == fake_receipt

        await client.close()

    @pytest.mark.asyncio
    async def test_submit_digest_partial_failure(self):
        """One calendar fails, one succeeds — partial results returned."""
        fake_receipt = b"\x00\x01"
        client = OTSCalendarClient(calendars=["https://cal1.test", "https://cal2.test"])

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.content = fake_receipt

        fail_resp = MagicMock()
        fail_resp.status_code = 500
        fail_resp.text = "Internal Server Error"

        with patch.object(
            client._client, "post", new_callable=AsyncMock,
            side_effect=[fail_resp, success_resp],
        ):
            results = await client.submit_digest(_sha256(b"test"))

        assert len(results) == 1
        assert results[0]["calendar"] == "https://cal2.test"

        await client.close()

    @pytest.mark.asyncio
    async def test_submit_digest_all_fail(self):
        """All calendars fail — empty list returned, no exception."""
        client = OTSCalendarClient(calendars=["https://cal1.test"])

        with patch.object(
            client._client, "post", new_callable=AsyncMock,
            side_effect=httpx.ConnectError("unreachable"),
        ):
            results = await client.submit_digest(_sha256(b"test"))

        assert results == []
        await client.close()

    @pytest.mark.asyncio
    async def test_upgrade_receipt_success(self):
        """Calendar returns 200 — upgraded proof returned."""
        upgraded = b"\x10\x20\x30"
        client = OTSCalendarClient()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = upgraded

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.upgrade_receipt("abcdef", "https://cal1.test")

        assert result == upgraded
        await client.close()

    @pytest.mark.asyncio
    async def test_upgrade_receipt_not_ready(self):
        """Calendar returns 404 — None returned (not yet confirmed)."""
        client = OTSCalendarClient()

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.upgrade_receipt("abcdef", "https://cal1.test")

        assert result is None
        await client.close()

    @pytest.mark.asyncio
    async def test_upgrade_receipt_network_error(self):
        """Network error during upgrade — None returned, no exception."""
        client = OTSCalendarClient()

        with patch.object(
            client._client, "get", new_callable=AsyncMock,
            side_effect=httpx.ConnectError("unreachable"),
        ):
            result = await client.upgrade_receipt("abcdef", "https://cal1.test")

        assert result is None
        await client.close()

    def test_default_calendars(self):
        client = OTSCalendarClient()
        assert client._calendars == DEFAULT_CALENDARS

    def test_custom_calendars(self):
        custom = ["https://my.calendar.test"]
        client = OTSCalendarClient(calendars=custom)
        assert client._calendars == custom


# ---------------------------------------------------------------------------
# Anchor tool tests
# ---------------------------------------------------------------------------

class TestAnchorLedgerTool:
    """Tests for anchor_ledger_tool."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Happy path: fetch balances, build tree, submit, store."""
        entries = [("npub1alice", '{"balance": 100}'), ("npub1bob", '{"balance": 200}')]

        vault = AsyncMock()
        vault.fetch_all_balances = AsyncMock(return_value=entries)
        vault.store_anchor = AsyncMock(return_value="42")

        with patch("tollbooth.tools.anchors.OTSCalendarClient") as MockOTS:
            mock_ots = AsyncMock()
            mock_ots.submit_digest = AsyncMock(return_value=[
                {"calendar": "https://cal1.test", "receipt_b64": "AAEC"},
            ])
            mock_ots.close = AsyncMock()
            mock_ots._calendars = ["https://cal1.test"]
            MockOTS.return_value = mock_ots

            result = await anchor_ledger_tool(vault, ots_calendars=["https://cal1.test"])

        assert result["success"] is True
        assert result["anchor_id"] == "42"
        assert result["leaf_count"] == 2
        assert result["calendars_submitted"] == 1
        assert result["status"] == "submitted"
        assert len(result["root_hash"]) == 64

        # Verify store_anchor was called with correct args
        vault.store_anchor.assert_called_once()
        call_kwargs = vault.store_anchor.call_args
        assert call_kwargs.kwargs["leaf_count"] == 2
        assert call_kwargs.kwargs["status"] == "submitted"

    @pytest.mark.asyncio
    async def test_no_balances(self):
        vault = AsyncMock()
        vault.fetch_all_balances = AsyncMock(return_value=[])
        result = await anchor_ledger_tool(vault)
        assert result["success"] is False
        assert "nothing to anchor" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fetch_error(self):
        vault = AsyncMock()
        vault.fetch_all_balances = AsyncMock(side_effect=Exception("DB down"))
        result = await anchor_ledger_tool(vault)
        assert result["success"] is False
        assert "DB down" in result["error"]

    @pytest.mark.asyncio
    async def test_no_ots_receipts(self):
        """OTS submission fails but anchor is still stored."""
        entries = [("npub1x", '{"balance": 1}')]
        vault = AsyncMock()
        vault.fetch_all_balances = AsyncMock(return_value=entries)
        vault.store_anchor = AsyncMock(return_value="99")

        with patch("tollbooth.tools.anchors.OTSCalendarClient") as MockOTS:
            mock_ots = AsyncMock()
            mock_ots.submit_digest = AsyncMock(return_value=[])
            mock_ots.close = AsyncMock()
            mock_ots._calendars = ["https://cal1.test"]
            MockOTS.return_value = mock_ots

            result = await anchor_ledger_tool(vault)

        assert result["success"] is True
        assert result["status"] == "no_receipts"
        assert result["calendars_submitted"] == 0


class TestGetAnchorProofTool:
    """Tests for get_anchor_proof_tool."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Happy path: fetch anchor, rebuild tree, generate proof."""
        entries = [("npub1alice", '{"balance": 100}'), ("npub1bob", '{"balance": 200}')]
        tree = MerkleTree(entries)

        anchor = {
            "id": 42,
            "root_hash": tree.root_hex,
            "leaf_count": 2,
            "status": "submitted",
            "leaf_hashes_json": json.dumps(tree.get_leaf_hashes()),
            "created_at": "2026-02-23T00:00:00+00:00",
        }

        vault = AsyncMock()
        vault.fetch_anchor = AsyncMock(return_value=anchor)

        result = await get_anchor_proof_tool(vault, "42", "npub1alice")
        assert result["success"] is True
        assert result["verified"] is True
        assert result["proof"]["root_hash"] == tree.root_hex
        assert "verification_guide" in result

    @pytest.mark.asyncio
    async def test_anchor_not_found(self):
        vault = AsyncMock()
        vault.fetch_anchor = AsyncMock(return_value=None)
        result = await get_anchor_proof_tool(vault, "999", "npub1x")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_npub_not_in_anchor(self):
        tree = MerkleTree([("npub1a", '{"x": 1}')])
        anchor = {
            "id": 1,
            "root_hash": tree.root_hex,
            "leaf_count": 1,
            "status": "submitted",
            "leaf_hashes_json": json.dumps(tree.get_leaf_hashes()),
        }
        vault = AsyncMock()
        vault.fetch_anchor = AsyncMock(return_value=anchor)

        result = await get_anchor_proof_tool(vault, "1", "npub1nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_corrupted_leaf_hashes(self):
        anchor = {
            "id": 1,
            "root_hash": "aa" * 32,
            "leaf_count": 1,
            "status": "submitted",
            "leaf_hashes_json": "not valid json",
        }
        vault = AsyncMock()
        vault.fetch_anchor = AsyncMock(return_value=anchor)

        result = await get_anchor_proof_tool(vault, "1", "npub1x")
        assert result["success"] is False
        assert "parse" in result["error"].lower() or "failed" in result["error"].lower()


class TestListAnchorsTool:
    """Tests for list_anchors_tool."""

    @pytest.mark.asyncio
    async def test_success(self):
        anchors = [
            {
                "id": 2, "root_hash": "bb" * 32, "leaf_count": 5,
                "status": "submitted", "ots_receipts_json": '[{"calendar": "x"}]',
                "created_at": "2026-02-23T01:00:00+00:00", "confirmed_at": None,
            },
            {
                "id": 1, "root_hash": "aa" * 32, "leaf_count": 3,
                "status": "confirmed", "ots_receipts_json": None,
                "created_at": "2026-02-22T00:00:00+00:00",
                "confirmed_at": "2026-02-22T06:00:00+00:00",
            },
        ]
        vault = AsyncMock()
        vault.list_anchors = AsyncMock(return_value=anchors)

        result = await list_anchors_tool(vault, limit=10)
        assert result["success"] is True
        assert result["count"] == 2
        assert result["anchors"][0]["anchor_id"] == "2"
        assert result["anchors"][0]["receipt_count"] == 1
        assert result["anchors"][1]["receipt_count"] == 0
        assert result["anchors"][1]["confirmed_at"] is not None

    @pytest.mark.asyncio
    async def test_with_status_filter(self):
        vault = AsyncMock()
        vault.list_anchors = AsyncMock(return_value=[])

        result = await list_anchors_tool(vault, status="confirmed")
        assert result["success"] is True
        assert result["count"] == 0
        vault.list_anchors.assert_called_once_with(limit=20, status="confirmed")

    @pytest.mark.asyncio
    async def test_vault_error(self):
        vault = AsyncMock()
        vault.list_anchors = AsyncMock(side_effect=Exception("connection lost"))
        result = await list_anchors_tool(vault)
        assert result["success"] is False
        assert "connection lost" in result["error"]
