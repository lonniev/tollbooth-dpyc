"""Bitcoin anchoring tools: anchor_ledger, get_anchor_proof, list_anchors.

These tools periodically commit a Merkle root of all ledger balances to
Bitcoin via OpenTimestamps. Patrons can independently verify their balance
was included in a Bitcoin-committed hash.

Zero hot-path impact — fully decoupled from credit tool calls.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from tollbooth.ots import MerkleTree, OTSCalendarClient

logger = logging.getLogger(__name__)


async def anchor_ledger_tool(
    vault: Any,
    ots_calendars: list[str] | None = None,
) -> dict[str, Any]:
    """Build a Merkle tree of all ledger balances and submit to OTS calendars.

    Fetches all balances from the vault, builds a SHA-256 Merkle tree,
    submits the root to OTS calendar servers, and stores the anchor
    record in the vault.

    Args:
        vault: A NeonVault instance (must have fetch_all_balances, store_anchor).
        ots_calendars: Optional list of OTS calendar URLs. Uses defaults if None.

    Returns:
        Anchor record with root_hash, leaf_count, receipts, and anchor_id.
    """
    # Fetch all balances from NeonVault
    try:
        entries = await vault.fetch_all_balances()
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch balances: {e}"}

    if not entries:
        return {"success": False, "error": "No balances found — nothing to anchor."}

    # Build Merkle tree
    tree = MerkleTree(entries)
    root_hex = tree.root_hex
    leaf_count = tree.leaf_count

    # Submit to OTS calendars
    ots = OTSCalendarClient(calendars=ots_calendars)
    try:
        receipts = await ots.submit_digest(tree.root)
    finally:
        await ots.close()

    # Build snapshot summary (npub → balance, no full ledger_json)
    snapshot: dict[str, Any] = {
        "entry_count": len(entries),
        "npubs": [npub for npub, _ in sorted(entries, key=lambda e: e[0])],
    }

    # Store anchor record
    now = datetime.now(timezone.utc)
    try:
        anchor_id = await vault.store_anchor(
            root_hash=root_hex,
            leaf_count=leaf_count,
            status="submitted" if receipts else "no_receipts",
            ots_receipts_json=json.dumps(receipts) if receipts else None,
            snapshot_json=json.dumps(snapshot),
            leaf_hashes_json=json.dumps(tree.get_leaf_hashes()),
            created_at=now.isoformat(),
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Anchor computed but failed to store: {e}",
            "root_hash": root_hex,
            "leaf_count": leaf_count,
            "receipts": len(receipts),
        }

    return {
        "success": True,
        "anchor_id": anchor_id,
        "root_hash": root_hex,
        "leaf_count": leaf_count,
        "calendars_submitted": len(receipts),
        "calendars_attempted": len(ots._calendars),
        "status": "submitted" if receipts else "no_receipts",
        "created_at": now.isoformat(),
        "message": (
            f"Anchored {leaf_count} ledger balances. "
            f"Merkle root: {root_hex[:16]}... "
            f"Submitted to {len(receipts)}/{len(ots._calendars)} OTS calendars. "
            f"Bitcoin confirmation expected in 1-6 hours."
        ),
    }


async def get_anchor_proof_tool(
    vault: Any,
    anchor_id: str,
    npub: str,
) -> dict[str, Any]:
    """Generate a Merkle inclusion proof for a patron's balance in an anchor.

    Reconstructs the Merkle tree from stored leaf hashes and generates
    an inclusion proof for the specified npub.

    Args:
        vault: A NeonVault instance (must have fetch_anchor).
        anchor_id: The anchor record ID.
        npub: The patron's Nostr public key.

    Returns:
        Inclusion proof with verification guide, or error if not found.
    """
    try:
        anchor = await vault.fetch_anchor(anchor_id)
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch anchor: {e}"}

    if not anchor:
        return {"success": False, "error": f"Anchor {anchor_id} not found."}

    # Reconstruct tree from stored leaf hashes
    try:
        leaf_hashes = json.loads(anchor["leaf_hashes_json"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {"success": False, "error": f"Failed to parse leaf hashes: {e}"}

    tree = MerkleTree.from_leaf_hashes(leaf_hashes)

    # Verify reconstructed root matches stored root
    if tree.root_hex != anchor["root_hash"]:
        return {
            "success": False,
            "error": "Integrity error: reconstructed root does not match stored root.",
        }

    # Generate inclusion proof
    proof = tree.get_proof(npub)
    if proof is None:
        return {
            "success": False,
            "error": f"npub {npub} not found in anchor {anchor_id}.",
        }

    if not proof.verify():
        return {
            "success": False,
            "error": "Internal error: generated proof fails verification.",
        }

    return {
        "success": True,
        "anchor_id": anchor_id,
        "root_hash": anchor["root_hash"],
        "anchor_status": anchor.get("status", "unknown"),
        "anchor_created_at": anchor.get("created_at"),
        "proof": proof.to_dict(),
        "verified": True,
        "verification_guide": (
            "This proof demonstrates that your balance (identified by npub) "
            "was included in a Merkle tree whose root was submitted to Bitcoin "
            "via OpenTimestamps. To independently verify:\n\n"
            "1. Recompute: leaf_hash = SHA256(your_npub + ':' + SHA256(your_ledger_json))\n"
            "2. Walk the sibling path: for each sibling, concatenate in the specified "
            "position (left/right) and SHA256 the result\n"
            "3. The final hash should equal root_hash\n"
            "4. The root_hash was submitted to OTS calendars and will be anchored "
            "in a Bitcoin block header via OpenTimestamps"
        ),
    }


async def list_anchors_tool(
    vault: Any,
    limit: int = 20,
    status: str | None = None,
) -> dict[str, Any]:
    """List recent anchor records.

    Args:
        vault: A NeonVault instance (must have list_anchors).
        limit: Maximum number of anchors to return (default 20).
        status: Optional filter by status (e.g., "submitted", "confirmed").

    Returns:
        List of anchor summaries.
    """
    try:
        anchors = await vault.list_anchors(limit=limit, status=status)
    except Exception as e:
        return {"success": False, "error": f"Failed to list anchors: {e}"}

    items: list[dict[str, Any]] = []
    for a in anchors:
        item: dict[str, Any] = {
            "anchor_id": str(a["id"]),
            "root_hash": a["root_hash"],
            "leaf_count": a["leaf_count"],
            "status": a["status"],
            "created_at": a.get("created_at"),
        }
        if a.get("confirmed_at"):
            item["confirmed_at"] = a["confirmed_at"]
        # Include receipt count if available
        if a.get("ots_receipts_json"):
            try:
                receipts = json.loads(a["ots_receipts_json"])
                item["receipt_count"] = len(receipts)
            except (json.JSONDecodeError, TypeError):
                item["receipt_count"] = 0
        else:
            item["receipt_count"] = 0
        items.append(item)

    return {
        "success": True,
        "count": len(items),
        "anchors": items,
    }
