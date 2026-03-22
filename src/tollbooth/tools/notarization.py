"""Bitcoin notarization tools: notarize_ledger, get_notarization_proof, list_notarizations.

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


async def notarize_ledger_tool(
    vault: Any,
    ots_calendars: list[str] | None = None,
) -> dict[str, Any]:
    """Build a Merkle tree of all ledger balances and submit to OTS calendars.

    Creates a Bitcoin-notarized snapshot of all patron balances.

    Args:
        vault: A NeonVault instance (must have fetch_all_balances, store_anchor).
        ots_calendars: Optional list of OTS calendar URLs. Uses defaults if None.

    Returns:
        Notarization record with root_hash, leaf_count, receipts, and notarization_id.
    """
    try:
        entries = await vault.fetch_all_balances()
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch balances: {e}"}

    if not entries:
        return {"success": False, "error": "No balances found — nothing to notarize."}

    tree = MerkleTree(entries)
    root_hex = tree.root_hex
    leaf_count = tree.leaf_count

    ots = OTSCalendarClient(calendars=ots_calendars)
    try:
        receipts = await ots.submit_digest(tree.root)
    finally:
        await ots.close()

    snapshot: dict[str, Any] = {
        "entry_count": len(entries),
        "npubs": [npub for npub, _ in sorted(entries, key=lambda e: e[0])],
    }

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
            "error": f"Notarization computed but failed to store: {e}",
            "root_hash": root_hex,
            "leaf_count": leaf_count,
            "receipts": len(receipts),
        }

    return {
        "success": True,
        "notarization_id": anchor_id,
        "root_hash": root_hex,
        "leaf_count": leaf_count,
        "calendars_submitted": len(receipts),
        "calendars_attempted": len(ots._calendars),
        "status": "submitted" if receipts else "no_receipts",
        "created_at": now.isoformat(),
        "message": (
            f"Notarized {leaf_count} ledger balances. "
            f"Merkle root: {root_hex[:16]}... "
            f"Submitted to {len(receipts)}/{len(ots._calendars)} OTS calendars. "
            f"Bitcoin confirmation expected in 1-6 hours."
        ),
    }


async def get_notarization_proof_tool(
    vault: Any,
    notarization_id: str,
    npub: str,
) -> dict[str, Any]:
    """Generate a Merkle inclusion proof for a patron's balance in a notarization.

    Args:
        vault: A NeonVault instance (must have fetch_anchor).
        notarization_id: The notarization record ID.
        npub: The patron's Nostr public key.

    Returns:
        Inclusion proof with verification guide, or error if not found.
    """
    try:
        anchor = await vault.fetch_anchor(notarization_id)
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch notarization: {e}"}

    if not anchor:
        return {"success": False, "error": f"Notarization {notarization_id} not found."}

    try:
        leaf_hashes = json.loads(anchor["leaf_hashes_json"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {"success": False, "error": f"Failed to parse leaf hashes: {e}"}

    tree = MerkleTree.from_leaf_hashes(leaf_hashes)

    if tree.root_hex != anchor["root_hash"]:
        return {
            "success": False,
            "error": "Integrity error: reconstructed root does not match stored root.",
        }

    proof = tree.get_proof(npub)
    if proof is None:
        return {
            "success": False,
            "error": f"npub {npub} not found in notarization {notarization_id}.",
        }

    if not proof.verify():
        return {
            "success": False,
            "error": "Internal error: generated proof fails verification.",
        }

    return {
        "success": True,
        "notarization_id": notarization_id,
        "root_hash": anchor["root_hash"],
        "status": anchor.get("status", "unknown"),
        "created_at": anchor.get("created_at"),
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


async def list_notarizations_tool(
    vault: Any,
    limit: int = 20,
    status: str | None = None,
) -> dict[str, Any]:
    """List recent notarization records.

    Args:
        vault: A NeonVault instance (must have list_anchors).
        limit: Maximum number of records to return (default 20).
        status: Optional filter by status (e.g., "submitted", "confirmed").

    Returns:
        List of notarization summaries.
    """
    try:
        anchors = await vault.list_anchors(limit=limit, status=status)
    except Exception as e:
        return {"success": False, "error": f"Failed to list notarizations: {e}"}

    items: list[dict[str, Any]] = []
    for a in anchors:
        item: dict[str, Any] = {
            "notarization_id": str(a["id"]),
            "root_hash": a["root_hash"],
            "leaf_count": a["leaf_count"],
            "status": a["status"],
            "created_at": a.get("created_at"),
        }
        if a.get("confirmed_at"):
            item["confirmed_at"] = a["confirmed_at"]
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
        "notarizations": items,
    }


# Backward compatibility aliases
anchor_ledger_tool = notarize_ledger_tool
get_anchor_proof_tool = get_notarization_proof_tool
list_anchors_tool = list_notarizations_tool
