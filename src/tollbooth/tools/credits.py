"""Credit management tools: purchase_credits, check_payment, check_balance, btcpay_status."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
from datetime import date, datetime, timedelta, timezone
from typing import Any

from tollbooth.btcpay_client import BTCPayClient, BTCPayAuthError, BTCPayError
from tollbooth.certificate import CertificateError, verify_certificate_auto
from tollbooth.config import TollboothConfig
from tollbooth.ledger import UserLedger
from tollbooth.ledger_cache import LedgerCache
from tollbooth.lnurl import LnurlResolutionError, resolve_lightning_address
from tollbooth.constants import LOW_BALANCE_FLOOR_API_SATS, MAX_INVOICE_SATS

logger = logging.getLogger(__name__)

# Default credit multiplier for users not in tier config
_DEFAULT_MULTIPLIER = 1

# Sanity ceiling for royalty payouts (independent of sats_to_btc_string ceiling).
# 2% of a 5M-sat purchase = 100,000 sats — anything above is suspect.
ROYALTY_PAYOUT_MAX_SATS = 100_000


async def _attempt_royalty_payout(
    btcpay: BTCPayClient,
    invoice_amount_sats: int,
    royalty_address: str,
    royalty_percent: float,
    royalty_min_sats: int,
) -> dict[str, Any] | None:
    """Attempt a royalty payout to the originator's Lightning Address.

    Returns a result dict on success or partial failure, None if below minimum.
    Never raises — catches all BTCPayError exceptions.
    """
    royalty_sats = int(invoice_amount_sats * royalty_percent)
    if royalty_sats < royalty_min_sats:
        return None

    if royalty_sats > ROYALTY_PAYOUT_MAX_SATS:
        logger.error(
            "Royalty payout %d sats exceeds ceiling of %d — refusing payout. "
            "Check royalty_percent (%.4f) or invoice amount (%d).",
            royalty_sats, ROYALTY_PAYOUT_MAX_SATS,
            royalty_percent, invoice_amount_sats,
        )
        return {
            "royalty_sats": royalty_sats,
            "royalty_address": royalty_address,
            "royalty_error": (
                f"Royalty amount ({royalty_sats:,} sats) exceeds safety ceiling "
                f"({ROYALTY_PAYOUT_MAX_SATS:,} sats). Payout refused."
            ),
        }

    # Resolve Lightning address to BOLT11 invoice (skip if already BOLT11)
    destination = royalty_address
    if not royalty_address.startswith("lnbc"):
        try:
            destination = await resolve_lightning_address(
                royalty_address, royalty_sats,
            )
            logger.info(
                "Resolved %s to BOLT11 invoice (%s…) for %d sats.",
                royalty_address, destination[:20], royalty_sats,
            )
        except LnurlResolutionError as e:
            logger.warning("LNURL resolution failed for %s: %s", royalty_address, e)
            return {
                "royalty_sats": royalty_sats,
                "royalty_address": royalty_address,
                "royalty_error": f"Lightning address resolution failed: {e}",
            }

    try:
        payout = await btcpay.create_payout(destination, royalty_sats)
        result = {
            "royalty_sats": royalty_sats,
            "royalty_address": royalty_address,
            "payout_destination": destination[:30] + "…" if len(destination) > 30 else destination,
            "payout_id": payout.get("id", ""),
            "payout_state": payout.get("state", "Unknown"),
        }
        if result["payout_state"] == "AwaitingApproval":
            result["payout_hint"] = (
                "Payout is awaiting approval. If payouts never settle, "
                "check Store Settings > Payout Processors > Automated Lightning Sender."
            )
        return result
    except BTCPayError as e:
        logger.warning("Royalty payout failed: %s", e)
        return {
            "royalty_sats": royalty_sats,
            "royalty_address": royalty_address,
            "royalty_error": str(e),
        }


def _get_tier_info(
    user_id: str,
    tier_config_json: str | None,
    user_tiers_json: str | None,
    default_credit_ttl_seconds: int | None = None,
) -> tuple[str, int, int | None]:
    """Look up tier name, credit multiplier, and credit TTL for a user.

    Returns (tier_name, multiplier, credit_ttl_seconds).
    """
    if not tier_config_json or not user_tiers_json:
        return "default", _DEFAULT_MULTIPLIER, default_credit_ttl_seconds

    try:
        tier_config = json.loads(tier_config_json)
        user_tiers = json.loads(user_tiers_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid tier config JSON; using default multiplier.")
        return "default", _DEFAULT_MULTIPLIER, default_credit_ttl_seconds

    tier_name = user_tiers.get(user_id, "default")
    tier = tier_config.get(tier_name, tier_config.get("default", {}))
    multiplier = int(tier.get("credit_multiplier", _DEFAULT_MULTIPLIER))

    # TTL: per-tier override, then config default, then None (never)
    ttl_raw = tier.get("credit_ttl_seconds")
    if ttl_raw is not None:
        ttl = int(ttl_raw)
    else:
        ttl = default_credit_ttl_seconds
    return tier_name, multiplier, ttl


def _get_multiplier(
    user_id: str,
    tier_config_json: str | None,
    user_tiers_json: str | None,
) -> int:
    """Look up credit multiplier for a user based on tier config."""
    _, multiplier, _ = _get_tier_info(user_id, tier_config_json, user_tiers_json)
    return multiplier


async def _create_purchase_invoice(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    amount_sats: int,
    tier_config_json: str | None,
    user_tiers_json: str | None,
    extra_metadata: dict[str, Any] | None = None,
    default_credit_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Shared logic: validate amount, create BTCPay invoice, record in ledger.

    Both certified (operator) and direct (Authority) purchases funnel here.
    """
    if amount_sats <= 0:
        return {"success": False, "error": "amount_sats must be positive."}

    if amount_sats > MAX_INVOICE_SATS:
        return {
            "success": False,
            "error": f"amount_sats exceeds maximum of {MAX_INVOICE_SATS:,} sats (0.01 BTC) per invoice.",
        }

    invoice_metadata: dict[str, Any] = {
        "user_id": user_id,
        "purpose": "credit_purchase",
    }
    if extra_metadata:
        invoice_metadata.update(extra_metadata)

    try:
        invoice = await btcpay.create_invoice(
            amount_sats,
            metadata=invoice_metadata,
        )
    except BTCPayError as e:
        return {"success": False, "error": f"BTCPay error: {e}"}

    invoice_id = invoice.get("id", "")
    checkout_link = invoice.get("checkoutLink", "")
    expiry = invoice.get("expirationTime", "")

    tier_name, multiplier, ttl = _get_tier_info(
        user_id, tier_config_json, user_tiers_json, default_credit_ttl_seconds,
    )
    expected_credits = amount_sats * multiplier

    # Record pending invoice — flush immediately so the invoice survives cache loss
    ledger = await cache.get(user_id)
    ledger.pending_invoices.append(invoice_id)
    ledger.record_invoice_created(
        invoice_id=invoice_id,
        amount_sats=amount_sats,
        multiplier=multiplier,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    cache.mark_dirty(user_id)
    if not await cache.flush_user(user_id):
        logger.warning("Failed to flush pending invoice %s for %s.", invoice_id, user_id)

    result: dict[str, Any] = {
        "success": True,
        "invoice_id": invoice_id,
        "amount_sats": amount_sats,
        "checkout_link": checkout_link,
        "expiration": expiry,
        "tier": tier_name,
        "multiplier": multiplier,
        "expected_credits": expected_credits,
        "message": (
            f"Invoice created for {amount_sats:,} sats.\n\n"
            f"Pay here: {checkout_link}\n"
            f"Expires: {expiry}\n"
            f"Tier: {tier_name} ({multiplier}x) — "
            f"you will receive {expected_credits:,} credits on settlement.\n\n"
            f'After paying, call check_payment with invoice_id: "{invoice_id}"'
        ),
    }
    if ttl is not None:
        result["credit_ttl_seconds"] = ttl
    return result


async def purchase_credits_tool(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    amount_sats: int,
    certificate: str,
    authority_npub: str,
    tier_config_json: str | None = None,
    user_tiers_json: str | None = None,
    default_credit_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Create a BTCPay invoice after verifying an Authority certificate.

    For OPERATOR use: the certified purchase flow. Every credit purchase
    requires a valid Authority-signed Nostr event certificate.
    The certificate's net_sats (amount after tax) determines the invoice amount.
    """
    # Trust gate — authority_npub must be configured
    if not authority_npub:
        return {
            "success": False,
            "error": "Operator misconfigured: authority_npub is not set. "
            "A Tollbooth Operator cannot operate without a trusted Authority.",
        }

    if not certificate:
        return {
            "success": False,
            "error": "A valid Authority certificate is required for every credit purchase. "
            "Call the Authority's certify_credits tool first.",
        }

    try:
        cert_claims = verify_certificate_auto(
            certificate, authority_npub=authority_npub,
        )
    except CertificateError as e:
        return {"success": False, "error": f"Certificate rejected: {e}"}

    # Use net_sats from the certificate as the invoice amount
    net_sats = cert_claims["net_sats"]

    result = await _create_purchase_invoice(
        btcpay, cache, user_id, net_sats,
        tier_config_json, user_tiers_json,
        extra_metadata={"certificate_jti": cert_claims["jti"]},
        default_credit_ttl_seconds=default_credit_ttl_seconds,
    )
    if result.get("success"):
        result["certificate_jti"] = cert_claims["jti"]
    return result


async def purchase_tax_credits_tool(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    amount_sats: int,
    tier_config_json: str | None = None,
    user_tiers_json: str | None = None,
    default_credit_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Create a BTCPay invoice for a direct tax credit purchase.

    For AUTHORITY use: operators buy tax credits directly from the Authority.
    No certificate needed — the Authority IS the trust anchor.
    """
    return await _create_purchase_invoice(
        btcpay, cache, user_id, amount_sats,
        tier_config_json, user_tiers_json,
        extra_metadata={"purpose": "tax_credit_purchase"},
        default_credit_ttl_seconds=default_credit_ttl_seconds,
    )


async def check_payment_tool(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    invoice_id: str,
    tier_config_json: str | None = None,
    user_tiers_json: str | None = None,
    default_credit_ttl_seconds: int | None = None,
    royalty_address: str | None = None,
    royalty_percent: float = 0.02,
    royalty_min_sats: int = 10,
) -> dict[str, Any]:
    """Poll a BTCPay invoice and credit the user's balance on settlement.

    Call this after the user pays an invoice from purchase_credits_tool. Safe
    to call multiple times — credits are only granted once per invoice
    (idempotent via credited_invoices). On settlement, also fires a royalty
    payout to the Tollbooth originator if royalty_address is configured.
    """
    try:
        invoice = await btcpay.get_invoice(invoice_id)
    except BTCPayError as e:
        return {"success": False, "error": f"BTCPay error: {e}"}

    status = invoice.get("status", "Unknown")
    additional = invoice.get("additionalStatus", "")
    ledger = await cache.get(user_id)

    result: dict[str, Any] = {
        "success": True,
        "invoice_id": invoice_id,
        "status": status,
    }
    if additional:
        result["additional_status"] = additional

    if status == "New":
        result["message"] = "Invoice created, awaiting payment."

    elif status == "Processing":
        result["message"] = "Payment seen, waiting for confirmation."

    elif status == "Settled":
        if invoice_id in ledger.credited_invoices:
            # Already credited — true idempotency check
            result["message"] = "Payment already credited."
            result["credits_granted"] = 0
        else:
            # Credit the user — flush immediately so credits survive cache loss
            amount_str = invoice.get("amount", "0")
            amount_sats = int(float(amount_str))
            _, multiplier, ttl = _get_tier_info(
                user_id, tier_config_json, user_tiers_json, default_credit_ttl_seconds,
            )
            credited = amount_sats * multiplier
            ledger.credit_deposit(credited, invoice_id, ttl_seconds=ttl)
            ledger.record_invoice_settled(
                invoice_id=invoice_id,
                api_sats_credited=credited,
                settled_at=datetime.now(timezone.utc).isoformat(),
                btcpay_status=status,
            )
            cache.mark_dirty(user_id)
            if not await cache.flush_user(user_id):
                logger.error(
                    "CRITICAL: Failed to flush %d credits for %s (invoice %s). "
                    "Credits are in memory but may be lost on restart.",
                    credited, user_id, invoice_id,
                )
            result["credits_granted"] = credited
            result["multiplier"] = multiplier
            result["message"] = f"Payment settled! {credited:,} credits added to your balance."

            # Attempt royalty payout (never blocks credit settlement)
            if royalty_address:
                royalty_result = await _attempt_royalty_payout(
                    btcpay, amount_sats, royalty_address,
                    royalty_percent, royalty_min_sats,
                )
                if royalty_result is not None:
                    result["royalty_payout"] = royalty_result

    elif status == "Expired":
        if invoice_id in ledger.pending_invoices:
            ledger.pending_invoices.remove(invoice_id)
        ledger.record_invoice_terminal(invoice_id, "Expired", status)
        cache.mark_dirty(user_id)
        await cache.flush_user(user_id)
        result["message"] = "Invoice expired. Create a new one with purchase_credits."

    elif status == "Invalid":
        if invoice_id in ledger.pending_invoices:
            ledger.pending_invoices.remove(invoice_id)
        ledger.record_invoice_terminal(invoice_id, "Invalid", status)
        cache.mark_dirty(user_id)
        await cache.flush_user(user_id)
        result["message"] = "Payment invalid."

    else:
        result["message"] = f"Unknown invoice status: {status}"

    result["balance_api_sats"] = ledger.balance_api_sats
    return result


async def check_balance_tool(
    cache: LedgerCache,
    user_id: str,
    tier_config_json: str | None = None,
    user_tiers_json: str | None = None,
    default_credit_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Return the user's current credit balance, tier info, and usage summary."""
    ledger = await cache.get(user_id)
    today = date.today().isoformat()

    tier_name, multiplier, _ = _get_tier_info(
        user_id, tier_config_json, user_tiers_json, default_credit_ttl_seconds,
    )

    result: dict[str, Any] = {
        "success": True,
        "tier": tier_name,
        "multiplier": multiplier,
        "balance_api_sats": ledger.balance_api_sats,
        "total_deposited_api_sats": ledger.total_deposited_api_sats,
        "total_consumed_api_sats": ledger.total_consumed_api_sats,
        "pending_invoices": len(ledger.pending_invoices),
        "last_deposit_at": ledger.last_deposit_at,
    }

    if "seed_balance_v1" in ledger.credited_invoices:
        result["seed_balance_granted"] = True

    # Tranche expiration info
    result["total_expired_api_sats"] = ledger.total_expired_api_sats
    expiring_24h = ledger.expiring_within(86400)
    if expiring_24h > 0:
        result["expiring_within_24h_sats"] = expiring_24h
    next_exp = ledger.next_expiration()
    if next_exp:
        result["next_expiration_iso"] = next_exp
    result["active_tranches"] = len([
        t for t in ledger.tranches if t.remaining_sats > 0
    ])

    # Include today's usage if available
    today_log = ledger.daily_log.get(today)
    if today_log:
        result["today_usage"] = {
            tool: {"calls": u.calls, "api_sats": u.api_sats}
            for tool, u in today_log.items()
        }

    # Invoice history summary
    if ledger.invoices:
        settled = [r for r in ledger.invoices.values() if r.status == "Settled"]
        pending = [r for r in ledger.invoices.values() if r.status == "Pending"]
        result["invoice_summary"] = {
            "total_invoices": len(ledger.invoices),
            "settled_count": len(settled),
            "pending_count": len(pending),
            "total_real_sats": sum(r.amount_sats for r in settled),
            "total_api_sats_credited": sum(r.api_sats_credited for r in settled),
        }

    return result


async def restore_credits_tool(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    invoice_id: str,
    tier_config_json: str | None = None,
    user_tiers_json: str | None = None,
    default_credit_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Restore credits from a paid invoice that was lost due to cache or vault issues."""
    # Check idempotency first
    ledger = await cache.get(user_id)
    if invoice_id in ledger.credited_invoices:
        return {
            "success": True,
            "invoice_id": invoice_id,
            "credits_granted": 0,
            "balance_api_sats": ledger.balance_api_sats,
            "message": "Invoice already credited — no duplicate credits applied.",
        }

    # Vault-first: check if we have a settled invoice record in the ledger
    vault_record = ledger.invoices.get(invoice_id)
    if vault_record and vault_record.status == "Settled" and vault_record.api_sats_credited > 0:
        # Restore from vault record — no BTCPay call needed
        credited = vault_record.api_sats_credited
        _, _, ttl = _get_tier_info(
            user_id, tier_config_json, user_tiers_json, default_credit_ttl_seconds,
        )
        ledger.credit_deposit(credited, invoice_id, ttl_seconds=ttl)
        cache.mark_dirty(user_id)
        if not await cache.flush_user(user_id):
            logger.error(
                "CRITICAL: Failed to flush vault-restored %d credits for %s (invoice %s).",
                credited, user_id, invoice_id,
            )
        return {
            "success": True,
            "invoice_id": invoice_id,
            "source": "vault_record",
            "amount_sats": vault_record.amount_sats,
            "multiplier": vault_record.multiplier,
            "credits_granted": credited,
            "balance_api_sats": ledger.balance_api_sats,
            "message": f"Restored {credited:,} credits from vault invoice record.",
        }

    # Fall back to BTCPay verification
    try:
        invoice = await btcpay.get_invoice(invoice_id)
    except BTCPayError as e:
        return {"success": False, "error": f"BTCPay error: {e}"}

    status = invoice.get("status", "Unknown")
    if status != "Settled":
        return {
            "success": False,
            "error": f"Invoice status is '{status}', not 'Settled'. Cannot restore.",
            "invoice_id": invoice_id,
        }

    # Credit the balance
    amount_str = invoice.get("amount", "0")
    amount_sats = int(float(amount_str))
    _, multiplier, ttl = _get_tier_info(
        user_id, tier_config_json, user_tiers_json, default_credit_ttl_seconds,
    )
    credited = amount_sats * multiplier

    ledger.credit_deposit(credited, invoice_id, ttl_seconds=ttl)
    ledger.record_invoice_settled(
        invoice_id=invoice_id,
        api_sats_credited=credited,
        settled_at=datetime.now(timezone.utc).isoformat(),
        btcpay_status=status,
    )
    cache.mark_dirty(user_id)
    if not await cache.flush_user(user_id):
        logger.error(
            "CRITICAL: Failed to flush restored %d credits for %s (invoice %s).",
            credited, user_id, invoice_id,
        )

    return {
        "success": True,
        "invoice_id": invoice_id,
        "source": "btcpay",
        "amount_sats": amount_sats,
        "multiplier": multiplier,
        "credits_granted": credited,
        "balance_api_sats": ledger.balance_api_sats,
        "message": f"Restored {credited:,} credits from invoice {invoice_id}.",
    }


async def reconcile_pending_invoices(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    tier_config_json: str | None = None,
    user_tiers_json: str | None = None,
    default_credit_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Reconcile pending invoices on startup: credit settled, remove terminal."""
    ledger = await cache.get(user_id)
    pending_copy = list(ledger.pending_invoices)
    if not pending_copy:
        return {"reconciled": 0, "actions": []}

    actions: list[dict[str, Any]] = []
    changed = False

    for invoice_id in pending_copy:
        try:
            invoice = await btcpay.get_invoice(invoice_id)
        except BTCPayError:
            logger.warning("Reconciliation: skipping %s (BTCPay error).", invoice_id)
            continue

        status = invoice.get("status", "Unknown")

        if status == "Settled" and invoice_id not in ledger.credited_invoices:
            amount_str = invoice.get("amount", "0")
            amount_sats = int(float(amount_str))
            _, multiplier, ttl = _get_tier_info(
                user_id, tier_config_json, user_tiers_json, default_credit_ttl_seconds,
            )
            credited = amount_sats * multiplier
            ledger.credit_deposit(credited, invoice_id, ttl_seconds=ttl)
            ledger.record_invoice_settled(
                invoice_id=invoice_id,
                api_sats_credited=credited,
                settled_at=datetime.now(timezone.utc).isoformat(),
                btcpay_status=status,
            )
            changed = True
            actions.append({
                "invoice_id": invoice_id,
                "action": "credited",
                "api_sats": credited,
            })

        elif status in ("Expired", "Invalid"):
            if invoice_id in ledger.pending_invoices:
                ledger.pending_invoices.remove(invoice_id)
            ledger.record_invoice_terminal(invoice_id, status, status)
            changed = True
            actions.append({
                "invoice_id": invoice_id,
                "action": "removed",
                "reason": status,
            })

    if changed:
        cache.mark_dirty(user_id)
        await cache.flush_user(user_id)

    return {"reconciled": len(actions), "actions": actions}


def compute_low_balance_warning(
    ledger: UserLedger,
    seed_balance_sats: int,
    low_balance_floor: int = LOW_BALANCE_FLOOR_API_SATS,
) -> dict[str, Any] | None:
    """Compute a low-balance warning dict if balance is running low.

    Returns None if balance is healthy (>= threshold).
    """
    # Find reference amount from last settled invoice
    settled = [r for r in ledger.invoices.values() if r.status == "Settled"]
    if settled:
        last = settled[-1]
        reference = last.api_sats_credited
    elif seed_balance_sats > 0 and "seed_balance_v1" in ledger.credited_invoices:
        reference = seed_balance_sats
    else:
        reference = low_balance_floor

    threshold = max(reference // 5, low_balance_floor)

    if ledger.balance_api_sats >= threshold:
        return None

    # Suggested top-up: last invoice's real amount_sats, capped
    if settled:
        suggested = settled[-1].amount_sats
        if suggested <= 0:
            suggested = 1000
    else:
        suggested = 1000
    suggested = min(suggested, MAX_INVOICE_SATS)

    return {
        "balance_api_sats": ledger.balance_api_sats,
        "threshold_api_sats": threshold,
        "suggested_top_up_sats": suggested,
        "purchase_command": f'Use purchase_credits with amount_sats={suggested}',
        "message": (
            f"Low balance: {ledger.balance_api_sats} api_sats remaining "
            f"(warning threshold: {threshold}). "
            f"Consider topping up with purchase_credits."
        ),
    }


async def account_statement_tool(
    cache: LedgerCache,
    user_id: str,
    days: int = 30,
) -> dict[str, Any]:
    """Generate a customer-facing account statement with purchase history and usage.

    Returns a structured statement suitable for customer proof-of-purchase
    and usage auditing. Includes: account summary, invoice line items, active
    credit tranches, all-time tool usage, and recent daily usage.

    Args:
        cache: The LedgerCache instance.
        user_id: The user's identity key.
        days: Number of days of daily usage to include (default 30).
    """
    ledger = await cache.get(user_id)
    now = datetime.now(timezone.utc)
    today = date.today()

    # -- Account summary ---------------------------------------------------
    summary: dict[str, Any] = {
        "balance_api_sats": ledger.balance_api_sats,
        "total_deposited_api_sats": ledger.total_deposited_api_sats,
        "total_consumed_api_sats": ledger.total_consumed_api_sats,
        "total_expired_api_sats": ledger.total_expired_api_sats,
    }

    # -- Invoice line items (sorted by created_at, most recent first) ------
    invoice_items: list[dict[str, Any]] = []
    for rec in sorted(
        ledger.invoices.values(),
        key=lambda r: r.created_at or "",
        reverse=True,
    ):
        item: dict[str, Any] = {
            "invoice_id": rec.invoice_id,
            "status": rec.status,
            "amount_sats": rec.amount_sats,
            "api_sats_credited": rec.api_sats_credited,
            "multiplier": rec.multiplier,
            "created_at": rec.created_at,
        }
        if rec.settled_at:
            item["settled_at"] = rec.settled_at
        invoice_items.append(item)

    # -- Active tranches ---------------------------------------------------
    tranche_items: list[dict[str, Any]] = []
    for t in ledger.tranches:
        if t.remaining_sats <= 0 or t.is_expired_at(now):
            continue
        entry: dict[str, Any] = {
            "granted_at": t.granted_at,
            "original_sats": t.original_sats,
            "remaining_sats": t.remaining_sats,
            "invoice_id": t.invoice_id,
        }
        if t.expires_at:
            entry["expires_at"] = t.expires_at
        tranche_items.append(entry)

    # -- All-time tool usage (sorted by api_sats descending) ---------------
    tool_usage_items: list[dict[str, Any]] = []
    for tool_name, usage in sorted(
        ledger.history.items(),
        key=lambda kv: kv[1].api_sats,
        reverse=True,
    ):
        tool_usage_items.append({
            "tool": tool_name,
            "calls": usage.calls,
            "api_sats": usage.api_sats,
        })

    # -- Daily usage log (last N days, most recent first) ------------------
    cutoff_date = (today - timedelta(days=days)).isoformat()
    daily_items: list[dict[str, Any]] = []
    for day_iso in sorted(ledger.daily_log.keys(), reverse=True):
        if day_iso < cutoff_date:
            break
        day_tools = ledger.daily_log[day_iso]
        day_total_calls = sum(u.calls for u in day_tools.values())
        day_total_sats = sum(u.api_sats for u in day_tools.values())
        daily_items.append({
            "date": day_iso,
            "total_calls": day_total_calls,
            "total_api_sats": day_total_sats,
            "tools": {
                name: {"calls": u.calls, "api_sats": u.api_sats}
                for name, u in sorted(
                    day_tools.items(),
                    key=lambda kv: kv[1].api_sats,
                    reverse=True,
                )
            },
        })

    return {
        "success": True,
        "generated_at": now.isoformat(),
        "statement_period_days": days,
        "account_summary": summary,
        "purchase_history": invoice_items,
        "active_tranches": tranche_items,
        "tool_usage_all_time": tool_usage_items,
        "daily_usage": daily_items,
    }


async def btcpay_status_tool(
    config: TollboothConfig,
    btcpay: BTCPayClient | None,
) -> dict[str, Any]:
    """Report BTCPay configuration state, connectivity, and permissions for diagnostics."""
    result: dict[str, Any] = {
        "btcpay_host": config.btcpay_host or None,
        "btcpay_store_id": config.btcpay_store_id or None,
        "btcpay_api_key_status": "present" if config.btcpay_api_key else "missing",
    }

    # Runtime version provenance — what's actually imported in this process
    versions: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("tollbooth-dpyc", "fastmcp"):
        try:
            versions[pkg.replace("-", "_")] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg.replace("-", "_")] = "unknown"
    result["versions"] = versions

    # Tier config
    if config.btcpay_tier_config:
        try:
            tiers = json.loads(config.btcpay_tier_config)
            result["tier_config"] = f"{len(tiers)} tier(s)"
        except (json.JSONDecodeError, TypeError):
            result["tier_config"] = "invalid JSON"
    else:
        result["tier_config"] = "missing"

    # User tiers
    if config.btcpay_user_tiers:
        try:
            users = json.loads(config.btcpay_user_tiers)
            result["user_tiers"] = f"{len(users)} user(s)"
        except (json.JSONDecodeError, TypeError):
            result["user_tiers"] = "invalid JSON"
    else:
        result["user_tiers"] = "missing"

    # Credit TTL
    result["credit_ttl_seconds"] = config.credit_ttl_seconds

    # Authority trust chain config
    authority_config: dict[str, Any] = {
        "npub_configured": bool(config.authority_npub),
        "certificate_verification_enabled": bool(config.authority_npub),
    }
    if config.authority_npub:
        authority_config["authority_npub"] = config.authority_npub
    result["authority_config"] = authority_config

    # Connectivity checks — only if all 3 connection vars present and client available
    connection_vars_present = bool(
        config.btcpay_host and config.btcpay_store_id and config.btcpay_api_key
    )

    # Royalty config
    royalty_enabled = bool(config.tollbooth_royalty_address)
    result["royalty_config"] = {
        "enabled": royalty_enabled,
        "address": config.tollbooth_royalty_address,
        "percent": config.tollbooth_royalty_percent,
        "min_sats": config.tollbooth_royalty_min_sats,
    }

    if connection_vars_present and btcpay is not None:
        # Health check
        try:
            await btcpay.health_check()
            result["server_reachable"] = True
        except BTCPayError:
            result["server_reachable"] = False
        except Exception:
            result["server_reachable"] = False

        # Store check
        try:
            store = await btcpay.get_store()
            result["store_name"] = store.get("name", "unknown")
        except BTCPayAuthError:
            result["store_name"] = "unauthorized"
        except BTCPayError:
            result["store_name"] = None
        except Exception:
            result["store_name"] = None

        # API key permissions check
        try:
            key_info = await btcpay.get_api_key_info()
            permissions = key_info.get("permissions", [])
            required = ["btcpay.store.cancreateinvoice", "btcpay.store.canviewinvoices"]
            if royalty_enabled:
                required.append("btcpay.store.cancreatenonapprovedpullpayments")
                required.append("btcpay.store.canviewstoresettings")
            present = [p for p in required if p in permissions]
            missing = [p for p in required if p not in permissions]
            result["api_key_permissions"] = {
                "permissions": permissions,
                "required": required,
                "present": present,
                "missing": missing,
            }
        except BTCPayError as e:
            result["api_key_permissions"] = {"error": str(e)}
        except Exception as e:
            result["api_key_permissions"] = {"error": str(e)}

        # Payout processor check (only if royalties enabled)
        if royalty_enabled:
            try:
                processors = await btcpay.get_payout_processors()
                lightning_processor = any(
                    "Lightning" in p.get("name", "") or "Lightning" in p.get("friendlyName", "")
                    for p in processors
                )
                result["payout_processor"] = {
                    "configured_count": len(processors),
                    "lightning_automated": lightning_processor,
                }
                if not lightning_processor:
                    result["payout_processor"]["warning"] = (
                        "No Lightning Payout Processor configured. "
                        "Royalty payouts will be created but never settle automatically. "
                        "Go to: Store Settings > Payout Processors > Automated Lightning Sender"
                    )
            except BTCPayError as e:
                result["payout_processor"] = {"error": str(e)}
            except Exception as e:
                result["payout_processor"] = {"error": str(e)}
    else:
        result["server_reachable"] = None
        result["store_name"] = None

    return result
