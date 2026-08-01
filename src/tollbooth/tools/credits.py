"""Credit management tools: purchase_credits, check_payment, check_balance, btcpay_status."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from tollbooth.btcpay_client import BTCPayAuthError, BTCPayClient, BTCPayError
from tollbooth.certificate import CertificateError, verify_certificate_auto
from tollbooth.config import TollboothConfig
from tollbooth.constants import LOW_BALANCE_FLOOR_API_SATS, MAX_INVOICE_SATS
from tollbooth.ledger import UserLedger
from tollbooth.ledger_cache import LedgerCache
from tollbooth.vault_backend import LedgerUnavailableError, LedgerWriteError


def _invoice_owner(invoice: dict[str, Any]) -> str | None:
    """The npub the invoice was created for, per its BTCPay metadata.

    A settled ``invoice_id`` is NOT a bearer token: crediting must confirm the
    invoice belongs to the account being credited, or a party who merely learns
    another patron's invoice_id (they surface in tool results, DMs, logs) could
    claim it. Every DPYC invoice is stamped with ``metadata.user_id`` at
    creation (see ``_create_purchase_invoice``); ``None`` means the field is
    absent (anomalous — allowed through so a metadata-less legacy invoice isn't
    bricked, but the mismatch case below is always refused).
    """
    meta = invoice.get("metadata")
    if isinstance(meta, dict):
        owner = meta.get("user_id")
        return owner if isinstance(owner, str) else None
    return None

logger = logging.getLogger(__name__)


async def _create_purchase_invoice(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    amount_sats: int,
    extra_metadata: dict[str, Any] | None = None,
    tranche_lifetime_seconds: int | None = None,
    invoice_dm_callback: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Shared logic: validate amount, create BTCPay invoice, record in ledger.

    Both certified (operator) and direct (Authority) purchases funnel here.
    1 sat = 1 api_sat (no multiplier).
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

    # Populate BTCPay's `orderId` so the store's Lightning Description
    # Template — typically "{StoreName} (Order ID: {OrderId})" — substitutes
    # a meaningful string into the BOLT11 description. Without this,
    # wallets like Wallet of Satoshi show "Paid to <StoreName> (Order ID:)"
    # with an empty Order ID field, which looks like a bug to the paying
    # user. The id is constructed from the purchase purpose, a 16-char slug
    # of the user identity, and a unix timestamp — enough to disambiguate
    # purchases in the patron's wallet history without exposing the full
    # npub.
    if "orderId" not in invoice_metadata:
        purpose_slug = str(invoice_metadata.get("purpose", "purchase"))
        user_slug = str(user_id)[:16]
        ts = int(datetime.now(UTC).timestamp())
        invoice_metadata["orderId"] = f"dpyc-{purpose_slug}-{user_slug}-{ts}"

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

    # Record the pending invoice write-through, so it survives cache loss AND a
    # concurrent writer. The invoice already exists at BTCPay by this point, so a
    # persistence failure here is never a reason to fail the call — but it does
    # mean the patron has an invoice with no ledger record, and recovery is a
    # `restore_credits` they may not know to call. Log loudly either way.
    def _record_pending(led: UserLedger) -> None:
        led.pending_invoices.append(invoice_id)
        led.record_invoice_created(
            invoice_id=invoice_id,
            amount_sats=amount_sats,
            multiplier=1,
            created_at=datetime.now(UTC).isoformat(),
        )

    try:
        await cache.mutate(user_id, _record_pending)
    except LedgerUnavailableError:
        logger.error(
            "Vault unavailable during purchase_credits — pending invoice %s "
            "NOT persisted to ledger for %s. Patron will need restore_credits "
            "to recover credits after paying.",
            invoice_id, user_id,
        )
    except LedgerWriteError:
        logger.error(
            "Lost every CAS race writing pending invoice %s for %s — the invoice "
            "exists at BTCPay with no ledger record. Patron will need "
            "restore_credits to recover credits after paying.",
            invoice_id, user_id,
        )

    # Fetch BOLT11 Lightning invoice string for direct wallet payment
    bolt11: str | None = None
    if invoice_id:
        try:
            bolt11 = await btcpay.get_lightning_invoice(invoice_id)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to fetch BOLT11 for invoice %s", invoice_id)

    result: dict[str, Any] = {
        "success": True,
        "invoice_id": invoice_id,
        "amount_sats": amount_sats,
        "expected_credits": amount_sats,
        "checkout_link": checkout_link,
        "expiration": expiry,
        "message": (
            f"Invoice created for {amount_sats:,} sats.\n\n"
            f"Pay here: {checkout_link}\n"
            f"Expires: {expiry}\n"
            f"You will receive {amount_sats:,} credits on settlement.\n\n"
            f'After paying, call check_payment with invoice_id: "{invoice_id}"'
        ),
    }
    if bolt11:
        result["lightning_invoice"] = bolt11
    if tranche_lifetime_seconds is not None:
        result["tranche_lifetime_seconds"] = tranche_lifetime_seconds

    # Fire-and-forget invoice DM — failure never blocks the purchase
    if invoice_dm_callback is not None:
        dm_text = (
            f"Lightning Invoice -- {amount_sats:,} sats\n\n"
            f"Pay here: {checkout_link}\n\n"
            f"Invoice: {invoice_id}\n"
            f"Expires: {expiry}\n\n"
            f"After paying, your credits appear automatically."
        )
        try:
            await invoice_dm_callback(dm_text)
            result["invoice_dm_sent"] = True
        except Exception:  # noqa: BLE001
            logger.warning("Invoice DM delivery failed for %s.", user_id)
            result["invoice_dm_sent"] = False

    return result


async def purchase_credits_tool(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    amount_sats: int,
    certificate: str,
    authority_npub: str,
    tranche_lifetime_seconds: int | None = None,
    ban_check_oracle_url: str | None = None,
    invoice_dm_callback: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Create a BTCPay invoice after verifying an Authority certificate.

    For OPERATOR use: the certified purchase flow. Every credit purchase
    requires a valid Authority-signed Nostr event certificate.
    The invoice is for the full amount_sats the patron requested — the
    certification fee is the operator's cost of doing business, not a
    patron-visible deduction.

    If *ban_check_oracle_url* is provided, checks the Oracle for ban status
    before proceeding.  Fail-closed: if the Oracle is unreachable, the
    purchase is denied — never grant free access.
    """
    # Ban check (fail-closed — Oracle unreachable = deny purchase)
    if ban_check_oracle_url:
        try:
            from tollbooth.oracle_client import OracleClient

            oracle = OracleClient(ban_check_oracle_url)
            ban_result = await oracle.check_ban_status(user_id)
            if ban_result.get("banned"):
                reason = ban_result.get("reason", "banned")
                return {
                    "success": False,
                    "error": f"Account suspended: {reason}. "
                    "Credit purchases are not available for banned members.",
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Ban check failed for %s — denying purchase (fail-closed): %s",
                user_id,
                exc,
            )
            return {
                "success": False,
                "error": "Unable to verify account standing — purchase denied. "
                "The Oracle may be temporarily unavailable. Try again later.",
            }

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

    # Invoice the patron for the full amount they requested — the
    # certification fee is absorbed by the operator, not the patron.
    invoice_sats = cert_claims["amount_sats"]

    result = await _create_purchase_invoice(
        btcpay, cache, user_id, invoice_sats,
        extra_metadata={"certificate_jti": cert_claims["jti"]},
        tranche_lifetime_seconds=tranche_lifetime_seconds,
        invoice_dm_callback=invoice_dm_callback,
    )
    if result.get("success"):
        result["certificate_jti"] = cert_claims["jti"]
    return result


async def direct_purchase_tool(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    amount_sats: int,
    tranche_lifetime_seconds: int | None = None,
) -> dict[str, Any]:
    """Create a BTCPay invoice for a direct credit purchase (no certificate).

    For AUTHORITY use: the Authority is the trust anchor, so it purchases
    credits directly without an upstream certificate.
    """
    return await _create_purchase_invoice(
        btcpay, cache, user_id, amount_sats,
        extra_metadata={"purpose": "direct_credit_purchase"},
        tranche_lifetime_seconds=tranche_lifetime_seconds,
    )


async def check_payment_tool(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    invoice_id: str,
    tranche_lifetime_seconds: int | None = None,
) -> dict[str, Any]:
    """Poll a BTCPay invoice and credit the user's balance on settlement.

    Call this after the user pays an invoice from purchase_credits_tool. Safe
    to call multiple times — credits are only granted once per invoice
    (idempotent via credited_invoices).
    """
    try:
        invoice = await btcpay.get_invoice(invoice_id)
    except BTCPayError as e:
        return {"success": False, "error": f"BTCPay error: {e}"}

    status = invoice.get("status", "Unknown")
    additional = invoice.get("additionalStatus", "")
    ledger = await cache.get_fresh(user_id)

    result: dict[str, Any] = {
        "success": True,
        "invoice_id": invoice_id,
        "status": status,
    }
    if additional:
        result["additional_status"] = additional

    async def _retire_invoice(terminal: str) -> None:
        """Drop a dead invoice from the pending list and record how it ended.

        Write-through like every other ledger mutation: the pending list is
        shared state, so retiring an invoice against a stale in-memory copy
        would resurrect invoices another replica had already retired.
        """
        def _apply(led: UserLedger) -> None:
            if invoice_id in led.pending_invoices:
                led.pending_invoices.remove(invoice_id)
            led.record_invoice_terminal(invoice_id, terminal, status)

        try:
            await cache.mutate(user_id, _apply)
        except (LedgerUnavailableError, LedgerWriteError) as exc:
            # Bookkeeping, not money: the invoice is dead at BTCPay either way,
            # and a later check_payment retires it again idempotently.
            logger.warning(
                "Could not retire %s invoice %s for %s: %s: %s",
                terminal, invoice_id, user_id, type(exc).__name__, exc,
            )

    if status == "New":
        result["message"] = "Invoice created, awaiting payment."

    elif status == "Processing":
        result["message"] = "Payment seen, waiting for confirmation."

    elif status == "Settled":
        # A settled invoice_id must belong to the caller being credited.
        # Without this, a proven patron who learns another patron's invoice_id
        # could claim it — the per-ledger credited_invoices idempotency guards
        # only the victim's ledger, not the claimer's, so it would mint free
        # credits (cross-account double-issuance).
        owner = _invoice_owner(invoice)
        if owner is not None and owner != user_id:
            logger.warning(
                "check_payment: invoice %s belongs to %s, not caller %s — refusing.",
                invoice_id, owner[:20], user_id[:20],
            )
            result["success"] = False
            result["credits_granted"] = 0
            result["error_code"] = "invoice_owner_mismatch"
            result["error"] = "This invoice was not created for your account."
            result["message"] = result["error"]
            return result

        amount_str = invoice.get("amount", "0")
        amount_sats = int(float(amount_str))
        settled_at = datetime.now(UTC).isoformat()

        def _apply_settlement(led: UserLedger) -> int:
            # Runs against FRESH ledger state inside the CAS write-through, and
            # is re-applied on each conflict retry — so the idempotency check
            # sees the definitive credited_invoices set, not a stale cache.
            if invoice_id in led.credited_invoices:
                return 0
            led.credit_deposit(amount_sats, invoice_id, ttl_seconds=tranche_lifetime_seconds)
            led.record_invoice_settled(
                invoice_id=invoice_id,
                api_sats_credited=amount_sats,
                settled_at=settled_at,
                btcpay_status=status,
            )
            return amount_sats

        try:
            credited = await cache.mutate(user_id, _apply_settlement)
        except (LedgerUnavailableError, LedgerWriteError) as exc:
            # Cold vault or exhausted CAS races — the credit did NOT persist.
            # The payment is safe at BTCPay; the caller retries once warm.
            # mutate() refuses to operate on an uncached/empty ledger, so this
            # replaces the old _vault_unavailable phantom-credit guard.
            logger.error(
                "Settle for %s (invoice %s) not persisted: %s: %s",
                user_id, invoice_id, type(exc).__name__, exc,
            )
            result["success"] = False
            result["credits_granted"] = 0
            result["persisted"] = False
            result["error_code"] = "vault_unavailable"
            result["error"] = (
                "Vault wasn't reachable during settle — credits NOT persisted. "
                "Your payment is safely settled at BTCPay; retry check_payment "
                "in 10–15 seconds to credit your balance."
            )
            result["message"] = result["error"]
            return result

        # mutate() wrote through to the vault; refresh the local ledger so the
        # trailing balance line reflects the new tranche.
        ledger = await cache.get(user_id)
        result["credits_granted"] = credited
        result["persisted"] = True
        result["message"] = (
            "Payment already credited."
            if credited == 0
            else f"Payment settled! {credited:,} credits added to your balance."
        )

    elif status == "Expired":
        await _retire_invoice("Expired")
        ledger = await cache.get(user_id)
        result["message"] = "Invoice expired. Create a new one with purchase_credits."

    elif status == "Invalid":
        await _retire_invoice("Invalid")
        ledger = await cache.get(user_id)
        result["message"] = "Payment invalid."

    else:
        result["message"] = f"Unknown invoice status: {status}"

    result["balance_api_sats"] = ledger.balance_api_sats
    return result


async def check_balance_tool(
    cache: LedgerCache,
    user_id: str,
) -> dict[str, Any]:
    """Return the user's current credit balance and usage summary."""
    ledger = await cache.get_fresh(user_id)
    today = date.today().isoformat()  # noqa: DTZ011

    vault_unavailable = getattr(ledger, "_vault_unavailable", False)

    result: dict[str, Any] = {
        "success": True,
        "balance_api_sats": ledger.balance_api_sats,
        "total_deposited_api_sats": ledger.total_deposited_api_sats,
        "total_consumed_api_sats": ledger.total_consumed_api_sats,
        "pending_invoices": len(ledger.pending_invoices),
        "pending_invoice_ids": list(ledger.pending_invoices),
        "last_deposit_at": ledger.last_deposit_at,
    }

    if vault_unavailable:
        result["vault_unavailable"] = True
        result["warning"] = (
            "Vault is not yet available — balance shown may be stale or zero. "
            "Try again in a moment."
        )

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
    from datetime import datetime
    now = datetime.now(UTC)
    active = [t for t in ledger.tranches if t.remaining_sats > 0 and not t.is_expired_at(now)]
    expired = [t for t in ledger.tranches if t.remaining_sats > 0 and t.is_expired_at(now)]
    result["active_tranches"] = len(active)
    result["tranches"] = [
        {
            "id": t.invoice_id or str(i),
            "amount_sats": t.original_sats,
            "remaining_sats": t.remaining_sats,
            "expires_at": t.expires_at,
            "created_at": t.granted_at,
        }
        for i, t in enumerate(active)
    ]
    if expired:
        result["expired_tranches"] = [
            {
                "id": t.invoice_id or str(i),
                "amount_sats": t.original_sats,
                "remaining_sats": t.remaining_sats,
                "expires_at": t.expires_at,
                "created_at": t.granted_at,
            }
            for i, t in enumerate(expired)
        ]

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
    tranche_lifetime_seconds: int | None = None,
) -> dict[str, Any]:
    """Restore credits from a paid invoice that was lost due to cache or vault issues."""
    # Check idempotency first
    ledger = await cache.get_fresh(user_id)
    if invoice_id in ledger.credited_invoices:
        return {
            "success": True,
            "invoice_id": invoice_id,
            "credits_granted": 0,
            "balance_api_sats": ledger.balance_api_sats,
            "message": "Invoice already credited — no duplicate credits applied.",
        }

    # Refuse to restore when the cache returned an uncached, vault-unavailable
    # ledger — any mutation here would be lost the moment this function returns,
    # and the credits_granted result would be a lie (same trap as check_payment).
    # Restore is the recovery path of last resort; it must not silently fail.
    if getattr(ledger, "_vault_unavailable", False):
        logger.error(
            "Vault unavailable during restore_credits for %s (invoice %s). "
            "Refusing to credit — would be lost on function return.",
            user_id, invoice_id,
        )
        return {
            "success": False,
            "invoice_id": invoice_id,
            "credits_granted": 0,
            "persisted": False,
            "error_code": "vault_unavailable",
            "error": (
                "Vault wasn't reachable. Restore aborted — your invoice at BTCPay "
                "is unaffected. Retry restore_credits in 10–15 seconds once the "
                "MCP has warmed up."
            ),
        }

    # Vault-first: check if we have a settled invoice record in the ledger
    vault_record = ledger.invoices.get(invoice_id)
    if vault_record and vault_record.status == "Settled" and vault_record.api_sats_credited > 0:
        # Restore from vault record — no BTCPay call needed
        credited = vault_record.api_sats_credited

        def _restore_from_record(led: UserLedger) -> int:
            # Runs against FRESH state and is re-applied on every CAS retry, so
            # it must be idempotent — `credit_deposit` is not, and a re-applied
            # restore would mint a second tranche. A tranche already carrying
            # this invoice_id IS the restore having landed, which is also the
            # precise question this tool exists to ask: is the tranche missing?
            if any(t.invoice_id == invoice_id for t in led.tranches):
                return 0
            led.credit_deposit(credited, invoice_id, ttl_seconds=tranche_lifetime_seconds)
            return credited

        try:
            granted = await cache.mutate(user_id, _restore_from_record)
        except (LedgerUnavailableError, LedgerWriteError) as exc:
            logger.error(
                "CRITICAL: vault-restored %d credits for %s (invoice %s) did NOT "
                "persist: %s: %s", credited, user_id, invoice_id,
                type(exc).__name__, exc,
            )
            return {
                "success": False,
                "invoice_id": invoice_id,
                "source": "vault_record",
                "persisted": False,
                "credits_granted": 0,
                "error_code": "vault_unavailable",
                "error": (
                    "Credits were NOT restored — the ledger could not be written. "
                    "Nothing was credited, so retrying restore_credits is safe."
                ),
            }

        ledger = await cache.get(user_id)
        return {
            "success": True,
            "invoice_id": invoice_id,
            "source": "vault_record",
            "persisted": True,
            "amount_sats": vault_record.amount_sats,
            "credits_granted": granted,
            "balance_api_sats": ledger.balance_api_sats,
            "message": (
                f"Restored {granted:,} credits from vault invoice record."
                if granted
                else "Already restored — a tranche for this invoice is present."
            ),
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

    # Restore only against the invoice's own patron — never move one patron's
    # settled invoice onto another account (defense-in-depth; restore is
    # operator-gated, so this guards operator error, not an external claim).
    owner = _invoice_owner(invoice)
    if owner is not None and owner != user_id:
        logger.warning(
            "restore_credits: invoice %s belongs to %s, not %s — refusing.",
            invoice_id, owner[:20], user_id[:20],
        )
        return {
            "success": False,
            "error": "This invoice was not created for that account.",
            "error_code": "invoice_owner_mismatch",
            "invoice_id": invoice_id,
        }

    # Credit the balance
    amount_str = invoice.get("amount", "0")
    amount_sats = int(float(amount_str))
    credited = amount_sats

    # Hoisted out of the mutation: a CAS retry re-applies the function, and a
    # settled_at computed inside would drift on each attempt.
    settled_at = datetime.now(UTC).isoformat()

    def _restore_from_btcpay(led: UserLedger) -> int:
        # Idempotent for the same reason as the vault-record path above: this
        # runs again on every conflict retry, and `credit_deposit` would happily
        # mint a second tranche for an invoice already restored.
        if any(t.invoice_id == invoice_id for t in led.tranches):
            return 0
        led.credit_deposit(credited, invoice_id, ttl_seconds=tranche_lifetime_seconds)
        led.record_invoice_settled(
            invoice_id=invoice_id,
            api_sats_credited=credited,
            settled_at=settled_at,
            btcpay_status=status,
        )
        return credited

    try:
        granted = await cache.mutate(user_id, _restore_from_btcpay)
    except (LedgerUnavailableError, LedgerWriteError) as exc:
        logger.error(
            "CRITICAL: restored %d credits for %s (invoice %s) did NOT persist: "
            "%s: %s", credited, user_id, invoice_id, type(exc).__name__, exc,
        )
        return {
            "success": False,
            "error": (
                "Credits were NOT restored — the ledger could not be written. "
                "Nothing was credited, so retrying restore_credits is safe."
            ),
            "invoice_id": invoice_id,
            "credits_granted": 0,
            "persisted": False,
        }

    ledger = await cache.get(user_id)
    return {
        "success": True,
        "invoice_id": invoice_id,
        "source": "btcpay",
        "amount_sats": amount_sats,
        "credits_granted": granted,
        "balance_api_sats": ledger.balance_api_sats,
        "persisted": True,
        "message": (
            f"Restored {granted:,} credits from invoice {invoice_id}."
            if granted
            else "Already restored — a tranche for this invoice is present."
        ),
    }


async def reconcile_pending_invoices(
    btcpay: BTCPayClient,
    cache: LedgerCache,
    user_id: str,
    tranche_lifetime_seconds: int | None = None,
) -> dict[str, Any]:
    """Reconcile pending invoices on startup: credit settled, remove terminal."""
    ledger = await cache.get_fresh(user_id)
    pending_copy = list(ledger.pending_invoices)
    if not pending_copy:
        return {"reconciled": 0, "actions": []}

    # Ask BTCPay what happened to each invoice FIRST. The mutation below is
    # synchronous and re-runs on every CAS retry, so no network call may live
    # inside it — and re-polling BTCPay once per lost race would be wasteful
    # besides. Decide here; apply once, atomically.
    to_settle: list[tuple[str, int, str]] = []
    to_retire: list[tuple[str, str]] = []

    for invoice_id in pending_copy:
        try:
            invoice = await btcpay.get_invoice(invoice_id)
        except BTCPayError:
            logger.warning("Reconciliation: skipping %s (BTCPay error).", invoice_id)
            continue

        status = invoice.get("status", "Unknown")
        if status == "Settled":
            to_settle.append((invoice_id, int(float(invoice.get("amount", "0"))), status))
        elif status in ("Expired", "Invalid"):
            to_retire.append((invoice_id, status))

    if not to_settle and not to_retire:
        return {"reconciled": 0, "actions": []}

    settled_at = datetime.now(UTC).isoformat()

    def _reconcile(led: UserLedger) -> list[dict[str, Any]]:
        # The already-credited check moved in here deliberately: against fresh
        # state it is the definitive answer, where the old check read a cache
        # another replica may already have advanced past.
        applied: list[dict[str, Any]] = []
        for invoice_id, sats, status in to_settle:
            if invoice_id in led.credited_invoices:
                continue
            led.credit_deposit(sats, invoice_id, ttl_seconds=tranche_lifetime_seconds)
            led.record_invoice_settled(
                invoice_id=invoice_id,
                api_sats_credited=sats,
                settled_at=settled_at,
                btcpay_status=status,
            )
            applied.append({
                "invoice_id": invoice_id, "action": "credited", "api_sats": sats,
            })
        for invoice_id, status in to_retire:
            if invoice_id in led.pending_invoices:
                led.pending_invoices.remove(invoice_id)
            led.record_invoice_terminal(invoice_id, status, status)
            applied.append({
                "invoice_id": invoice_id, "action": "removed", "reason": status,
            })
        return applied

    try:
        actions = await cache.mutate(user_id, _reconcile)
    except (LedgerUnavailableError, LedgerWriteError) as exc:
        # Nothing was written, so nothing is half-done — the next reconcile
        # re-polls BTCPay and applies the same conclusions.
        logger.error(
            "Reconciliation for %s did not persist: %s: %s",
            user_id, type(exc).__name__, exc,
        )
        return {"reconciled": 0, "actions": [], "persisted": False}

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
    ledger = await cache.get_fresh(user_id)
    now = datetime.now(UTC)
    today = date.today()  # noqa: DTZ011

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

    if connection_vars_present and btcpay is not None:
        # Health check
        try:
            await btcpay.health_check()
            result["server_reachable"] = True
        except BTCPayError:
            result["server_reachable"] = False
        except Exception:  # noqa: BLE001
            result["server_reachable"] = False

        # Store check
        try:
            store = await btcpay.get_store()
            result["store_name"] = store.get("name", "unknown")
        except BTCPayAuthError:
            result["store_name"] = "unauthorized"
        except BTCPayError:
            result["store_name"] = None
        except Exception:  # noqa: BLE001
            result["store_name"] = None

        # API key permissions check
        try:
            key_info = await btcpay.get_api_key_info()
            permissions = key_info.get("permissions", [])
            required = ["btcpay.store.cancreateinvoice", "btcpay.store.canviewinvoices"]
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
        except Exception as e:  # noqa: BLE001
            result["api_key_permissions"] = {"error": str(e)}
    else:
        result["server_reachable"] = None
        result["store_name"] = None

    return result
