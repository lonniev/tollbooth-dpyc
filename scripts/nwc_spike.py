"""Pay a Lightning address from your own wallet over Nostr Wallet Connect.

A hand-run check of ``tollbooth.nwc`` against a real wallet. It moves real
sats, so it asks before it pays.

    python scripts/nwc_spike.py you@your-wallet.example 21
    python scripts/nwc_spike.py --lookup <payment hash>

The connection string is typed at a hidden prompt. It is never printed, never
written to disk, and never taken from the command line, where it would land in
shell history.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from datetime import UTC, datetime

from tollbooth import bolt11, nwc
from tollbooth.lnurl import resolve_lightning_address


def _connection() -> nwc.NwcConnection:
    if not sys.stdin.isatty():
        sys.exit("Run this in a terminal: the connection string is typed at a hidden prompt.")
    return nwc.parse_connection_uri(getpass.getpass("Wallet connection string (hidden): "))


async def _show_lookup(conn: nwc.NwcConnection, payment_hash: str) -> None:
    try:
        seen = await nwc.lookup_invoice(conn, payment_hash=payment_hash)
    except nwc.NwcError as exc:
        print(f"   wallet says: {exc.code} — {exc.message}")
        return
    print(f"   state={seen.state}  settled={seen.settled}  fees_msat={seen.fees_paid_msat}")


async def _pay(address: str, sats: int) -> None:
    conn = _connection()

    print("1. Asking the wallet what this connection may do…")
    info = await nwc.get_info(conn)
    print(f"   wallet={info.alias}  network={info.network}  may: {', '.join(info.methods)}")

    print(f"2. Asking {address} for an invoice for {sats} sats…")
    invoice = await resolve_lightning_address(address, sats)
    asked = bolt11.decode(invoice)
    expires = datetime.fromtimestamp(asked.expires_at, UTC).isoformat(timespec="seconds")
    print(f"   invoice asks for {asked.amount_msat} msat, expires {expires}")
    print(f"3. Payment hash (record this BEFORE paying): {asked.payment_hash}")

    if input(f"4. Type 'pay' to send {sats} real sats: ").strip().lower() != "pay":
        sys.exit("   Nothing was sent.")

    try:
        paid = await nwc.pay_invoice(conn, invoice, amount_msat=sats * 1000)
    except nwc.NwcAmountMismatch as exc:
        sys.exit(f"   Refused before sending: {exc}")
    except nwc.NwcError as exc:
        sys.exit(f"   The wallet said no: {exc.code} — {exc.message}")
    except nwc.NwcNotSent as exc:
        sys.exit(f"   Nothing was sent: {exc}")
    except nwc.NwcOutcomeUnknown as exc:
        print(f"   Outcome unknown: {exc}. Looking the payment up…")
        await _show_lookup(conn, asked.payment_hash)
        return

    print(f"   PAID. preimage={paid.preimage}  fees_msat={paid.fees_paid_msat}")
    print("5. Asking the wallet to confirm by payment hash…")
    await _show_lookup(conn, paid.payment_hash)


async def _lookup(payment_hash: str) -> None:
    await _show_lookup(_connection(), payment_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("address", nargs="?", help="payee Lightning address, user@domain")
    parser.add_argument("sats", nargs="?", type=int, help="whole sats to pay")
    parser.add_argument("--lookup", metavar="HASH", help="only look a payment up by its hash")
    args = parser.parse_args()

    if args.lookup:
        asyncio.run(_lookup(args.lookup))
    elif args.address and args.sats and args.sats > 0:
        asyncio.run(_pay(args.address, args.sats))
    else:
        parser.error("give a Lightning address and a positive number of sats, or --lookup HASH")


if __name__ == "__main__":
    main()
