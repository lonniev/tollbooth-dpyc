"""Constants for Tollbooth micropayment gating."""

from enum import IntEnum


MAX_INVOICE_SATS = 1_000_000  # 0.01 BTC cap per invoice
LOW_BALANCE_FLOOR_API_SATS = 100  # minimum warning threshold


class ToolTier(IntEnum):
    """Cost tiers for tool-call metering (satoshis per call)."""

    FREE = 0
    READ = 1
    WRITE = 5
    HEAVY = 10
