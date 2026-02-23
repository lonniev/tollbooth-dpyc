"""Tollbooth DPYC — Don't Pester Your Customer.

Bitcoin Lightning micropayments for MCP servers.
"""

__version__ = "0.1.23"

from tollbooth.certificate import CertificateError, verify_certificate, normalize_public_key, key_fingerprint, UNDERSTOOD_PROTOCOLS
from tollbooth.config import TollboothConfig
from tollbooth.ledger import UserLedger, ToolUsage, InvoiceRecord, Tranche
from tollbooth.btcpay_client import BTCPayClient, BTCPayError, BTCPayAuthError
from tollbooth.vault_backend import VaultBackend
from tollbooth.ledger_cache import LedgerCache
from tollbooth.constants import ToolTier, MAX_INVOICE_SATS, LOW_BALANCE_FLOOR_API_SATS
from tollbooth.vaults import TheBrainVault, NeonVault, NeonQueryError
from tollbooth.ots import MerkleTree, InclusionProof, OTSCalendarClient

try:
    from tollbooth.nostr_audit import NostrAuditPublisher, AuditedVault
except ImportError:
    NostrAuditPublisher = None  # type: ignore[assignment,misc]
    AuditedVault = None  # type: ignore[assignment,misc]

__all__ = [
    "CertificateError",
    "TollboothConfig",
    "UserLedger",
    "ToolUsage",
    "InvoiceRecord",
    "Tranche",
    "BTCPayClient",
    "BTCPayError",
    "BTCPayAuthError",
    "VaultBackend",
    "LedgerCache",
    "TheBrainVault",
    "NeonVault",
    "NeonQueryError",
    "ToolTier",
    "MAX_INVOICE_SATS",
    "LOW_BALANCE_FLOOR_API_SATS",
    "verify_certificate",
    "normalize_public_key",
    "key_fingerprint",
    "UNDERSTOOD_PROTOCOLS",
    "NostrAuditPublisher",
    "AuditedVault",
    "MerkleTree",
    "InclusionProof",
    "OTSCalendarClient",
]
