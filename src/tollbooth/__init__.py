"""Tollbooth DPYC — Don't Pester Your Customer.

Bitcoin Lightning micropayments for MCP servers.
"""

__version__ = "0.1.33"

from tollbooth.certificate import CertificateError, verify_certificate_auto, UNDERSTOOD_PROTOCOLS
from tollbooth.config import TollboothConfig
from tollbooth.ledger import UserLedger, ToolUsage, InvoiceRecord, Tranche
from tollbooth.btcpay_client import BTCPayClient, BTCPayError, BTCPayAuthError
from tollbooth.vault_backend import VaultBackend
from tollbooth.credential_vault_backend import CredentialVaultBackend
from tollbooth.ledger_cache import LedgerCache
from tollbooth.constants import ToolTier, MAX_INVOICE_SATS, LOW_BALANCE_FLOOR_API_SATS
from tollbooth.pricing import ToolPricing
from tollbooth.vaults import TheBrainVault, NeonVault, NeonQueryError
from tollbooth.ots import MerkleTree, InclusionProof, OTSCalendarClient
from tollbooth.nostr_certificate import verify_nostr_certificate, NOSTR_CERT_KIND

try:
    from tollbooth.nostr_audit import NostrAuditPublisher, AuditedVault
except ImportError:
    NostrAuditPublisher = None  # type: ignore[assignment,misc]
    AuditedVault = None  # type: ignore[assignment,misc]

try:
    from tollbooth.nostr_credentials import (
        NostrCredentialExchange,
        NostrProfile,
        CourierError,
        CourierNotReady,
        CourierTimeout,
        CourierValidationError,
    )
    from tollbooth.credential_templates import (
        CredentialTemplate,
        FieldSpec,
        TemplateValidationError,
    )
except ImportError:
    NostrCredentialExchange = None  # type: ignore[assignment,misc]
    NostrProfile = None  # type: ignore[assignment,misc]
    CourierError = None  # type: ignore[assignment,misc]
    CourierNotReady = None  # type: ignore[assignment,misc]
    CourierTimeout = None  # type: ignore[assignment,misc]
    CourierValidationError = None  # type: ignore[assignment,misc]
    CredentialTemplate = None  # type: ignore[assignment,misc]
    FieldSpec = None  # type: ignore[assignment,misc]
    TemplateValidationError = None  # type: ignore[assignment,misc]

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
    "CredentialVaultBackend",
    "LedgerCache",
    "TheBrainVault",
    "NeonVault",
    "NeonQueryError",
    "ToolPricing",
    "ToolTier",
    "MAX_INVOICE_SATS",
    "LOW_BALANCE_FLOOR_API_SATS",
    "verify_certificate_auto",
    "verify_nostr_certificate",
    "NOSTR_CERT_KIND",
    "UNDERSTOOD_PROTOCOLS",
    "NostrAuditPublisher",
    "AuditedVault",
    "MerkleTree",
    "InclusionProof",
    "OTSCalendarClient",
    "NostrCredentialExchange",
    "NostrProfile",
    "CourierError",
    "CourierNotReady",
    "CourierTimeout",
    "CourierValidationError",
    "CredentialTemplate",
    "FieldSpec",
    "TemplateValidationError",
]
