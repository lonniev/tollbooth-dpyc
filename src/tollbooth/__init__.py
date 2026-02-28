"""Tollbooth DPYC — Don't Pester Your Customer.

Bitcoin Lightning micropayments for MCP servers.
"""

__version__ = "0.1.44"

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

try:
    from tollbooth.secure_courier import SecureCourierService
except ImportError:
    SecureCourierService = None  # type: ignore[assignment,misc]

try:
    from tollbooth.nostr_diagnostics import courier_health, courier_ping
except ImportError:
    courier_health = None  # type: ignore[assignment,misc]
    courier_ping = None  # type: ignore[assignment,misc]

try:
    from tollbooth.nostr_notifications import NotificationManager, NotificationPreferences
except ImportError:
    NotificationManager = None  # type: ignore[assignment,misc]
    NotificationPreferences = None  # type: ignore[assignment,misc]

try:
    from tollbooth.constraints import (
        ToolConstraint,
        ConstraintContext,
        ConstraintResult,
        PriceModifier,
        LedgerSnapshot,
        PatronIdentity,
        EnvironmentSnapshot,
        ConstraintEngine,
        ConstraintGate,
        TemporalWindowConstraint,
        FiniteSupplyConstraint,
        PeriodicRefreshConstraint,
        CouponConstraint,
        FreeTrialConstraint,
        LoyaltyDiscountConstraint,
        BulkBonusConstraint,
        HappyHourConstraint,
        JsonExpressionConstraint,
        load_constraints,
        validate_config,
    )
except ImportError:
    ToolConstraint = None  # type: ignore[assignment,misc]
    ConstraintContext = None  # type: ignore[assignment,misc]
    ConstraintResult = None  # type: ignore[assignment,misc]
    PriceModifier = None  # type: ignore[assignment,misc]
    LedgerSnapshot = None  # type: ignore[assignment,misc]
    PatronIdentity = None  # type: ignore[assignment,misc]
    EnvironmentSnapshot = None  # type: ignore[assignment,misc]
    ConstraintEngine = None  # type: ignore[assignment,misc]
    ConstraintGate = None  # type: ignore[assignment,misc]
    TemporalWindowConstraint = None  # type: ignore[assignment,misc]
    FiniteSupplyConstraint = None  # type: ignore[assignment,misc]
    PeriodicRefreshConstraint = None  # type: ignore[assignment,misc]
    CouponConstraint = None  # type: ignore[assignment,misc]
    FreeTrialConstraint = None  # type: ignore[assignment,misc]
    LoyaltyDiscountConstraint = None  # type: ignore[assignment,misc]
    BulkBonusConstraint = None  # type: ignore[assignment,misc]
    HappyHourConstraint = None  # type: ignore[assignment,misc]
    JsonExpressionConstraint = None  # type: ignore[assignment,misc]
    load_constraints = None  # type: ignore[assignment,misc]
    validate_config = None  # type: ignore[assignment,misc]

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
    "SecureCourierService",
    "courier_health",
    "courier_ping",
    "NotificationManager",
    "NotificationPreferences",
    # Constraint Engine
    "ConstraintGate",
    "ToolConstraint",
    "ConstraintContext",
    "ConstraintResult",
    "PriceModifier",
    "LedgerSnapshot",
    "PatronIdentity",
    "EnvironmentSnapshot",
    "ConstraintEngine",
    "TemporalWindowConstraint",
    "FiniteSupplyConstraint",
    "PeriodicRefreshConstraint",
    "CouponConstraint",
    "FreeTrialConstraint",
    "LoyaltyDiscountConstraint",
    "BulkBonusConstraint",
    "HappyHourConstraint",
    "JsonExpressionConstraint",
    "load_constraints",
    "validate_config",
]
