"""Tollbooth DPYC — Don't Pester Your Customer.

Bitcoin Lightning micropayments for MCP servers.
"""

from importlib.metadata import version as _meta_version

__version__ = _meta_version("tollbooth-dpyc")

from tollbooth.certificate import CertificateError, verify_certificate_auto, UNDERSTOOD_PROTOCOLS
from tollbooth.registry import DPYCRegistry, RegistryError, resolve_authority_npub, resolve_authority_service, resolve_oracle_service, resolve_service_by_name, DEFAULT_REGISTRY_URL
from tollbooth.config import TollboothConfig
from tollbooth.ledger import UserLedger, ToolUsage, InvoiceRecord, Tranche
from tollbooth.btcpay_client import BTCPayClient, BTCPayError, BTCPayAuthError
from tollbooth.vault_backend import VaultBackend
from tollbooth.credential_vault_backend import CredentialVaultBackend, SessionBindingBackend
from tollbooth.actor_types import ActorRole, ObsoletePractice, ToolPath, ToolPathInfo
from tollbooth.slug_tools import make_slug_tool
from tollbooth.operator_protocol import (
    OperatorProtocol,
    OPERATOR_BASE_CATALOG,
    OPERATOR_OBSOLETE_PRACTICES,
    OperatorConformanceError,
    validate_operator_tools,
    async_validate_operator_tools,
)
from tollbooth.authority_protocol import AuthorityProtocol, AUTHORITY_BASE_CATALOG
from tollbooth.oracle_protocol import OracleProtocol
from tollbooth.ledger_cache import LedgerCache
from tollbooth.constants import MAX_INVOICE_SATS, LOW_BALANCE_FLOOR_API_SATS, ECOSYSTEM_LINKS
from tollbooth.tool_identity import ToolIdentity, STANDARD_IDENTITIES, capability_uuid
from tollbooth.pricing import ToolPricing
from tollbooth.pricing_model import PricingModel, ToolPrice, PipelineStep
from tollbooth.vaults import TheBrainVault, NeonVault, NeonQueryError
from tollbooth.bootstrap import BootstrapClient, BootstrapResult, ensure_bootstrapped
from tollbooth.runtime import OperatorRuntime, resolve_npub
from tollbooth.session_cache import SessionCache
from tollbooth.patron_session import PatronSessionCache
from tollbooth.shortlinks import create_shortlink
from tollbooth.vault_encryption import VaultCipher
from tollbooth.ots import MerkleTree, InclusionProof, OTSCalendarClient
from tollbooth.nostr_certificate import verify_nostr_certificate, NOSTR_CERT_KIND
from tollbooth.identity_credential import (
    sign_identity_credential,
    verify_identity_credential,
    verify_credential_chain,
    IdentityCredentialError,
    IDENTITY_CREDENTIAL_KIND,
    IDENTITY_CREDENTIAL_TAG,
    IDENTITY_CREDENTIAL_LABEL,
)

try:
    from tollbooth.authority_client import AuthorityCertifier, AuthorityCertifyError
except ImportError:
    AuthorityCertifier = None  # type: ignore[assignment,misc]
    AuthorityCertifyError = None  # type: ignore[assignment,misc]

try:
    from tollbooth.oracle_client import OracleClient, OracleClientError
except ImportError:
    OracleClient = None  # type: ignore[assignment,misc]
    OracleClientError = None  # type: ignore[assignment,misc]

try:
    from tollbooth.pricing_store import PricingModelStore
except ImportError:
    PricingModelStore = None  # type: ignore[assignment,misc]

try:
    from tollbooth.pricing_resolver import PricingResolver
except ImportError:
    PricingResolver = None  # type: ignore[assignment,misc]

try:
    from tollbooth.nostr_audit import NostrAuditPublisher, AuditedVault
except ImportError:
    NostrAuditPublisher = None  # type: ignore[assignment,misc]
    AuditedVault = None  # type: ignore[assignment,misc]

try:
    from tollbooth.identity_proof import (
        verify_proof, create_proof,
        PROOF_EVENT_KIND, OWNERSHIP_SENTINEL,
    )
    from tollbooth.proven_npub import ProvenNpubCache, ProvenNpub
except ImportError:
    verify_proof = None  # type: ignore[assignment,misc]
    create_proof = None  # type: ignore[assignment,misc]
    PROOF_EVENT_KIND = None  # type: ignore[assignment,misc]
    OWNERSHIP_SENTINEL = None  # type: ignore[assignment,misc]
    ProvenNpubCache = None  # type: ignore[assignment,misc]
    ProvenNpub = None  # type: ignore[assignment,misc]

try:
    from tollbooth.acl_verify import verify_acl_event, check_acl_access, ACL_EVENT_KIND
except ImportError:
    verify_acl_event = None  # type: ignore[assignment,misc]
    check_acl_access = None  # type: ignore[assignment,misc]
    ACL_EVENT_KIND = None  # type: ignore[assignment,misc]

try:
    from tollbooth.acl_store import ToolAclStore
except ImportError:
    ToolAclStore = None  # type: ignore[assignment,misc]

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
    from tollbooth.credential_card import (
        CREDENTIAL_CARD_KIND,
        CredentialCardError,
        CredentialCardExpired,
        CredentialCardInvalid,
        decode_credential_card,
        encode_credential_card,
        render_qr,
    )
except ImportError:
    CREDENTIAL_CARD_KIND = None  # type: ignore[assignment,misc]
    CredentialCardError = None  # type: ignore[assignment,misc]
    CredentialCardExpired = None  # type: ignore[assignment,misc]
    CredentialCardInvalid = None  # type: ignore[assignment,misc]
    decode_credential_card = None  # type: ignore[assignment,misc]
    encode_credential_card = None  # type: ignore[assignment,misc]
    render_qr = None  # type: ignore[assignment,misc]

try:
    from tollbooth.nostr_diagnostics import (
        courier_health,
        courier_ping,
        probe_relay_liveness,
        resolve_relays,
        DEFAULT_RELAY,
        FALLBACK_RELAY_POOL,
    )
except ImportError:
    courier_health = None  # type: ignore[assignment,misc]
    courier_ping = None  # type: ignore[assignment,misc]
    probe_relay_liveness = None  # type: ignore[assignment,misc]
    resolve_relays = None  # type: ignore[assignment,misc]
    DEFAULT_RELAY = None  # type: ignore[assignment,misc]
    FALLBACK_RELAY_POOL = None  # type: ignore[assignment,misc]

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
        SurgePricingConstraint,
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
    SurgePricingConstraint = None  # type: ignore[assignment,misc]
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
    "SessionBindingBackend",
    # Actor Protocols
    "ActorRole",
    "ObsoletePractice",
    "ToolPath",
    "ToolPathInfo",
    "OperatorProtocol",
    "OPERATOR_BASE_CATALOG",
    "OPERATOR_OBSOLETE_PRACTICES",
    "OperatorConformanceError",
    "validate_operator_tools",
    "async_validate_operator_tools",
    "AuthorityProtocol",
    "AUTHORITY_BASE_CATALOG",
    "OracleProtocol",
    "make_slug_tool",
    "LedgerCache",
    "TheBrainVault",
    "NeonVault",
    "NeonQueryError",
    "ToolPricing",
    "PricingModel",
    "ToolPrice",
    "PipelineStep",
    "PricingModelStore",
    "PricingResolver",
    "ToolIdentity",
    "STANDARD_IDENTITIES",
    "capability_uuid",
    "MAX_INVOICE_SATS",
    "LOW_BALANCE_FLOOR_API_SATS",
    "ECOSYSTEM_LINKS",
    "verify_certificate_auto",
    "verify_nostr_certificate",
    "NOSTR_CERT_KIND",
    "UNDERSTOOD_PROTOCOLS",
    # Identity Credential
    "sign_identity_credential",
    "verify_identity_credential",
    "verify_credential_chain",
    "IdentityCredentialError",
    "IDENTITY_CREDENTIAL_KIND",
    "IDENTITY_CREDENTIAL_TAG",
    "IDENTITY_CREDENTIAL_LABEL",
    # Registry
    "DPYCRegistry",
    "RegistryError",
    "resolve_authority_npub",
    "resolve_authority_service",
    "resolve_oracle_service",
    "resolve_service_by_name",
    "DEFAULT_REGISTRY_URL",
    # Authority Client
    "AuthorityCertifier",
    "AuthorityCertifyError",
    # Oracle Client
    "OracleClient",
    "OracleClientError",
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
    # Credential Card
    "CREDENTIAL_CARD_KIND",
    "CredentialCardError",
    "CredentialCardExpired",
    "CredentialCardInvalid",
    "encode_credential_card",
    "decode_credential_card",
    "render_qr",
    # Operator Proof & Npub Ownership
    "verify_proof",
    "create_proof",
    "PROOF_EVENT_KIND",
    "OWNERSHIP_SENTINEL",
    "ProvenNpubCache",
    "ProvenNpub",
    # ACL
    "verify_acl_event",
    "check_acl_access",
    "ACL_EVENT_KIND",
    "ToolAclStore",
    "courier_health",
    "courier_ping",
    "probe_relay_liveness",
    "resolve_relays",
    "DEFAULT_RELAY",
    "FALLBACK_RELAY_POOL",
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
    "SurgePricingConstraint",
    "load_constraints",
    "validate_config",
    # Bootstrap & Runtime
    "BootstrapClient",
    "BootstrapResult",
    "ensure_bootstrapped",
    "OperatorRuntime",
    "resolve_npub",
    "VaultCipher",
    # Session Caches
    "SessionCache",
    "PatronSessionCache",
    # Shortlinks
    "create_shortlink",
]
