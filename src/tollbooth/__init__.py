"""Tollbooth DPYC — Don't Pester Your Customer.

Bitcoin Lightning micropayments for MCP servers.
"""

from tollbooth.version import resolve_service_version

__version__ = resolve_service_version("tollbooth-dpyc", __file__)

from tollbooth.actor_types import ActorRole, ObsoletePractice, ToolPath, ToolPathInfo
from tollbooth.authority_protocol import AUTHORITY_BASE_CATALOG, AuthorityProtocol
from tollbooth.bootstrap import BootstrapClient, BootstrapResult, ensure_bootstrapped
from tollbooth.btcpay_client import BTCPayAuthError, BTCPayClient, BTCPayError
from tollbooth.certificate import UNDERSTOOD_PROTOCOLS, CertificateError, verify_certificate_auto
from tollbooth.config import TollboothConfig
from tollbooth.constants import ECOSYSTEM_LINKS, LOW_BALANCE_FLOOR_API_SATS, MAX_INVOICE_SATS
from tollbooth.credential_vault_backend import CredentialVaultBackend, SessionBindingBackend
from tollbooth.identity_credential import (
    IDENTITY_CREDENTIAL_KIND,
    IDENTITY_CREDENTIAL_LABEL,
    IDENTITY_CREDENTIAL_TAG,
    IdentityCredentialError,
    sign_identity_credential,
    verify_credential_chain,
    verify_identity_credential,
)
from tollbooth.ledger import InvoiceRecord, ToolUsage, Tranche, UserLedger
from tollbooth.ledger_cache import LedgerCache
from tollbooth.nostr_certificate import NOSTR_CERT_KIND, verify_nostr_certificate
from tollbooth.oauth_config import OAuthProviderConfig
from tollbooth.operator_protocol import (
    OPERATOR_BASE_CATALOG,
    OPERATOR_OBSOLETE_PRACTICES,
    OperatorConformanceError,
    OperatorProtocol,
    async_validate_operator_tools,
    validate_operator_tools,
)
from tollbooth.oracle_protocol import OracleProtocol
from tollbooth.ots import InclusionProof, MerkleTree, OTSCalendarClient
from tollbooth.patron_session import PatronSessionCache
from tollbooth.pricing import ToolPricing
from tollbooth.pricing_model import PipelineStep, PricingModel, ToolPrice
from tollbooth.registry import (
    DEFAULT_REGISTRY_URL,
    DPYCRegistry,
    RegistryError,
    resolve_authority_npub,
    resolve_authority_service,
    resolve_oracle_service,
    resolve_service_by_name,
)
from tollbooth.runtime import OperatorRuntime, resolve_npub
from tollbooth.session_cache import SessionCache
from tollbooth.shortlinks import create_shortlink
from tollbooth.slug_tools import make_slug_tool
from tollbooth.tool_identity import STANDARD_IDENTITIES, ToolIdentity, capability_uuid
from tollbooth.vault_backend import VaultBackend
from tollbooth.vault_encryption import VaultCipher
from tollbooth.vaults import NeonQueryError, NeonVault, TheBrainVault

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
    from tollbooth.nostr_audit import AuditedVault, NostrAuditPublisher
except ImportError:
    NostrAuditPublisher = None  # type: ignore[assignment,misc]
    AuditedVault = None  # type: ignore[assignment,misc]

try:
    from tollbooth.identity_proof import (
        OWNERSHIP_SENTINEL,
        PROOF_EVENT_KIND,
        create_proof,
        verify_proof,
    )
    from tollbooth.proven_npub import ProvenNpub, ProvenNpubCache
except ImportError:
    verify_proof = None  # type: ignore[assignment,misc]
    create_proof = None  # type: ignore[assignment,misc]
    PROOF_EVENT_KIND = None  # type: ignore[assignment,misc]
    OWNERSHIP_SENTINEL = None  # type: ignore[assignment,misc]
    ProvenNpubCache = None  # type: ignore[assignment,misc]
    ProvenNpub = None  # type: ignore[assignment,misc]

try:
    from tollbooth.credential_templates import (
        CredentialTemplate,
        FieldSpec,
        TemplateValidationError,
    )
    from tollbooth.nostr_credentials import (
        CourierError,
        CourierNotReady,
        CourierTimeout,
        CourierValidationError,
        NostrCredentialExchange,
        NostrProfile,
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
    )
except ImportError:
    courier_health = None  # type: ignore[assignment,misc]
    courier_ping = None  # type: ignore[assignment,misc]
    probe_relay_liveness = None  # type: ignore[assignment,misc]
    resolve_relays = None  # type: ignore[assignment,misc]

try:
    from tollbooth.relay_registry import (
        RelayRegistry,
        RelayRegistryError,
        get_relays,
    )
except ImportError:
    RelayRegistry = None  # type: ignore[assignment,misc]
    RelayRegistryError = None  # type: ignore[assignment,misc]
    get_relays = None  # type: ignore[assignment,misc]

try:
    from tollbooth.nostr_notifications import NotificationManager, NotificationPreferences
except ImportError:
    NotificationManager = None  # type: ignore[assignment,misc]
    NotificationPreferences = None  # type: ignore[assignment,misc]

try:
    from tollbooth.x402_client import X402Client, x402_wallet_template
except ImportError:
    X402Client = None  # type: ignore[assignment,misc]
    x402_wallet_template = None  # type: ignore[assignment,misc]

from tollbooth.async_situation import (
    AsyncJobSituation,
    situation_response_from_row,
)
from tollbooth.llm_route import (
    LlmRoute,
    build_messages_request,
    classify_llm_failure,
    llm_failure_situation,
    model_for,
    resolve_route,
    web_fetch_tool,
    web_search_tool,
)
from tollbooth.upstream_payment import (
    classify_upstream_payment,
    is_x402_payment_challenge,
    upstream_payment_situation,
)

try:
    from tollbooth.constraints import (
        BulkBonusConstraint,
        ConstraintContext,
        ConstraintGate,
        ConstraintResult,
        CouponConstraint,
        EnvironmentSnapshot,
        FiniteSupplyConstraint,
        FreeTrialConstraint,
        HappyHourConstraint,
        JsonExpressionConstraint,
        LedgerSnapshot,
        LoyaltyDiscountConstraint,
        PatronIdentity,
        PeriodicRefreshConstraint,
        PriceModifier,
        SurgePricingConstraint,
        TemporalWindowConstraint,
        ToolConstraint,
        validate_step,
    )
except ImportError:
    ToolConstraint = None  # type: ignore[assignment,misc]
    ConstraintContext = None  # type: ignore[assignment,misc]
    ConstraintResult = None  # type: ignore[assignment,misc]
    PriceModifier = None  # type: ignore[assignment,misc]
    LedgerSnapshot = None  # type: ignore[assignment,misc]
    PatronIdentity = None  # type: ignore[assignment,misc]
    EnvironmentSnapshot = None  # type: ignore[assignment,misc]
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
    validate_step = None  # type: ignore[assignment,misc]

__all__ = [
    "AUTHORITY_BASE_CATALOG",
    # Credential Card
    "CREDENTIAL_CARD_KIND",
    "DEFAULT_REGISTRY_URL",
    "ECOSYSTEM_LINKS",
    "IDENTITY_CREDENTIAL_KIND",
    "IDENTITY_CREDENTIAL_LABEL",
    "IDENTITY_CREDENTIAL_TAG",
    "LOW_BALANCE_FLOOR_API_SATS",
    "MAX_INVOICE_SATS",
    "NOSTR_CERT_KIND",
    "OPERATOR_BASE_CATALOG",
    "OPERATOR_OBSOLETE_PRACTICES",
    "OWNERSHIP_SENTINEL",
    "PROOF_EVENT_KIND",
    "STANDARD_IDENTITIES",
    "UNDERSTOOD_PROTOCOLS",
    # Actor Protocols
    "ActorRole",
    "AsyncJobSituation",
    "AuditedVault",
    # Authority Client
    "AuthorityCertifier",
    "AuthorityCertifyError",
    "AuthorityProtocol",
    "BTCPayAuthError",
    "BTCPayClient",
    "BTCPayError",
    # Bootstrap & Runtime
    "BootstrapClient",
    "BootstrapResult",
    "BulkBonusConstraint",
    "CertificateError",
    "ConstraintContext",
    # Constraint Engine
    "ConstraintGate",
    "ConstraintResult",
    "CouponConstraint",
    "CourierError",
    "CourierNotReady",
    "CourierTimeout",
    "CourierValidationError",
    "CredentialCardError",
    "CredentialCardExpired",
    "CredentialCardInvalid",
    "CredentialTemplate",
    "CredentialVaultBackend",
    # Registry
    "DPYCRegistry",
    "EnvironmentSnapshot",
    "FieldSpec",
    "FiniteSupplyConstraint",
    "FreeTrialConstraint",
    "HappyHourConstraint",
    "IdentityCredentialError",
    "InclusionProof",
    "InvoiceRecord",
    "JsonExpressionConstraint",
    "LedgerCache",
    "LedgerSnapshot",
    "LlmRoute",
    "LoyaltyDiscountConstraint",
    "MerkleTree",
    "NeonQueryError",
    "NeonVault",
    "NostrAuditPublisher",
    "NostrCredentialExchange",
    "NostrProfile",
    "NotificationManager",
    "NotificationPreferences",
    "OAuthProviderConfig",
    "OTSCalendarClient",
    "ObsoletePractice",
    "OperatorConformanceError",
    "OperatorProtocol",
    "OperatorRuntime",
    # Oracle Client
    "OracleClient",
    "OracleClientError",
    "OracleProtocol",
    "PatronIdentity",
    "PatronSessionCache",
    "PeriodicRefreshConstraint",
    "PipelineStep",
    "PriceModifier",
    "PricingModel",
    "PricingModelStore",
    "PricingResolver",
    "ProvenNpub",
    "ProvenNpubCache",
    "RegistryError",
    "RelayRegistry",
    "RelayRegistryError",
    "SecureCourierService",
    "SessionBindingBackend",
    # Session Caches
    "SessionCache",
    "SurgePricingConstraint",
    "TemplateValidationError",
    "TemporalWindowConstraint",
    "TheBrainVault",
    "TollboothConfig",
    "ToolConstraint",
    "ToolIdentity",
    "ToolPath",
    "ToolPathInfo",
    "ToolPrice",
    "ToolPricing",
    "ToolUsage",
    "Tranche",
    "UserLedger",
    "VaultBackend",
    "VaultCipher",
    "X402Client",
    "async_validate_operator_tools",
    "build_messages_request",
    "capability_uuid",
    "classify_llm_failure",
    "classify_upstream_payment",
    "courier_health",
    "courier_ping",
    "create_proof",
    # Shortlinks
    "create_shortlink",
    "decode_credential_card",
    "encode_credential_card",
    "ensure_bootstrapped",
    "get_relays",
    "is_x402_payment_challenge",
    "llm_failure_situation",
    "make_slug_tool",
    "model_for",
    "probe_relay_liveness",
    "render_qr",
    "resolve_authority_npub",
    "resolve_authority_service",
    "resolve_npub",
    "resolve_oracle_service",
    "resolve_relays",
    "resolve_route",
    "resolve_service_by_name",
    # Identity Credential
    "sign_identity_credential",
    "situation_response_from_row",
    "upstream_payment_situation",
    "validate_operator_tools",
    "validate_step",
    "verify_certificate_auto",
    "verify_credential_chain",
    "verify_identity_credential",
    "verify_nostr_certificate",
    # Operator Proof & Npub Ownership
    "verify_proof",
    "web_fetch_tool",
    "web_search_tool",
    "x402_wallet_template",
]
