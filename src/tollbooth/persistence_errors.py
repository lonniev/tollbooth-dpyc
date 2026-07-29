"""Naming what went wrong when persistence refuses.

A failed read against the operator's database is not one situation but three,
and they call for opposite responses: a cold connection clears on its own, a
quota ceiling clears only when someone pays, and a permission fault clears only
when someone repairs the schema. Telling a caller "retry shortly" for the second
two is worse than saying nothing — it sends them to wait out an outage that will
never end.

The distinction used to live only in ``PricingResolver``'s error properties, read
by one call site. The credential vault needs exactly the same reading, so the
classification moves here where both can share it — the peer of
``llm_route.classify_llm_failure`` and ``upstream_payment.classify_upstream_payment``.

Every situation returned here is a member of ``ErrorCode``, so a caller can hand
it straight to the situation tables that already carry finished copy for it.
"""

from __future__ import annotations

# Postgres SQLSTATE classes that will never resolve by retrying — the
# operator's database needs repair, not patience. Class 28 = invalid
# authorization, class 3D = invalid catalog, class 42 = syntax errors and
# access-rule violations (42501 permission denied, 42P01 undefined table).
_PERMANENT_SQLSTATE_CLASSES = ("28", "3D", "42")


def is_permanent_sql_error(exc: Exception) -> bool:
    """True when *exc* is a NeonQueryError carrying a permanent SQLSTATE."""
    code = getattr(exc, "code", "")
    return bool(code) and code[:2] in _PERMANENT_SQLSTATE_CLASSES


def is_quota_error(exc: Exception) -> bool:
    """True when *exc* is the persistence provider (Neon) refusing with HTTP
    402 — the database has exhausted its compute/storage quota. Non-transient
    and distinct from a SQL fault: the books are locked for billing, not code."""
    return getattr(exc, "status", 0) == 402


def classify_persistence_failure(exc: Exception) -> str:
    """Name the situation behind a failed persistence read.

    Returns an ``ErrorCode`` value:

    * ``persistence_quota_exceeded`` — the provider refused at the gateway;
      retrying cannot help and the Authority must restore capacity.
    * ``persistence_misconfigured`` — a permanent SQLSTATE; the operator must
      repair the database.
    * ``vault_bootstrapping`` — anything else, which in practice means a cold or
      briefly unreachable connection. This is the only one a caller should be
      told to wait out.

    The default is deliberately the *optimistic* one: an unrecognized transport
    error is far more often a cold start than a permanent fault, and calling it
    transient costs a caller one wasted retry, while calling a cold start
    permanent strands them.
    """
    from tollbooth.constants import ErrorCode

    if is_quota_error(exc):
        return ErrorCode.PERSISTENCE_QUOTA_EXCEEDED
    if is_permanent_sql_error(exc):
        return ErrorCode.PERSISTENCE_MISCONFIGURED
    return ErrorCode.VAULT_BOOTSTRAPPING
