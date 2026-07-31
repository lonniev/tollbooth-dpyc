"""What went wrong restoring an OAuth session, and the facts that say so.

A situation used to be a bare string — ``"token_expired"`` — travelling out of
``OperatorRuntime.restore_oauth_session``. The string is a verdict with its
evidence stripped, and the stripping happened at exactly the wrong moment:
``refresh_access_token`` is the only code in the fleet that sees the provider's
HTTP status and error body, and it handed forward six characters of conclusion.

Downstream, a scheduled post held for ``oauth_token_expired`` was
indistinguishable from one held because an operator rotated an app secret,
because a refresh timed out and was never classified, or because the vault held
no refresh token to spend. eXcalibur's operator reconnected X on that advice for
days while the real cause — a refresh answer lost in flight — went unnamed.

So a situation carries its evidence now. ``code`` is still the routing key and
still maps 1:1 to a patron-facing recipe in ``oauth_situation_response``;
``detail`` and the provider fields are what make the log worth reading.

Nothing here is secret: ``detail`` arrives already redacted of the refresh token
(see ``oauth2_collector._redact``), and no field ever holds a credential.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OAuthSituation:
    """One reason a session could not be restored, with what we know about it.

    Args:
        code: The routing key — ``"token_expired"``, ``"refresh_token_lost"``,
            a persistence situation forwarded from the vault, and so on. This
            is what ``oauth_situation_response`` maps to a recipe, and what a
            calling agent branches on.
        detail: Free-form, human-readable, already-redacted facts. The
            provider's own words where we have them; an exception type where we
            don't. Safe to log and safe to show.
        status_code: Upstream HTTP status, when an upstream answered at all.
        oauth_error: The RFC 6749 §5.2 error code the provider named, when it
            named one.
        observed_at: ISO-8601 UTC instant the fact was observed. Set when the
            situation refers to something that happened *earlier* than this
            call — a lost refresh, say — so the report can cite the moment.
    """

    code: str
    detail: str = ""
    status_code: int = 0
    oauth_error: str = ""
    observed_at: str = ""

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("an OAuthSituation must have a code")

    def with_detail(self, detail: str) -> OAuthSituation:
        """A copy carrying *detail*, for enriching a situation in flight."""
        return OAuthSituation(
            code=self.code, detail=detail, status_code=self.status_code,
            oauth_error=self.oauth_error, observed_at=self.observed_at,
        )

    def as_dict(self) -> dict[str, object]:
        """The non-empty fields, for merging into a tool response."""
        out: dict[str, object] = {"error_code": self.code}
        if self.detail:
            out["detail"] = self.detail
        if self.status_code:
            out["upstream_status"] = self.status_code
        if self.oauth_error:
            out["upstream_oauth_error"] = self.oauth_error
        if self.observed_at:
            out["observed_at"] = self.observed_at
        return out
