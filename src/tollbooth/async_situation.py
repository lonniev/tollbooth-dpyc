"""Structured failure situations for async (claim-check) jobs.

A job runner (in-process path) or a spec's ``shape_result`` (closure/detached
path) raises :class:`AsyncJobSituation` to report a failure that the CALLING
FRONTEND should be able to render as informative UX — e.g. "the operator's AI
provider is out of credit, no fare was charged, try again shortly" — while
keeping the raw upstream error operator-side (Prefect logs).

Two audiences, deliberately separated:

- The **operator** sees the raw upstream status + body in the detached flow's
  Prefect logs (and the wheel's own ``logger`` for the in-process path).
- The **frontend / DPYC patron** sees only the curated fields carried here:
  a machine ``error_code`` (to branch UX), a safe human ``message``, optional
  ``next_steps``, and a ``transient`` flag (so a UI can offer retry and a
  scheduler can stop hammering a non-recoverable endpoint).

``OperatorRuntime.fetch_async_job`` / ``_run_job`` surface these fields and
persist them on the job row; they NEVER forward a raw exception string to the
patron. This follows the SDK's "situations, not failures" convention (cf.
``tollbooth.upstream_payment``).
"""

from __future__ import annotations

import json
from typing import Any

_ROW_KIND = "async_job_situation"
_DEFAULT_NEXT = "The fee was refunded. Start a new request to retry."


class AsyncJobSituation(Exception):
    """A curated, frontend-facing async-job failure.

    Raise from a job runner or ``shape_result``. ``error_code`` is a stable
    machine token for UX branching; ``message`` is safe human copy (no raw
    upstream body); ``transient`` says whether a retry could succeed.
    """

    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        next_steps: str = "",
        transient: bool = False,
    ) -> None:
        self.error_code = error_code
        self.message = message
        self.next_steps = next_steps
        self.transient = transient
        super().__init__(message)

    def to_row(self) -> str:
        """Serialize for the job row's ``error`` column (so the situation
        survives across polls — see :func:`situation_response_from_row`)."""
        return json.dumps({
            "kind": _ROW_KIND,
            "error_code": self.error_code,
            "message": self.message,
            "next_steps": self.next_steps,
            "transient": self.transient,
        })

    def to_response(self) -> dict[str, Any]:
        """The patron/frontend response for the settling fetch."""
        return {
            "success": True,
            "status": "error",
            "error_code": self.error_code,
            "error": self.message,
            "next_steps": self.next_steps or _DEFAULT_NEXT,
            "transient": self.transient,
            "refunded": True,
        }


def situation_response_from_row(error_text: str) -> dict[str, Any]:
    """Rebuild a patron/frontend response from a stored row ``error`` value.

    If the row holds a serialized :class:`AsyncJobSituation`, return its
    structured fields; otherwise treat the text as an already-safe plain
    message (back-compat with rows written before structured situations, and
    with the generic refund path which stores a generic string, never a raw
    exception).
    """
    base = {"success": True, "status": "error", "refunded": True}
    data: Any = None
    try:
        data = json.loads(error_text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict) and data.get("kind") == _ROW_KIND:
        return {
            **base,
            "error_code": data.get("error_code", "job_failed"),
            "error": data.get("message", ""),
            "next_steps": data.get("next_steps") or _DEFAULT_NEXT,
            "transient": bool(data.get("transient", False)),
        }
    return {**base, "error": error_text or "Job execution failed.", "next_steps": _DEFAULT_NEXT}
