"""Generic detached job flow for DPYC durable async execution.

Runs on a Prefect **Managed** work pool (its own EC2 compute), triggered by
``PrefectClosureExecutor.submit`` in ``tollbooth.async_executor``. It receives
ONE parameter — an AES-256-GCM-sealed, base64 *closure* (a self-describing job
spec) — decrypts it with a symmetric key held as a Prefect Secret block,
executes the declarative spec, and returns the result.

Design guarantees:

- **Generic, not domain-specific.** It knows a small set of op *primitives*
  (``http_request``, ``plan/v1``) and zero named jobs. The closure says what to do.
- **No executable code is ever received** — the closure is pure data. New
  capabilities are new ops added *here* (versioned in git), never shipped over
  the wire. There is no ``eval``/``pickle`` surface.
- **No Neon, no nsec.** This flow never touches the operator's database or key.
  The triggering MCP retrieves this flow's return value via the Prefect API and
  persists it into its own Neon job row.
- **Secrets stay sealed.** The Anthropic key (or any secret the MCP baked into
  the request) reaches Prefect only inside the encrypted closure; this flow
  decrypts it in memory and never logs it (``log_prints=False``; nothing here
  prints the spec, the headers, or the ciphertext).
- **Plans emit one artifact per stage** so the wheel can harvest partial
  progress and resume (#181). Stages are an internal reliability mechanism —
  never a billing boundary.

Deploy (from the public wheel repo, so Managed can clone the code)::

    flow.from_source(
        source="https://github.com/lonniev/tollbooth-dpyc.git",
        entrypoint="flows/dpyc_job_flow.py:dpyc_job_flow",
    ).deploy(name="dpyc-jobs", work_pool_name="<managed-pool>")

One Prefect Secret block per operator, named ``dpyc-closure-key-<key_id>``, must
hold the 64-hex symmetric key that operator seals with (its ``closure_seal_key``
vault entry). ``key_id`` is ``OperatorRuntime.durable_key_id()`` — a non-secret
hash of the operator npub — and travels in the cleartext run parameters.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from tollbooth.vault_encryption import VaultCipher

_CLOSURE_AAD = "dpyc-closure/v1"
_CLOSURE_KEY_PREFIX = "dpyc-closure-key"
_DEFAULT_TIMEOUT = 210.0
_PLAN_OP = "plan/v1"
_HTTP_OP = "http_request"


def _open_closure(closure_b64: str, key_id: str) -> dict[str, Any]:
    """Decrypt the sealed closure into a job spec. Raises on tamper/wrong key.

    ``key_id`` selects this operator's Secret block ``dpyc-closure-key-<key_id>``
    — each operator holds its own closure key, so the single shared flow serves
    them all without any operator's key being able to open another's closure.
    """
    from prefect.blocks.system import Secret

    block = f"{_CLOSURE_KEY_PREFIX}-{key_id}"
    key_hex = Secret.load(block).get()
    spec = json.loads(VaultCipher(key_hex).decrypt(closure_b64, aad=_CLOSURE_AAD))
    if not isinstance(spec, dict):
        raise TypeError("closure did not decrypt to a job spec")
    return spec


def _do_http_request(req: dict[str, Any]) -> dict[str, Any]:
    """The one op primitive: perform a described HTTP call, return the response.

    The output never echoes the request headers (which carry auth), only the
    response status and body — the MCP shapes it from there.
    """
    method = str(req.get("method", "POST")).upper()
    url = req["url"]
    with httpx.Client(timeout=req.get("timeout", _DEFAULT_TIMEOUT)) as client:
        resp = client.request(
            method, url, headers=req.get("headers"), json=req.get("json")
        )
    # Return the response for EVERY status — including non-2xx. The generic flow
    # is a faithful messenger: deciding whether a status is a success or a failure
    # is DOMAIN policy, which belongs in the triggering MCP's shape_result (it can
    # classify the upstream error into a curated, frontend-facing situation). The
    # status + body reach the MCP via the artifact (operator-only). We also log a
    # non-2xx here so the operator sees the reason in the Prefect run logs; the
    # body is the upstream's own error, not the request's auth headers. (Genuine
    # transport errors — connection, timeout — still raise from httpx above and
    # FAIL the run, which the MCP refunds generically.)
    if resp.is_error:
        try:
            from prefect import get_run_logger

            get_run_logger().warning(
                "upstream %s %s for %s %s: %s",
                resp.status_code, resp.reason_phrase, method, url, resp.text[:2000],
            )
        except Exception:  # noqa: BLE001, S110 — logging must never break the messenger
            pass
    out: dict[str, Any] = {"status": resp.status_code}
    try:
        out["json"] = resp.json()
    except ValueError:
        out["text"] = resp.text
    return out


def create_markdown_artifact(*, markdown: str, key: str | None = None, **_kw: Any) -> None:
    """Thin wrapper around Prefect's artifact publisher — patchable in unit tests.

    Imported lazily so the pure plan/http helpers stay importable without the
    ``[prefect]`` extra. No-op when prefect is absent.
    """
    try:
        from prefect.artifacts import create_markdown_artifact as _prefect_create
    except ImportError:
        return
    _prefect_create(markdown=markdown, key=key)


def _publish_stage_artifact(stage_id: str, result: dict[str, Any]) -> None:
    """Publish one artifact per stage so the wheel can harvest partial progress.

    Keyed ``stage-<id>`` so ``PrefectClosureExecutor.poll`` can order and count
    them while the run is still open (#181). Best-effort outside a flow context
    (unit tests call the pure dispatcher without Prefect installed).
    """
    try:
        create_markdown_artifact(
            markdown=json.dumps(result),
            key=f"stage-{stage_id}" if stage_id else None,
        )
    except Exception as exc:  # noqa: BLE001 — artifact is observability; run must still return
        # Best-effort only — never let artifact publish kill a completed stage.
        try:
            from prefect import get_run_logger

            get_run_logger().debug("stage artifact publish skipped: %s", exc)
        except Exception:  # noqa: BLE001, S110
            pass


def _do_plan_v1(
    spec: dict[str, Any],
    *,
    http: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a multi-stage plan: one HTTP request per stage, one artifact each.

    Stages are internal reliability only — never a billing boundary. The fare
    for the firing was already assessed once by the MCP; this flow never debits
    or refunds. ``budget_seconds`` (optional) is recorded for operator visibility;
    per-stage ``request.timeout`` overrides ``_DEFAULT_TIMEOUT``.

    Returns a structured plan result the MCP settles via ``shape_stage``.
    """
    do_http = http or _do_http_request
    stages = spec.get("stages") or []
    if not isinstance(stages, list) or not stages:
        raise ValueError("plan/v1 requires a non-empty stages list")

    completed: list[dict[str, Any]] = []
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise TypeError(f"plan/v1 stage {i} must be a dict")
        stage_id = str(stage.get("id") or f"stage-{i}")
        req = stage.get("request")
        if not isinstance(req, dict):
            raise TypeError(f"plan/v1 stage {stage_id!r} requires a request dict")
        # Honour per-stage timeout from the operator's spec (#181 open Q #2).
        req = dict(req)
        if "timeout" not in req:
            req["timeout"] = _DEFAULT_TIMEOUT
        result = do_http(req)
        _publish_stage_artifact(stage_id, result)
        completed.append({"id": stage_id, "result": result})

    return {
        "op": _PLAN_OP,
        "stages": completed,
        "completed": len(completed),
        "total": len(stages),
        "budget_seconds": spec.get("budget_seconds"),
    }


def _dispatch_op(spec: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a decrypted job spec to its op primitive. Pure (no Prefect)."""
    op = spec.get("op")
    if op == _HTTP_OP:
        return _do_http_request(spec["request"])
    if op == _PLAN_OP:
        return _do_plan_v1(spec)
    raise ValueError(f"unknown closure op: {op!r}")


def _run_job(closure_b64: str, key_id: str) -> dict[str, Any]:
    """Open, dispatch, publish final artifact. Shared by the @flow wrapper."""
    from prefect.artifacts import create_markdown_artifact

    spec = _open_closure(closure_b64, key_id)
    result = _dispatch_op(spec)
    # Final aggregate artifact (http_request path + plan summary). Per-stage
    # artifacts were already published inside _do_plan_v1.
    create_markdown_artifact(markdown=json.dumps(result), key="result")
    return result


try:
    from prefect import flow as _prefect_flow
except ImportError:  # plain wheel / unit tests — pure helpers stay importable
    def dpyc_job_flow(closure_b64: str, key_id: str) -> dict[str, Any]:
        """Unavailable without the ``[prefect]`` extra — pure helpers still work."""
        raise ImportError(
            "dpyc_job_flow requires the 'prefect' extra "
            "(pin tollbooth-dpyc[...,prefect])"
        )
else:

    @_prefect_flow(name="dpyc-job-flow", retries=0, log_prints=False)
    def dpyc_job_flow(closure_b64: str, key_id: str) -> dict[str, Any]:
        """Open a sealed closure, dispatch its op, publish the result as an artifact.

        ``key_id`` is the non-secret per-operator selector for the closure key
        (names the ``dpyc-closure-key-<key_id>`` Secret block); the closure body
        stays sealed until the matching key opens it.

        The result travels back to the triggering MCP via a **Prefect Artifact**
        (stored in Prefect Cloud, auto-associated with this flow run, retrievable
        with the MCP's existing API key) — NOT via Prefect result storage, whose
        default is the worker's local disk and so is unreadable from another host.
        The artifact body is the JSON result; it carries the upstream *response*
        only, never the request's auth headers.

        For ``plan/v1``, one artifact is also published *per stage* (keyed
        ``stage-<id>``) so the wheel can harvest partial progress while the run is
        still open; the final artifact is the full plan result.

        ``retries=0`` deliberately: the MCP's claim-check layer owns retry/refund
        semantics (a fresh ``start_async_job`` is the retry), and ``http_request``
        against a non-idempotent POST must not be auto-replayed.
        """
        return _run_job(closure_b64, key_id)
