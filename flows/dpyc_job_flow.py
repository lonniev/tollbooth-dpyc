"""Generic detached job flow for DPYC durable async execution.

Runs on a Prefect **Managed** work pool (its own EC2 compute), triggered by
``PrefectClosureExecutor.submit`` in ``tollbooth.async_executor``. It receives
ONE parameter — an AES-256-GCM-sealed, base64 *closure* (a self-describing job
spec) — decrypts it with a symmetric key held as a Prefect Secret block,
executes the declarative spec, and returns the result.

Design guarantees:

- **Generic, not domain-specific.** It knows a small set of op *primitives*
  (currently ``http_request``) and zero named jobs. The closure says what to do.
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

Deploy (from the public wheel repo, so Managed can clone the code)::

    flow.from_source(
        source="https://github.com/lonniev/tollbooth-dpyc.git",
        entrypoint="flows/dpyc_job_flow.py:dpyc_job_flow",
    ).deploy(name="dpyc-jobs", work_pool_name="<managed-pool>")

A Prefect Secret block named ``dpyc-closure-key`` must hold the 64-hex symmetric
key the MCP seals with (``closure_seal_key`` in the operator vault).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from prefect import flow
from prefect.artifacts import create_markdown_artifact
from prefect.blocks.system import Secret

from tollbooth.vault_encryption import VaultCipher

_CLOSURE_AAD = "dpyc-closure/v1"
_CLOSURE_KEY_BLOCK = "dpyc-closure-key"
_DEFAULT_TIMEOUT = 210.0


def _open_closure(closure_b64: str) -> dict[str, Any]:
    """Decrypt the sealed closure into a job spec. Raises on tamper/wrong key."""
    key_hex = Secret.load(_CLOSURE_KEY_BLOCK).get()
    spec = json.loads(VaultCipher(key_hex).decrypt(closure_b64, aad=_CLOSURE_AAD))
    if not isinstance(spec, dict):
        raise ValueError("closure did not decrypt to a job spec")
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
    out: dict[str, Any] = {"status": resp.status_code}
    try:
        out["json"] = resp.json()
    except ValueError:
        out["text"] = resp.text
    return out


@flow(name="dpyc-job-flow", retries=0, log_prints=False)
def dpyc_job_flow(closure_b64: str) -> dict[str, Any]:
    """Open a sealed closure, dispatch its op, publish the result as an artifact.

    The result travels back to the triggering MCP via a **Prefect Artifact**
    (stored in Prefect Cloud, auto-associated with this flow run, retrievable
    with the MCP's existing API key) — NOT via Prefect result storage, whose
    default is the worker's local disk and so is unreadable from another host.
    The artifact body is the JSON result; it carries the upstream *response*
    only, never the request's auth headers.

    ``retries=0`` deliberately: the MCP's claim-check layer owns retry/refund
    semantics (a fresh ``start_async_job`` is the retry), and ``http_request``
    against a non-idempotent POST must not be auto-replayed.
    """
    spec = _open_closure(closure_b64)
    op = spec.get("op")
    if op == "http_request":
        result = _do_http_request(spec["request"])
    else:
        raise ValueError(f"unknown closure op: {op!r}")
    create_markdown_artifact(markdown=json.dumps(result))
    return result
