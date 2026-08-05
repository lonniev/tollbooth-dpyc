"""Pluggable executors for claim-check async jobs.

``OperatorRuntime.start_async_job`` persists a job to Neon and then hands it to
an executor to actually run. Two executors ship:

- :class:`InProcessExecutor` (default) runs the job as a concurrent ``asyncio``
  task in the operator's own process — correct for genuinely long-lived hosts,
  but a job started this way dies if the host is a serverless front that
  freezes/recycles mid-run.
- :class:`PrefectClosureExecutor` dispatches the job to detached
  Prefect-managed compute. The runtime seals a self-describing *closure* (a job
  spec — see :meth:`OperatorRuntime.register_job_spec`) and triggers a generic
  Prefect flow that opens it, performs the work, and returns the result to
  Prefect. The runtime retrieves that result on the patron's next
  ``fetch_async_job`` poll. Only ciphertext crosses to Prefect; no Neon access,
  no domain logic, and no executable code travel to the flow.

``prefect`` is an OPTIONAL dependency (the ``[prefect]`` extra). It is imported
lazily inside :class:`PrefectClosureExecutor` so operators who never opt in do
not need it installed even though this module is always importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid a runtime import cycle with runtime.py
    from tollbooth.runtime import OperatorRuntime


@runtime_checkable
class JobExecutor(Protocol):
    """Submit a persisted claim for execution and later retrieve its outcome.

    ``submit`` returns an opaque handle. The in-process executor returns ``""``
    (the Neon row stays the single source of truth and ``poll`` is a no-op). A
    detached executor returns a handle (e.g. a Prefect flow-run id) that
    ``poll`` resolves to a terminal outcome.
    """

    async def submit(self, claim: str, closure_b64: str | None) -> str:
        """Dispatch the work for ``claim``. Return a handle ("" if none)."""
        ...

    async def poll(self, handle: str) -> dict[str, Any] | None:
        """Resolve a handle. ``None`` while still running; otherwise a dict
        ``{"status": "completed"|"failed", "result": Any, "error": str|None}``."""
        ...


class InProcessExecutor:
    """Run the job as a concurrent asyncio task in this process (today's path)."""

    def __init__(self, runtime: OperatorRuntime) -> None:
        self._runtime = runtime

    async def submit(self, claim: str, closure_b64: str | None) -> str:
        import asyncio

        # closure_b64 is unused in-process: the registered runner does the work
        # directly against the operator's own vault and Neon.
        asyncio.create_task(self._runtime._run_job(claim))
        return ""  # no detached handle — the Neon row is the source of truth

    async def poll(self, handle: str) -> dict[str, Any] | None:
        return None  # fetch_async_job reads the Neon row directly


class PrefectClosureExecutor:
    """Dispatch a sealed closure to a generic Prefect-managed flow.

    Constructed from credentials the operator holds in its OWN Neon vault (the
    Prefect API URL/key for the standalone account). nsec never reaches Prefect.
    """

    def __init__(
        self, *, deployment: str, api_url: str, api_key: str, key_id: str
    ) -> None:
        # deployment e.g. "dpyc-job-flow/dpyc-jobs". api_url/api_key target the
        # standalone Prefect Cloud account; held in memory only. key_id is the
        # non-secret selector for this operator's closure key — it names the
        # ``dpyc-closure-key-<key_id>`` Prefect Secret block and rides in the
        # cleartext envelope so the shared flow loads the right key.
        self._deployment = deployment
        self._api_url = api_url
        self._api_key = api_key
        self._key_id = key_id

    def _settings_ctx(self) -> Any:
        # Force THIS operator's standalone-account API URL/key for the duration
        # of one call, overriding any ambient PREFECT_* the host sets for its
        # OWN workspace. The MCP runs on a platform (e.g. Prefect Horizon) that
        # populates PREFECT_API_URL/KEY for its own account, so a plain
        # ``os.environ.setdefault`` is a no-op and the client would target the
        # wrong account (401, or deployment-not-found) — which would silently
        # drop us onto the in-process fallback. ``temporary_settings`` re-derives
        # the settings instead, and is contextvar-scoped so concurrent operators
        # don't clobber each other.
        from prefect.settings import (
            PREFECT_API_KEY,
            PREFECT_API_URL,
            temporary_settings,
        )

        return temporary_settings(
            updates={
                PREFECT_API_URL: self._api_url,
                PREFECT_API_KEY: self._api_key,
            }
        )

    async def submit(self, claim: str, closure_b64: str | None) -> str:
        # Fail fast on a missing seal. Prefect's deployment schema requires
        # ``closure_b64: string``; sending None costs a network hop and comes
        # back as a remote 409 ("None is not of type 'string'"), which
        # ``start_async_job`` then logs as a generic dispatch failure and
        # silently falls back in-process — the exact shape that made detached
        # resolve appear healthy while never dispatching (#178 / excalibur#341).
        if not isinstance(closure_b64, str) or not closure_b64:
            raise ValueError(
                "PrefectClosureExecutor.submit requires a non-empty sealed "
                f"closure_b64 string; got {closure_b64!r}. Register a job_spec "
                "and seal before submit, or run this kind in-process."
            )

        from prefect.deployments import run_deployment

        # timeout=0 ⇒ fire-and-return: create + schedule the run, do NOT wait.
        # Only the ciphertext closure + the non-secret key_id selector cross —
        # no secrets, no params, no code. The settings ctx points run_deployment
        # at the standalone account (NOT the host platform's own workspace).
        with self._settings_ctx():
            flow_run = await run_deployment(
                name=self._deployment,
                parameters={"closure_b64": closure_b64, "key_id": self._key_id},
                timeout=0,
            )
        return str(flow_run.id)

    async def poll(self, handle: str) -> dict[str, Any] | None:
        import uuid as _uuid

        from prefect.client.orchestration import get_client
        from prefect.client.schemas.filters import (
            ArtifactFilter,
            ArtifactFilterFlowRunId,
        )

        run_id = _uuid.UUID(handle)
        # Sync settings override wraps the async client (run_deployment/get_client
        # both read the current settings); temporary_settings is a sync CM.
        with self._settings_ctx():
            async with get_client() as client:
                run = await client.read_flow_run(run_id)
                state = run.state
                # Read ALL artifacts for this run (plan/v1 emits one per stage
                # plus a final aggregate). limit was 1 before #181, which hid
                # partial progress and multi-stage harvest.
                arts = await client.read_artifacts(
                    artifact_filter=ArtifactFilter(
                        flow_run_id=ArtifactFilterFlowRunId(any_=[run_id])
                    ),
                    limit=100,
                )
                harvested = _harvest_artifacts(arts)

                if state is None or not state.is_final():
                    # Still running — surface partial plan progress when any
                    # stage artifacts have landed so the wheel can report
                    # "k of n" without waiting for the final aggregate (#181).
                    if harvested is not None and harvested.get("op") == "plan/v1":
                        return {
                            "status": "working",
                            "result": harvested,
                            "error": None,
                        }
                    return None
                if state.is_completed():
                    return {
                        "status": "completed",
                        "result": harvested,
                        "error": None,
                    }
                # failed / crashed / cancelled — report only the state type, never
                # the upstream body (it could echo a sealed value). Attach any
                # partial plan harvest so the operator can console/fallback from
                # stages that did land (consolation ≠ fulfilment; fare is separate).
                return {
                    "status": "failed",
                    "result": harvested,
                    "error": str(state.type),
                }


def _harvest_artifacts(arts: list[Any]) -> Any:
    """Collapse Prefect artifacts for a flow run into a single result payload.

    ``plan/v1`` publishes one artifact per stage (key ``stage-<id>``) plus a
    final aggregate (key ``result``). Prefer the aggregate when present; else
    rebuild a plan-shaped payload from stage artifacts so a still-running or
    crashed plan still yields partial progress. A single-op ``http_request``
    run publishes one unkeyed/result artifact — return its body as before.
    """
    import json

    if not arts:
        return None

    def _parse(data: Any) -> Any:
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (ValueError, TypeError):
                return data
        return data

    final: Any = None
    stages: list[dict[str, Any]] = []
    singles: list[Any] = []

    for art in arts:
        key = getattr(art, "key", None) or ""
        body = _parse(getattr(art, "data", None))
        if key == "result" or (
            isinstance(body, dict) and body.get("op") == "plan/v1" and "stages" in body
        ):
            # Prefer an explicit final aggregate; a plan-shaped body also wins.
            if key == "result" or final is None:
                final = body
            continue
        if isinstance(key, str) and key.startswith("stage-"):
            stage_id = key[len("stage-") :]
            stages.append({"id": stage_id, "result": body})
            continue
        singles.append(body)

    if isinstance(final, dict) and final.get("op") == "plan/v1":
        return final
    if stages:
        return {
            "op": "plan/v1",
            "stages": stages,
            "completed": len(stages),
            "total": len(stages),  # unknown full total while still running
        }
    if final is not None:
        return final
    if singles:
        return singles[0]
    return _parse(arts[0].data) if arts else None
