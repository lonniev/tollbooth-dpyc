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

    def __init__(self, runtime: "OperatorRuntime") -> None:
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

    def __init__(self, *, deployment: str, api_url: str, api_key: str) -> None:
        # deployment e.g. "dpyc-job-flow/dpyc-jobs". api_url/api_key target the
        # standalone Prefect Cloud account; held in memory only.
        self._deployment = deployment
        self._api_url = api_url
        self._api_key = api_key

    def _client_ctx(self) -> Any:
        import os

        from prefect.client.orchestration import get_client

        # The MCP front carries no PREFECT_* env (creds live in the vault), so
        # point the client at the standalone account explicitly.
        os.environ.setdefault("PREFECT_API_URL", self._api_url)
        os.environ.setdefault("PREFECT_API_KEY", self._api_key)
        return get_client()

    async def submit(self, claim: str, closure_b64: str | None) -> str:
        from prefect.deployments import run_deployment

        # timeout=0 ⇒ fire-and-return: create + schedule the run, do NOT wait.
        # Only the ciphertext closure crosses — no secrets, no params, no code.
        flow_run = await run_deployment(
            name=self._deployment,
            parameters={"closure_b64": closure_b64},
            timeout=0,
        )
        return str(flow_run.id)

    async def poll(self, handle: str) -> dict[str, Any] | None:
        import json
        import uuid as _uuid

        from prefect.client.schemas.filters import (
            ArtifactFilter,
            ArtifactFilterFlowRunId,
        )

        run_id = _uuid.UUID(handle)
        async with self._client_ctx() as client:
            run = await client.read_flow_run(run_id)
            state = run.state
            if state is None or not state.is_final():
                return None
            if state.is_completed():
                # The flow publishes its result as a Prefect Artifact (Prefect
                # result storage defaults to the worker's local disk, which the
                # MCP cannot read). Read it back by flow-run id.
                arts = await client.read_artifacts(
                    artifact_filter=ArtifactFilter(
                        flow_run_id=ArtifactFilterFlowRunId(any_=[run_id])
                    ),
                    limit=1,
                )
                result: Any = None
                if arts:
                    data = arts[0].data
                    result = json.loads(data) if isinstance(data, str) else data
                return {"status": "completed", "result": result, "error": None}
            # failed / crashed / cancelled — report only the state type, never
            # the upstream body (it could echo a sealed value).
            return {"status": "failed", "result": None, "error": str(state.type)}
