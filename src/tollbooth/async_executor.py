"""Pluggable executors for claim-check async jobs.

``OperatorRuntime.start_async_job`` persists a job to Neon and then hands it to
an executor to actually run. Two executors ship:

- :class:`InProcessExecutor` (default) runs the job as a concurrent ``asyncio``
  task in the operator's own process. Correct for a genuinely long-lived host —
  it is what runs inside the MCP session on FastMCP Cloud — but a job started
  this way dies if that front freezes or recycles mid-run.
- :class:`ModalExecutor` spawns the SAME registered runner onto Modal, detached
  compute that outlives the request. Nothing is translated on the way: Modal
  runs the operator's own code, so there is no wire format, no interpreter and
  nothing to seal.

A sealed-closure executor (Prefect) preceded this and was deleted in 0.82.0.
It existed because a generic remote flow could not execute operator code, which
forced an entire apparatus into being: an AES-256-GCM closure, a non-secret
``key_id`` selector, a hand-created per-operator Secret block, an op vocabulary
(``http_request`` / ``plan/v1``), a generic interpreter flow cloned from git at
run time, and results smuggled back through Prefect *artifacts* because its
result storage is worker-local disk. Seven concepts to move one function call
off-box. Running the operator's own code deletes all seven rather than
replacing them.

``modal`` is an OPTIONAL dependency (the ``[modal]`` extra), imported lazily
inside :class:`ModalExecutor` so operators who never opt in do not need it
installed even though this module is always importable.
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
    detached executor returns a handle (e.g. a Modal call id) that ``poll``
    resolves to a terminal outcome.

    ``submit`` took a ``closure_b64`` argument until 0.82.0. Nothing seals any
    more, so the parameter is gone rather than accepted-and-ignored — a vestigial
    argument is an invitation to reintroduce the thing it served.
    """

    async def submit(self, claim: str) -> str:
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

    async def submit(self, claim: str) -> str:
        import asyncio

        # The registered runner does the work directly against the operator's
        # own vault and Neon — the same runner ModalExecutor spawns remotely.
        asyncio.create_task(self._runtime._run_job(claim))
        return ""  # no detached handle — the Neon row is the source of truth

    async def poll(self, handle: str) -> dict[str, Any] | None:
        return None  # fetch_async_job reads the Neon row directly


class ModalExecutor:
    """Dispatch a job to Modal — the operator's OWN runner, on detached compute.

    The whole point, and why it is three lines of real work: Modal runs *your
    code*, so there is nothing to translate. ``submit`` is
    :class:`InProcessExecutor` with ``spawn`` where ``asyncio.create_task`` was,
    and the registered ``register_job_runner`` function is unchanged.

    That deletes an apparatus rather than replacing it. The sealed closure, the
    ``key_id`` selector, the per-operator ``dpyc-closure-key-<key_id>`` Secret
    block, the op vocabulary (``http_request`` / ``plan/v1``), the generic
    interpreter flow and the artifact result channel all exist for one reason —
    the Prefect flow cannot execute operator code, so a wire format and an
    interpreter had to be invented. None of it is needed here.

    **Trust boundary, stated plainly.** The operator's Modal app runs the
    operator's own package with the operator's own secrets, so Modal holds what
    the MCP host already holds. That is a second host of the same class, not a
    new class of exposure — and it is why no seal is required: each operator
    deploys its own app, so nothing crosses a tenant boundary.

    ``modal`` is an OPTIONAL dependency (the ``[modal]`` extra), imported lazily
    so this module stays importable for operators who never opt in.
    """

    def __init__(self, *, app_name: str, function_name: str = "run_job") -> None:
        self._app_name = app_name
        self._function_name = function_name
        self._fn: Any = None

    def _function(self) -> Any:
        # Resolved once per process. `from_name` is a lookup against the
        # operator's deployed app; credentials come from the ambient Modal token
        # the runtime installed before constructing this executor.
        if self._fn is None:
            import modal

            self._fn = modal.Function.from_name(self._app_name, self._function_name)
        return self._fn

    async def submit(self, claim: str) -> str:
        """Spawn the job and return Modal's call id as the durable handle.

        Fire-and-return: ``spawn`` does not wait, so the tool response is not
        bound to the work. Measured at 0.22–0.34s against a 200s job.
        """
        import asyncio

        # `spawn` is sync and does network I/O; keep it off the event loop so a
        # slow control-plane call cannot stall the tool response it precedes.
        call = await asyncio.to_thread(self._function().spawn, claim)
        return str(call.object_id)

    async def poll(self, handle: str) -> dict[str, Any] | None:
        """Resolve a call id. ``None`` while still running.

        A cancelled or crashed run surfaces as a distinguishable terminal state
        rather than silence — Modal raises on ``get`` for both, and the message
        is the operator's, never a patron's.
        """
        import asyncio

        import modal

        def _resolve() -> dict[str, Any] | None:
            call = modal.FunctionCall.from_id(handle)
            try:
                # timeout=0 polls: return immediately if not yet finished.
                return {"status": "completed", "result": call.get(timeout=0), "error": None}
            except TimeoutError:
                return None
            except Exception as exc:  # noqa: BLE001 — cancelled / crashed / expired
                return {"status": "failed", "result": None,
                        "error": f"{type(exc).__name__}: {exc}"}

        return await asyncio.to_thread(_resolve)
