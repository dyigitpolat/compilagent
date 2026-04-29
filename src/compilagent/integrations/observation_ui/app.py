"""FastAPI app for the compilagent observation UI.

Read-only viewer for the on-disk `TraceStore`, plus thin write endpoints
(`POST /api/runs/workload`) that drive an `OptimizationSession` in a
background task. Backend-specific artifact suffixes are routed through the
core's `ArtifactRendererRegistry`; the UI never branches on backend identity.

The SPA is no-build vanilla JS at `static/index.html`. Every endpoint here
corresponds to one fetch / WebSocket call in `static/app.js`.

NOTE: this module deliberately does NOT use `from __future__ import
annotations`. FastAPI's WebSocket-parameter detection inspects the raw
annotation object on the route function — under PEP 563 the annotation is
a string and FastAPI can't resolve `WebSocket`, so it demotes the parameter
to a query field. The route then rejects every connection with
`1008 / Field required` (uvicorn logs that as `WebSocket /ws ... 403`).
"""

import asyncio
import json
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from compilagent.bootstrap import (
    get_recent_load_failures,
    load_entry_point_integrations,
)
from compilagent.core.backend import backend_registry
from compilagent.core.workload_registry import workload_registry
from compilagent.harness.base import HarnessRunRequest
from compilagent.harness.registry import harness_registry
from compilagent.session.session import OptimizationSession, run_session
from compilagent.settings import CompilagentSettings
from compilagent.storage.trace_store import TraceStore
from compilagent.storage.workspace import OptimizationWorkspace

from .previews import render_preview, safe_resolve

_STATIC_DIR = Path(__file__).parent / "static"

_GPU_QUERY_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)


# --------------------------------------------------------------------- helpers


def _harness_extra(settings: CompilagentSettings) -> dict[str, Any]:
    out: dict[str, Any] = dict(settings.harness_extra or {})
    if settings.anthropic_api_key is not None:
        out.setdefault(
            "anthropic_api_key", settings.anthropic_api_key.get_secret_value()
        )
    if settings.mistral_api_key is not None:
        out.setdefault(
            "mistral_api_key", settings.mistral_api_key.get_secret_value()
        )
    if settings.openai_api_key is not None:
        out.setdefault(
            "openai_api_key", settings.openai_api_key.get_secret_value()
        )
    return out


def _query_gpu_telemetry(timeout_s: float = 2.0) -> list[dict[str, Any]]:
    """Run nvidia-smi once; return per-GPU dicts. Empty list when unavailable."""

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_GPU_QUERY_FIELDS):
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu_pct": _to_float(parts[2]),
                    "memory_used_mib": _to_float(parts[3]),
                    "memory_total_mib": _to_float(parts[4]),
                    "temperature_c": _to_float(parts[5]),
                    "power_w": _to_float(parts[6]),
                }
            )
        except (ValueError, IndexError):
            continue
    return rows


def _to_float(value: str) -> float | None:
    if value in ("", "[N/A]", "N/A", "[Not Supported]"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _summarise_runs(events: list) -> list[dict[str, Any]]:
    """Group `session.started` / `session.finished` / `session.failed` by run_id."""

    runs: dict[str, dict[str, Any]] = {}
    for ev in events:
        run_id = ev.run_id
        if not run_id:
            continue
        kind = ev.kind
        payload = ev.payload or {}
        row = runs.setdefault(
            run_id,
            {
                "run_id": run_id,
                "workload_id": None,
                "backend_id": None,
                "harness": None,
                "started_at": None,
                "finished_at": None,
                "status": "running",
                "best_speedup": None,
                "successful_count": None,
            },
        )
        if kind == "session.started":
            row["started_at"] = ev.timestamp
            row["workload_id"] = payload.get("workload_id") or row["workload_id"]
            row["backend_id"] = payload.get("backend_id") or row["backend_id"]
            row["harness"] = payload.get("harness") or row["harness"]
        elif kind == "session.finished":
            row["finished_at"] = ev.timestamp
            row["status"] = "finished"
            row["successful_count"] = payload.get("successful_count")
            leaderboard = payload.get("leaderboard") or []
            best = next(
                (
                    r
                    for r in leaderboard
                    if r.get("candidate_id") not in (None, "baseline")
                ),
                None,
            )
            if best:
                row["best_speedup"] = best.get("speedup_vs_baseline")
        elif kind == "session.failed":
            row["finished_at"] = ev.timestamp
            row["status"] = "failed"
    return sorted(
        runs.values(),
        key=lambda r: r.get("started_at") or 0.0,
        reverse=True,
    )


# ----------------------------------------------------------------------- app


def create_app(
    workspace_root: Path | None = None,
    *,
    settings: CompilagentSettings | None = None,
) -> Any:
    """Build the FastAPI app. Uvicorn imports this and runs it."""

    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import (
        FileResponse,
        JSONResponse,
        Response,
        StreamingResponse,
    )
    from fastapi.staticfiles import StaticFiles

    settings = settings or CompilagentSettings.from_env()
    cwd = Path.cwd()
    workspace_path = workspace_root or cwd
    workspace = OptimizationWorkspace(
        session_cwd=workspace_path,
        root_name=settings.workspace_dir_name,
    ).ensure()
    trace_store = TraceStore(workspace.root).ensure()

    # Bring entry-point-advertised integrations online once at startup.
    load_entry_point_integrations()
    startup_failures = get_recent_load_failures()

    app = FastAPI(title="Compilagent Observation UI")
    app.state.startup_errors = startup_failures
    app.state.boot_time = time.time()

    # ---- index + static ----------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        favicon_path = _STATIC_DIR / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        return Response(status_code=204)

    app.mount("/assets", StaticFiles(directory=_STATIC_DIR), name="assets")

    # ---- registries --------------------------------------------------------

    @app.get("/api/runtime/config")
    async def runtime_config() -> JSONResponse:
        return JSONResponse(
            {
                "settings": settings.public_metadata(),
                "backends": [
                    {
                        "id": bid,
                        "artifact_stages": list(
                            backend_registry.get(bid).artifact_stages
                        ),
                    }
                    for bid in backend_registry.ids()
                ],
                "harnesses": [
                    {
                        "id": hid,
                        "supported_providers": list(
                            getattr(harness_registry.get(hid), "supported_providers", ())
                        ),
                    }
                    for hid in harness_registry.ids()
                ],
                "workloads": [spec.serialize() for spec in workload_registry.specs()],
                "boot_time": app.state.boot_time,
            }
        )

    @app.get("/api/backends")
    async def list_backends() -> JSONResponse:
        return JSONResponse(
            {
                "backends": [
                    {
                        "id": bid,
                        "artifact_stages": list(
                            backend_registry.get(bid).artifact_stages
                        ),
                    }
                    for bid in backend_registry.ids()
                ]
            }
        )

    @app.get("/api/harnesses")
    async def list_harnesses() -> JSONResponse:
        return JSONResponse(
            {
                "harnesses": [
                    {
                        "id": hid,
                        "supported_providers": list(
                            getattr(harness_registry.get(hid), "supported_providers", ())
                        ),
                    }
                    for hid in harness_registry.ids()
                ]
            }
        )

    @app.get("/api/workloads")
    async def list_workloads() -> JSONResponse:
        return JSONResponse(
            {
                "workloads": [
                    spec.serialize() for spec in workload_registry.specs()
                ]
            }
        )

    @app.get("/api/workloads/diagnostics")
    async def workload_diagnostics() -> JSONResponse:
        return JSONResponse({"startup_errors": list(app.state.startup_errors)})

    @app.get("/api/workloads/{workload_id}/source")
    async def workload_source(workload_id: str) -> JSONResponse:
        if workload_id not in workload_registry.ids():
            raise HTTPException(404, f"workload `{workload_id}` is not registered")
        try:
            payload = workload_registry.get_builder_source(workload_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"failed to read workload source: {exc!r}") from exc
        return JSONResponse(payload)

    # ---- runs --------------------------------------------------------------

    @app.post("/api/runs/workload")
    async def start_workload_run(payload: dict) -> JSONResponse:
        workload_id = str(payload.get("workload_id", "")).strip()
        if not workload_id:
            raise HTTPException(400, "workload_id is required")
        if workload_id not in workload_registry.ids():
            raise HTTPException(404, f"workload `{workload_id}` is not registered")
        harness_id = str(payload.get("harness", settings.harness)).strip()
        if harness_id not in harness_registry.ids():
            raise HTTPException(404, f"harness `{harness_id}` is not registered")
        max_candidates = int(payload.get("max_candidates", settings.max_candidates))
        run_id = f"ui-{uuid.uuid4().hex[:10]}"
        model_id = str(payload.get("model_id") or settings.model_name)
        user_prompt = str(
            payload.get("user_prompt")
            or payload.get("prompt")
            or "Inspect, propose 3 candidates, run them, synthesize findings."
        )

        async def _drive() -> None:
            session = OptimizationSession(
                workload_id=workload_id,
                run_id=run_id,
                workspace=workspace,
                sink=trace_store,
                max_candidates=max_candidates,
            )
            request = HarnessRunRequest(
                toolset=session.toolset,
                system_instructions=(
                    f"Optimize workload `{workload_id}` from the observation UI."
                ),
                user_prompt=user_prompt,
                model_id=model_id,
                reasoning_effort=settings.reasoning_effort,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                max_turns=int(settings.harness_extra.get("max_turns", 24)),
                extra={**_harness_extra(settings), "cwd": str(workspace.session_cwd)},
            )
            harness = harness_registry.get(harness_id)
            try:
                await run_session(session=session, harness=harness, request=request)
            except Exception as exc:  # noqa: BLE001
                trace_store.emit_kv(
                    "session.failed",
                    payload={"error_type": type(exc).__name__, "message": str(exc)},
                    run_id=run_id,
                )
            finally:
                session.finalize()

        asyncio.create_task(_drive())
        return JSONResponse(
            {
                "run_id": run_id,
                "harness": harness_id,
                "workload_id": workload_id,
                "model_id": model_id,
                "max_candidates": max_candidates,
            }
        )

    @app.get("/api/runs")
    async def list_runs() -> JSONResponse:
        events = trace_store.read_events(
            kinds=["session.started", "session.finished", "session.failed"]
        )
        return JSONResponse({"runs": _summarise_runs(events)})

    @app.get("/api/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        kinds: str | None = None,
        limit: int | None = None,
    ) -> JSONResponse:
        kind_filter = kinds.split(",") if kinds else None
        events = trace_store.read_events(
            run_id=run_id, kinds=kind_filter, limit=limit
        )
        return JSONResponse({"events": [e.serialize() for e in events]})

    @app.get("/api/runs/{run_id}/leaderboard")
    async def run_leaderboard(run_id: str) -> JSONResponse:
        events = trace_store.read_events(run_id=run_id, kinds=["leaderboard.updated"])
        if not events:
            return JSONResponse({"run_id": run_id, "rows": []})
        latest = events[-1]
        return JSONResponse(
            {
                "run_id": run_id,
                "timestamp": latest.timestamp,
                "rows": (latest.payload or {}).get("rows", []),
            }
        )

    # ---- artifacts ---------------------------------------------------------

    @app.get("/api/artifacts/preview/{path:path}")
    async def artifact_preview(path: str, max_chars: int = 80_000) -> JSONResponse:
        return JSONResponse(render_preview(workspace, path, max_chars=max_chars))

    @app.get("/api/artifacts/{path:path}")
    async def artifact_download(path: str) -> Response:
        try:
            resolved = safe_resolve(workspace, path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(404, f"artifact not found: {path}")
        return FileResponse(resolved)

    # ---- telemetry ---------------------------------------------------------

    @app.get("/api/telemetry/gpu")
    async def gpu_telemetry() -> JSONResponse:
        return JSONResponse(
            {"gpus": _query_gpu_telemetry(), "fetched_at": time.time()}
        )

    # ---- live event stream -------------------------------------------------

    @app.websocket("/ws")
    async def ws_events(websocket: WebSocket) -> None:
        await websocket.accept()
        live = websocket.query_params.get("live", "0") == "1"
        run_filter = websocket.query_params.get("run_id") or None
        cursor = len(trace_store.read_events()) if live else 0
        try:
            while True:
                events = trace_store.read_events()
                if cursor < len(events):
                    for ev in events[cursor:]:
                        if run_filter and ev.run_id != run_filter:
                            continue
                        await websocket.send_text(
                            json.dumps(ev.serialize(), default=str)
                        )
                    cursor = len(events)
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    @app.get("/api/stream")
    async def sse_events(
        live: int = 0,
        run_id: str | None = None,
    ) -> StreamingResponse:
        async def _iter() -> AsyncIterator[bytes]:
            cursor = len(trace_store.read_events()) if live else 0
            while True:
                events = trace_store.read_events()
                if cursor < len(events):
                    for ev in events[cursor:]:
                        if run_id and ev.run_id != run_id:
                            continue
                        line = json.dumps(ev.serialize(), default=str)
                        yield f"data: {line}\n\n".encode()
                    cursor = len(events)
                await asyncio.sleep(0.5)

        return StreamingResponse(_iter(), media_type="text/event-stream")

    return app
