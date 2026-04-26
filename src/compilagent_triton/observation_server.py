from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent_runner import run_agent_optimization
from .episodes import EpisodeStore
from .events import ObservationEvent, redact
from .examples import (
    RunRequest,
    create_run_id,
    get_example,
    list_examples,
    preview_kernel_source,
    preview_source,
    run_registered_example,
)
from .settings import CompilagentSettings
from .trace_store import TraceStore
from .workspace import OptimizationWorkspace

UI_DIR = Path(__file__).with_name("observer_ui")


def create_app(*, workspace_root: Path | None = None) -> FastAPI:
    root = (workspace_root or (Path.cwd() / ".compilagent-triton")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    trace_store = TraceStore(root).ensure()
    workspace = OptimizationWorkspace(root.parent, root.name).ensure()
    app = FastAPI(title="Compilagent Triton Observer")
    app.state.workspace_root = root
    app.state.trace_store = trace_store
    app.state.workspace = workspace
    app.state.runs = {}
    app.state.active_run_id = None
    app.state.runtime_config = _runtime_config(root.parent)

    # Eagerly register backends + workloads so `/api/workloads` and the
    # backend selector are populated on first request. We log failures rather
    # than swallow them silently — that's why the user saw an empty dropdown
    # without any clue what went wrong.
    import logging as _logging

    app.state.startup_errors = []
    log = _logging.getLogger("compilagent.observer")
    try:
        from .backends import import_backend_packages  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        msg = f"backends package import failed: {exc!r}"
        log.warning(msg)
        app.state.startup_errors.append(msg)
    else:
        imported = import_backend_packages()
        if not imported:
            app.state.startup_errors.append("no backend subpackages registered")
    try:
        from .workloads.registry import import_workload_packages
        import_workload_packages()
    except Exception as exc:  # noqa: BLE001
        msg = f"workload registry import failed: {exc!r}"
        log.warning(msg)
        app.state.startup_errors.append(msg)

    if UI_DIR.exists():
        app.mount("/assets", StaticFiles(directory=UI_DIR), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        index_path = UI_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="observer UI is missing")
        return FileResponse(index_path)

    @app.get("/api/sessions")
    def sessions() -> dict[str, Any]:
        events = trace_store.read_events()
        session_ids = sorted({event.session_id for event in events if event.session_id})
        episode_ids = sorted({event.episode_id for event in events if event.episode_id})
        return {"sessions": session_ids, "episodes": episode_ids, "event_count": len(events)}

    @app.get("/api/events")
    def events(
        session_id: str | None = None,
        episode_id: str | None = None,
        after: str | None = None,
        limit: int = Query(default=500, ge=0, le=5000),
    ) -> dict[str, Any]:
        event_list = trace_store.read_events(
            session_id=session_id,
            episode_id=episode_id,
            after=after,
            limit=limit,
        )
        return {"events": [event.model_dump(mode="json") for event in event_list]}

    @app.get("/api/examples")
    def examples() -> dict[str, Any]:
        return {"examples": list_examples()}

    @app.get("/api/workloads/diagnostics")
    def workload_diagnostics() -> dict[str, Any]:
        """Surface startup-time registry errors so the UI can show them."""
        return {"startup_errors": list(getattr(app.state, "startup_errors", []))}

    @app.get("/api/workloads/{workload_id}/source")
    def workload_source(workload_id: str) -> dict[str, Any]:
        """Return the workload module's Python source — the "JIT source" view.

        For Triton kernel workloads this is the kernel module file. For
        PyTorch full_model workloads it's the builder module (which contains
        the `nn.Module` model construction code), which is what the agent and
        the user mean by "the model's JIT source" — the actual Python that's
        about to be handed to `torch.compile`.
        """

        import inspect as _inspect
        import importlib

        from .workloads.registry import workload_registry, import_workload_packages

        import_workload_packages()
        try:
            spec = workload_registry.get_spec(workload_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        module_path, _, fn_name = spec.entrypoint.partition(":")
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            return {
                "workload_id": workload_id, "source_kind": "missing",
                "language": "python",
                "warning": f"could not import workload module `{module_path}`: {exc!r}",
                "source": "",
            }
        try:
            file_path = _inspect.getsourcefile(module) or ""
            text = Path(file_path).read_text(encoding="utf-8") if file_path else _inspect.getsource(module)
        except Exception as exc:  # noqa: BLE001
            return {
                "workload_id": workload_id, "source_kind": "missing",
                "language": "python",
                "warning": f"could not read workload source: {exc!r}",
                "source": "",
            }
        return {
            "workload_id": workload_id,
            "source_kind": "workload",
            "language": "python",
            "source_path": file_path,
            "symbol": fn_name or None,
            "source": text,
            "line_count": text.count("\n") + (1 if text else 0),
        }

    @app.get("/api/workloads")
    def workloads() -> dict[str, Any]:
        """List workloads from the new backend-agnostic registry.

        Each entry includes the workload spec (typed shape/dtype/budget/tolerance)
        plus the active backend's artifact_stages so the UI can show the right
        Compiler-tab vocabulary per workload.
        """

        from .backends import backend_registry, import_backend_packages
        from .workloads.registry import import_workload_packages, workload_registry

        # Make sure self-registering modules ran. Idempotent.
        import_workload_packages()
        import_backend_packages()

        out: list[dict[str, Any]] = []
        for spec in workload_registry.specs():
            entry = spec.serialize()
            try:
                backend = backend_registry.get(spec.backend_id)
                entry["backend"] = {
                    "id": backend.id,
                    "artifact_stages": list(backend.artifact_stages),
                    "device": {
                        "arch": backend.device_capability().arch,
                        "name": backend.device_capability().name,
                    },
                }
            except KeyError:
                entry["backend"] = {"id": spec.backend_id, "available": False}
            out.append(entry)
        return {"workloads": out, "registered_backends": backend_registry.ids()}

    @app.get("/api/examples/{example_id}")
    def example(example_id: str, max_chars: int = Query(default=40_000, ge=1000, le=200_000)) -> dict[str, Any]:
        try:
            return preview_source(example_id, max_chars=max_chars)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/examples/{example_id}/kernel")
    def example_kernel(
        example_id: str,
        max_chars: int = Query(default=40_000, ge=1000, le=200_000),
    ) -> dict[str, Any]:
        try:
            return preview_kernel_source(example_id, max_chars=max_chars)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runtime/config")
    def runtime_config() -> dict[str, Any]:
        return dict(app.state.runtime_config)

    @app.post("/api/runtime/config")
    def update_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
        from .backends import backend_registry

        allowed = {
            "harness", "mode", "model", "backend", "workload_id", "max_candidates",
        }
        updates = {key: value for key, value in config.items() if key in allowed}
        if "max_candidates" in updates:
            try:
                n = int(updates["max_candidates"])
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail="max_candidates must be an integer >= 1"
                )
            if n < 1 or n > 32:
                raise HTTPException(
                    status_code=400, detail="max_candidates must be in [1, 32]"
                )
            updates["max_candidates"] = n
        if "harness" in updates and updates["harness"] not in {"pydantic_ai", "claude_agent_sdk"}:
            raise HTTPException(status_code=400, detail="unsupported harness")
        if "mode" in updates and updates["mode"] not in {"benchmark", "optimize"}:
            raise HTTPException(status_code=400, detail="unsupported mode")
        if "backend" in updates and updates["backend"] not in backend_registry.ids():
            raise HTTPException(
                status_code=400,
                detail=f"unknown backend; registered: {backend_registry.ids()}",
            )
        if "model" in updates:
            new_model = str(updates["model"])
            if ":" not in new_model:
                raise HTTPException(
                    status_code=400,
                    detail="model must be `<provider>:<name>` (e.g. mistral:mistral-large-latest)",
                )
            os.environ["COMPILAGENT_MODEL"] = new_model
        app.state.runtime_config = {**app.state.runtime_config, **updates}
        return dict(app.state.runtime_config)

    @app.post("/api/runs")
    def start_run(request: RunRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        try:
            example_spec = get_example(request.example_id)
            config = request.config.bounded_for(example_spec)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not example_spec.enabled:
            raise HTTPException(status_code=400, detail=example_spec.disabled_reason or "example disabled")
        if app.state.active_run_id is not None:
            raise HTTPException(status_code=409, detail="a run is already active")
        run_id = create_run_id(request.example_id)
        payload = {
            "run_id": run_id,
            "example_id": request.example_id,
            "family": example_spec.kernel_family,
            "mode": request.mode,
            "config": config.model_dump(mode="json"),
        }
        app.state.runs[run_id] = {"status": "queued", **payload}
        app.state.active_run_id = run_id
        trace_store.emit("run.requested", payload=payload)
        background_tasks.add_task(_run_example_task, app, request.model_copy(update={"config": config}), run_id)
        return {"run_id": run_id, "status": "queued"}

    @app.post("/api/runs/workload")
    def start_workload_run(
        config: dict[str, Any], background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        from .workloads.registry import workload_registry, import_workload_packages

        import_workload_packages()
        workload_id = str(config.get("workload_id") or "").strip()
        if not workload_id:
            raise HTTPException(status_code=400, detail="workload_id is required")
        try:
            spec = workload_registry.get_spec(workload_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if app.state.active_run_id is not None:
            raise HTTPException(status_code=409, detail="a run is already active")
        # Resolve max_candidates: per-request override > runtime_config > default 4.
        try:
            max_candidates = int(
                config.get("max_candidates")
                or app.state.runtime_config.get("max_candidates")
                or 4
            )
        except (TypeError, ValueError):
            max_candidates = 4
        max_candidates = max(1, min(32, max_candidates))
        # Resolve harness: per-request override > runtime_config > "pydantic_ai".
        harness = str(
            config.get("harness")
            or app.state.runtime_config.get("harness")
            or "pydantic_ai"
        )
        if harness not in {"pydantic_ai", "claude_agent_sdk"}:
            raise HTTPException(
                status_code=400,
                detail=f"unknown harness `{harness}`",
            )
        if harness == "claude_agent_sdk":
            active_model = (
                app.state.runtime_config.get("model")
                or app.state.runtime_config.get("model_name")
                or ""
            )
            if active_model and not (
                active_model.startswith("anthropic:") or "claude" in active_model.lower()
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Claude Agent SDK harness only supports Anthropic "
                        f"models; active model is `{active_model}`. Switch "
                        "harness to `pydantic_ai` or pick an `anthropic:...` "
                        "model in the runtime config."
                    ),
                )
        from datetime import UTC, datetime
        from uuid import uuid4
        run_id = (
            f"{workload_id.replace('_', '-')}-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        )
        payload = {
            "run_id": run_id,
            "workload_id": workload_id,
            "backend_id": spec.backend_id,
            "kind": "workload",
            "max_candidates": max_candidates,
            "harness": harness,
        }
        app.state.runs[run_id] = {"status": "queued", **payload}
        app.state.active_run_id = run_id
        trace_store.emit("run.requested", payload=payload)
        background_tasks.add_task(
            _run_workload_task, app, workload_id, run_id, max_candidates, harness,
        )
        return {
            "run_id": run_id, "status": "queued",
            "max_candidates": max_candidates, "harness": harness,
        }

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str) -> dict[str, Any]:
        run = app.state.runs.get(run_id)
        if run is None:
            events_for_run = _events_for_loop(trace_store.read_events(), run_id)
            if not events_for_run:
                raise HTTPException(status_code=404, detail="run not found")
            return _loop_state(run_id, events_for_run)
        return {**run, "events": [event.model_dump(mode="json") for event in _events_for_loop(trace_store.read_events(), run_id)]}

    @app.get("/api/loops")
    def loops(limit: int = Query(default=200, ge=0, le=2000)) -> dict[str, Any]:
        events_for_loops = trace_store.read_events(limit=limit)
        return {"loops": _summarize_loops(events_for_loops)}

    @app.get("/api/loops/{loop_id}")
    def loop(loop_id: str) -> dict[str, Any]:
        events_for_loop = _events_for_loop(trace_store.read_events(), loop_id)
        if not events_for_loop:
            raise HTTPException(status_code=404, detail="loop not found")
        return _loop_state(loop_id, events_for_loop)

    @app.get("/api/benchmarks")
    def benchmarks(limit: int = Query(default=500, ge=0, le=5000)) -> dict[str, Any]:
        events_for_benchmarks = trace_store.read_events(limit=limit)
        return {"benchmarks": _benchmark_rows(root, events_for_benchmarks)}

    @app.get("/api/benchmarks/{run_id}/series")
    def benchmark_series(run_id: str) -> dict[str, Any]:
        events_for_run = _events_for_loop(trace_store.read_events(), run_id)
        if not events_for_run:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run_id": run_id, "series": _series_for_run(root, events_for_run)}

    @app.get("/api/comparisons")
    def comparisons(limit: int = Query(default=500, ge=0, le=5000)) -> dict[str, Any]:
        return {"comparisons": _comparison_rows(trace_store.read_events(limit=limit))}

    @app.get("/api/episodes/{episode_id}")
    def episode(episode_id: str) -> dict[str, Any]:
        try:
            loaded = EpisodeStore(workspace).load(episode_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return loaded.model_dump(mode="json", exclude_none=True)

    @app.get("/api/artifacts/{artifact_path:path}/preview")
    def artifact_preview_suffix(
        artifact_path: str,
        max_chars: int = Query(default=40_000, ge=100, le=200_000),
    ) -> dict[str, Any]:
        return _artifact_preview(root, artifact_path, max_chars=max_chars)

    @app.get("/api/artifacts/preview/{artifact_path:path}")
    def artifact_preview_prefix(
        artifact_path: str,
        max_chars: int = Query(default=40_000, ge=100, le=200_000),
    ) -> dict[str, Any]:
        return _artifact_preview(root, artifact_path, max_chars=max_chars)

    @app.get("/api/artifacts/{artifact_path:path}")
    def artifact(artifact_path: str) -> FileResponse:
        path = _safe_artifact_path(root, artifact_path)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(path)

    @app.get("/api/telemetry/gpu")
    def gpu_telemetry() -> dict[str, Any]:
        return {"gpus": read_gpu_telemetry()}

    @app.get("/api/logs")
    def logs(limit: int = Query(default=200, ge=0, le=1000)) -> dict[str, Any]:
        return {"lines": trace_store.tail_logs(limit=limit)}

    @app.get("/stream")
    async def stream(after: str | None = None) -> StreamingResponse:
        return StreamingResponse(
            _event_stream(trace_store, after=after),
            media_type="text/event-stream",
        )

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        last_id: str | None = websocket.query_params.get("after")
        live_only = websocket.query_params.get("live") == "1"
        if last_id is None and live_only:
            # Skip historical events; start streaming from the next new one.
            existing = trace_store.read_events(limit=1)
            if existing:
                # Iterate to the end to find the highest event id seen so far.
                tail = trace_store.read_events(limit=0)
                if tail:
                    last_id = tail[-1].event_id
        try:
            while True:
                events = trace_store.read_events(after=last_id, limit=1000)
                if events:
                    for event in events:
                        last_id = event.event_id
                        await websocket.send_text(event.model_dump_json())
                    continue
                await asyncio.sleep(0.15)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                return

    return app


def read_gpu_telemetry() -> list[dict[str, Any]]:
    query = ",".join(
        (
            "index",
            "name",
            "utilization.gpu",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "power.draw",
        )
    )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        gpus.append(
            {
                "index": _to_int(parts[0]),
                "name": parts[1],
                "utilization_gpu_pct": _to_float(parts[2]),
                "memory_used_mib": _to_float(parts[3]),
                "memory_total_mib": _to_float(parts[4]),
                "temperature_c": _to_float(parts[5]),
                "power_w": _to_float(parts[6]),
            }
        )
    return gpus


def _runtime_config(project_root: Path) -> dict[str, Any]:
    settings = CompilagentSettings.from_env(project_root=project_root)
    return {
        "harness": settings.harness,
        "mode": "benchmark",
        "model": settings.model_name,
        "reasoning_effort": settings.reasoning_effort,
        "permission_mode": settings.claude_sdk_permission_mode,
        "capabilities": [
            {
                "id": "pydantic_ai",
                "label": "Current",
                "summary": "ACP-native pydantic-ai harness with approval-gated optimizer tools.",
            },
            {
                "id": "claude_agent_sdk",
                "label": "Claude Agent SDK",
                "summary": "Claude Agent SDK loop with in-process MCP access to optimizer tools.",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Compilagent observation dashboard.")
    parser.add_argument("--workspace-root", type=Path, default=Path(".compilagent-triton"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_app(workspace_root=args.workspace_root),
        host=args.host,
        port=args.port,
    )


async def _event_stream(trace_store: TraceStore, *, after: str | None = None) -> AsyncIterator[str]:
    last_event_id = after
    while True:
        events = trace_store.read_events(after=last_event_id, limit=500)
        for event in events:
            last_event_id = event.event_id
            yield f"id: {event.event_id}\nevent: {event.kind}\ndata: {event.model_dump_json()}\n\n"
        await asyncio.sleep(1.0)


def _run_workload_task(
    app: FastAPI, workload_id: str, run_id: str, max_candidates: int = 4,
    harness: str = "pydantic_ai",
) -> None:
    """Drive the backend-agnostic workload runner in a background task.

    `harness` selects the agent driver — either pydantic-ai (default) or the
    Claude Agent SDK. Both share the same `WorkloadSession` tool surface so
    backend / workload / model dispatch is identical between them.
    """

    trace_store: TraceStore = app.state.trace_store
    root: Path = app.state.workspace_root
    try:
        app.state.runs[run_id] = {**app.state.runs.get(run_id, {}), "status": "running"}
        from .workload_runner import run_workload_optimization

        run_workload_optimization(
            workload_id=workload_id,
            run_id=run_id,
            workspace_root=root,
            trace_store=trace_store,
            max_candidates=max_candidates,
            harness=harness,
        )
        app.state.runs[run_id] = {**app.state.runs.get(run_id, {}), "status": "completed"}
    except Exception as exc:
        app.state.runs[run_id] = {
            **app.state.runs.get(run_id, {}),
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    finally:
        if app.state.active_run_id == run_id:
            app.state.active_run_id = None


def _run_example_task(app: FastAPI, request: RunRequest, run_id: str) -> None:
    trace_store: TraceStore = app.state.trace_store
    root: Path = app.state.workspace_root
    try:
        app.state.runs[run_id] = {**app.state.runs.get(run_id, {}), "status": "running"}
        if request.mode == "optimize":
            run_agent_optimization(
                request=request,
                run_id=run_id,
                workspace_root=root,
                trace_store=trace_store,
            )
        else:
            run_registered_example(
                request=request,
                run_id=run_id,
                workspace_root=root,
                trace_store=trace_store,
            )
        app.state.runs[run_id] = {**app.state.runs.get(run_id, {}), "status": "completed"}
    except Exception as exc:
        app.state.runs[run_id] = {
            **app.state.runs.get(run_id, {}),
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    finally:
        if app.state.active_run_id == run_id:
            app.state.active_run_id = None


def _summarize_loops(events: list[ObservationEvent]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ObservationEvent]] = {}
    for event in events:
        loop_id = _event_loop_id(event)
        if loop_id is None:
            continue
        grouped.setdefault(loop_id, []).append(event)
    return [_loop_summary(loop_id, loop_events) for loop_id, loop_events in grouped.items()]


def _loop_state(loop_id: str, events: list[ObservationEvent]) -> dict[str, Any]:
    return {
        **_loop_summary(loop_id, events),
        "events": [event.model_dump(mode="json") for event in events],
        "benchmarks": _benchmark_rows(None, events),
        "comparisons": _comparison_rows(events),
    }


def _loop_summary(loop_id: str, events: list[ObservationEvent]) -> dict[str, Any]:
    latest = events[-1]
    status = _loop_status(events)
    benchmark = next((event for event in reversed(events) if event.kind == "benchmark.completed"), None)
    best = benchmark.payload.get("best", {}) if benchmark is not None else {}
    return {
        "id": loop_id,
        "status": status,
        "event_count": len(events),
        "started_at": events[0].timestamp.isoformat(),
        "updated_at": latest.timestamp.isoformat(),
        "family": _first_payload_value(events, "family"),
        "example_id": _first_payload_value(events, "example_id"),
        "best_candidate_id": best.get("candidate_id"),
        "best_median_ms": best.get("median_ms"),
        "speedup_vs_baseline": best.get("speedup_vs_baseline"),
    }


def _loop_status(events: list[ObservationEvent]) -> str:
    kinds = [event.kind for event in events]
    if "run.failed" in kinds or "tool.failed" in kinds:
        return "failed"
    if "run.completed" in kinds or "loop.summary" in kinds:
        return "completed"
    if "run.started" in kinds or "benchmark.started" in kinds:
        return "running"
    if "run.requested" in kinds:
        return "queued"
    return "observed"


def _event_loop_id(event: ObservationEvent) -> str | None:
    payload = event.payload or {}
    for key in ("run_id", "loop_id", "episode_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return event.episode_id or event.session_id


def _events_for_loop(events: list[ObservationEvent], loop_id: str) -> list[ObservationEvent]:
    return [event for event in events if _event_loop_id(event) == loop_id or event.episode_id == loop_id]


def _first_payload_value(events: list[ObservationEvent], key: str) -> Any:
    for event in events:
        if key in event.payload:
            return event.payload[key]
    return None


def _benchmark_rows(root: Path | None, events: list[ObservationEvent]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.kind != "benchmark.completed":
            continue
        payload = event.payload
        run_id = payload.get("run_id") or event.event_id
        family = payload.get("family")
        result_rows = payload.get("results")
        if not isinstance(result_rows, list) and root is not None:
            result_rows = _load_result_rows_from_artifacts(root, event.artifact_paths)
        if isinstance(result_rows, list):
            for result in result_rows:
                if isinstance(result, dict):
                    rows.append({"run_id": run_id, "family": family, **result})
            continue
        best = payload.get("best")
        if isinstance(best, dict):
            rows.append({"run_id": run_id, "family": family, **best})
    return rows


def _series_for_run(root: Path, events: list[ObservationEvent]) -> list[dict[str, Any]]:
    return sorted(
        _benchmark_rows(root, events),
        key=lambda row: (
            row.get("median_ms") if isinstance(row.get("median_ms"), int | float) else float("inf"),
            str(row.get("candidate_id", "")),
        ),
    )


def _comparison_rows(events: list[ObservationEvent]) -> list[dict[str, Any]]:
    rows = [
        {"event_id": event.event_id, "timestamp": event.timestamp.isoformat(), **event.payload}
        for event in events
        if event.kind == "comparison.created"
    ]
    if rows:
        return rows
    for event in events:
        if event.kind != "benchmark.completed":
            continue
        best = event.payload.get("best")
        if not isinstance(best, dict):
            continue
        speedup = best.get("speedup_vs_baseline")
        rows.append(
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "run_id": event.payload.get("run_id"),
                "family": event.payload.get("family"),
                "candidate_id": best.get("candidate_id"),
                "speedup_vs_baseline": speedup,
                "delta_percent": (speedup - 1) * 100 if isinstance(speedup, int | float) else None,
                "conclusion": _comparison_conclusion(speedup),
            }
        )
    return rows


def _comparison_conclusion(speedup: Any) -> str:
    if not isinstance(speedup, int | float):
        return "no baseline"
    if speedup > 1.02:
        return "candidate improved baseline"
    if speedup < 0.98:
        return "candidate regressed baseline"
    return "within noise band"


def _load_result_rows_from_artifacts(root: Path, artifact_paths: list[str]) -> list[dict[str, Any]] | None:
    for artifact_path in artifact_paths:
        if not artifact_path.endswith(".json"):
            continue
        path = _safe_artifact_path(root, artifact_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return None


def _artifact_preview(root: Path, artifact_path: str, *, max_chars: int) -> dict[str, Any]:
    path = _safe_artifact_path(root, artifact_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    raw = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(raw) > max_chars
    text = raw[:max_chars] if truncated else raw
    suffix = path.suffix.lower()
    if suffix == ".json":
        render_mode = "json"
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            render_mode = "text"
    elif suffix in {".md", ".markdown"}:
        render_mode = "markdown"
    elif suffix in {".ttir", ".ttgir", ".mlir", ".llir", ".ptx"}:
        render_mode = "ir"
    elif suffix in {".py", ".txt", ".log"}:
        render_mode = "text"
    else:
        render_mode = "text"
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "render_mode": render_mode,
        "language": _language_for_suffix(suffix),
        "text": text,
        "truncated": truncated,
    }


def _language_for_suffix(suffix: str) -> str:
    return {
        ".json": "json",
        ".md": "markdown",
        ".markdown": "markdown",
        ".py": "python",
        ".ttir": "mlir",
        ".ttgir": "mlir",
        ".mlir": "mlir",
        ".llir": "llvm",
        ".ptx": "ptx",
    }.get(suffix, "text")


def _safe_artifact_path(root: Path, artifact_path: str) -> Path:
    """Resolve `artifact_path` to a real file under `root`.

    The UI's URL builder strips leading slashes (so absolute paths arrive here
    as `home/yigit/...`). We try both shapes — first as an absolute path (by
    re-prepending the slash), then as a relative path under `root`. Either way
    the final resolved path must live inside `root` so we don't leak files
    outside the workspace.
    """

    candidates: list[Path] = []
    raw = artifact_path or ""
    if raw.startswith("/"):
        candidates.append(Path(raw).resolve())
    else:
        candidates.append(Path("/" + raw).resolve())
        candidates.append((root / raw).resolve())

    root_resolved = root.resolve()
    for path in candidates:
        try:
            path.relative_to(root_resolved)
        except ValueError:
            continue
        return path
    raise HTTPException(status_code=403, detail="artifact path escapes workspace")


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def redacted_json_response(data: dict[str, Any]) -> JSONResponse:
    return JSONResponse(redact(data))
