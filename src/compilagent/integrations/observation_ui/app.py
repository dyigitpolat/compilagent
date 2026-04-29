"""FastAPI app exposing the compilagent observation UI.

The UI is read-only: it streams events from the on-disk `TraceStore`,
serves artifact previews via the central `ArtifactRendererRegistry`, and
lists registered workloads / backends / harnesses for the SPA's selectors.

`POST /api/runs/workload` triggers an optimization in a background task —
the same path the legacy server exposed — but the heavy lifting is the
core's `OptimizationSession` + `run_session`, not a custom runtime.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from compilagent.bootstrap import load_entry_point_integrations
from compilagent.core.backend import backend_registry
from compilagent.core.workload_registry import workload_registry
from compilagent.harness.base import HarnessRunRequest
from compilagent.harness.registry import harness_registry
from compilagent.session.session import OptimizationSession, run_session
from compilagent.settings import CompilagentSettings
from compilagent.storage.trace_store import TraceStore
from compilagent.storage.workspace import OptimizationWorkspace

from .previews import render_preview

_STATIC_DIR = Path(__file__).parent / "static"


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


def create_app(
    workspace_root: Path | None = None,
    *,
    settings: CompilagentSettings | None = None,
) -> Any:
    """Build the FastAPI app. Uvicorn imports this and runs it."""

    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    settings = settings or CompilagentSettings.from_env()
    cwd = Path.cwd()
    workspace_path = workspace_root or cwd
    workspace = OptimizationWorkspace(
        session_cwd=workspace_path,
        root_name=settings.workspace_dir_name,
    ).ensure()
    trace_store = TraceStore(workspace.root).ensure()

    # Load entry-point integrations once at startup so backends/harnesses are
    # available to the SPA's selectors.
    load_entry_point_integrations()

    app = FastAPI(title="Compilagent Observation UI")

    # ---- index + static ----------------------------------------------------

    @app.get("/")
    async def index():
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    # ---- registries --------------------------------------------------------

    @app.get("/api/backends")
    async def list_backends():
        return JSONResponse(
            {
                "backends": [
                    {"id": bid, "artifact_stages": list(backend_registry.get(bid).artifact_stages)}
                    for bid in backend_registry.ids()
                ]
            }
        )

    @app.get("/api/harnesses")
    async def list_harnesses():
        out = []
        for hid in harness_registry.ids():
            h = harness_registry.get(hid)
            out.append(
                {
                    "id": hid,
                    "supported_providers": list(getattr(h, "supported_providers", ())),
                }
            )
        return JSONResponse({"harnesses": out})

    @app.get("/api/workloads")
    async def list_workloads():
        return JSONResponse(
            {
                "workloads": [
                    spec.serialize() for spec in workload_registry.specs()
                ]
            }
        )

    @app.get("/api/runtime/config")
    async def runtime_config():
        return JSONResponse(settings.public_metadata())

    # ---- events ------------------------------------------------------------

    @app.get("/api/events")
    async def list_events(
        run_id: str | None = None,
        kinds: str | None = None,
        limit: int | None = None,
    ):
        kind_filter = kinds.split(",") if kinds else None
        events = trace_store.read_events(
            run_id=run_id, kinds=kind_filter, limit=limit
        )
        return JSONResponse(
            {"events": [e.serialize() for e in events]}
        )

    # ---- artifact previews -------------------------------------------------

    @app.get("/api/artifacts/preview")
    async def artifact_preview(path: str, max_chars: int = 40_000):
        preview = render_preview(workspace, path, max_chars=max_chars)
        return JSONResponse(preview)

    # ---- runs --------------------------------------------------------------

    @app.post("/api/runs/workload")
    async def start_workload_run(payload: dict):
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

        async def _drive():
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
                user_prompt=str(
                    payload.get(
                        "prompt",
                        "Inspect, propose 3 candidates, run, synthesize.",
                    )
                ),
                model_id=str(payload.get("model_id", settings.model_name)),
                reasoning_effort=settings.reasoning_effort,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                max_turns=int(settings.harness_extra.get("max_turns", 24)),
                extra={**_harness_extra(settings), "cwd": str(workspace.session_cwd)},
            )
            harness = harness_registry.get(harness_id)
            try:
                await run_session(session=session, harness=harness, request=request)
            finally:
                session.finalize()

        asyncio.create_task(_drive())
        return JSONResponse({"run_id": run_id, "harness": harness_id})

    # ---- live event stream -------------------------------------------------

    @app.websocket("/ws")
    async def ws_events(websocket: WebSocket):
        await websocket.accept()
        cursor = 0
        try:
            while True:
                events = trace_store.read_events()
                if cursor < len(events):
                    for ev in events[cursor:]:
                        await websocket.send_text(json.dumps(ev.serialize(), default=str))
                    cursor = len(events)
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    @app.get("/api/stream")
    async def stream_events() -> Any:
        from fastapi.responses import StreamingResponse

        async def _iter() -> AsyncIterator[bytes]:
            cursor = 0
            while True:
                events = trace_store.read_events()
                if cursor < len(events):
                    for ev in events[cursor:]:
                        line = json.dumps(ev.serialize(), default=str)
                        yield f"data: {line}\n\n".encode()
                    cursor = len(events)
                await asyncio.sleep(0.5)

        return StreamingResponse(_iter(), media_type="text/event-stream")

    return app
