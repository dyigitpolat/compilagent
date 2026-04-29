"""Tests for the observation UI's HTTP routes.

Uses FastAPI's `TestClient` against an in-process app rooted at a tmp dir.
Verifies the registry-backed endpoints + the suffix-table-free preview path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from compilagent.core.backend import backend_registry
from compilagent.core.workload import (
    BenchmarkBudget,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.core.workload_registry import register_workload
from compilagent.harness.registry import harness_registry
from compilagent.observation.artifacts import (
    ArtifactPreview,
    artifact_renderer_registry,
)
from compilagent.observation.events import EventKind
from compilagent.settings import CompilagentSettings
from compilagent.storage.trace_store import TraceStore


def _make_app(tmp_path: Path):
    from compilagent.integrations.observation_ui import create_app

    settings = CompilagentSettings(model_name="test", harness="nop")
    return create_app(workspace_root=tmp_path, settings=settings)


# ----------------------------------------------------------------- basics


def test_index_serves(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200


def test_assets_mounted_at_assets(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/assets/app.js")
        assert r.status_code == 200
        r = client.get("/assets/styles.css")
        assert r.status_code == 200


def test_favicon_returns_204_when_missing(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/favicon.ico")
        assert r.status_code in (200, 204)


# ----------------------------------------------------------- registries


def test_runtime_config_returns_seed_payload(tmp_path):
    class _B:
        id = "fake_b"
        artifact_stages: tuple[str, ...] = ("ir",)

    backend_registry.register("fake_b", _B)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/runtime/config")
        assert r.status_code == 200
        body = r.json()
        assert body["settings"]["harness"] == "nop"
        assert body["settings"]["model_name"] == "test"
        assert any(b["id"] == "fake_b" for b in body["backends"])
        assert isinstance(body["harnesses"], list)
        assert isinstance(body["workloads"], list)


def test_backends_endpoint_returns_registry(tmp_path):
    class _B:
        id = "fake_b"
        artifact_stages: tuple[str, ...] = ("ir",)

    backend_registry.register("fake_b", _B)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/backends")
        assert r.status_code == 200
        ids = [b["id"] for b in r.json()["backends"]]
        assert "fake_b" in ids


def test_harnesses_endpoint_returns_registry(tmp_path):
    class _H:
        id = "fake_h"
        supported_providers: tuple[str, ...] = ("provider_a",)

    harness_registry.register("fake_h", _H)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/harnesses")
        assert r.status_code == 200
        ids = [h["id"] for h in r.json()["harnesses"]]
        assert "fake_h" in ids


# --------------------------------------------------------- workloads


def test_workloads_lists_specs(tmp_path):
    spec = WorkloadSpec(
        id="ui_test_workload", title="UI test", description="x",
        kind=WorkloadKind.KERNEL, backend_id="fake_b",
        tolerance=ToleranceConfig(),
        budget=BenchmarkBudget(warmup=1, repetitions=1, max_seconds=1.0),
    )

    @register_workload(spec)
    def _build(s):
        return WorkloadInstance(spec=s, forward=lambda: None)

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/workloads")
        assert r.status_code == 200
        ids = [w["id"] for w in r.json()["workloads"]]
        assert "ui_test_workload" in ids


def test_workload_source_returns_module_text(tmp_path):
    spec = WorkloadSpec(
        id="ui_src_workload", title="src", description="",
        kind=WorkloadKind.KERNEL, backend_id="fake_b",
    )

    @register_workload(spec)
    def _build(s):  # pragma: no cover - registered for the test only
        return WorkloadInstance(spec=s, forward=lambda: None)

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/workloads/ui_src_workload/source")
        assert r.status_code == 200
        body = r.json()
        assert body["language"] == "python"
        # The test module's text should contain the workload id.
        assert "ui_src_workload" in body["source"]


def test_workload_source_unknown_id_returns_404(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/workloads/no_such/source")
        assert r.status_code == 404


def test_workload_diagnostics_returns_recent_failures(tmp_path):
    """If the entry-point loader recorded failures, /diagnostics surfaces them."""

    from compilagent import bootstrap

    bootstrap._reset_entry_point_cache()
    bootstrap._LAST_FAILURES.append(
        {"module": "fake_broken", "group": "compilagent.integrations",
         "error_type": "ImportError", "message": "boom", "traceback": ""}
    )
    try:
        app = _make_app(tmp_path)
        # `create_app` calls `load_entry_point_integrations()` which clears
        # `_LAST_FAILURES`. To simulate a real startup failure we re-attach
        # via app.state directly:
        app.state.startup_errors = [
            {"module": "fake_broken", "error_type": "ImportError",
             "message": "boom", "traceback": "", "group": "compilagent.integrations"}
        ]
        with TestClient(app) as client:
            r = client.get("/api/workloads/diagnostics")
            assert r.status_code == 200
            errs = r.json()["startup_errors"]
            assert any(e["module"] == "fake_broken" for e in errs)
    finally:
        bootstrap._reset_entry_point_cache()


# --------------------------------------------------------- runs


def test_runs_summary_groups_session_events(tmp_path):
    """`/api/runs` reads the trace store and groups session.* events by run_id."""

    from compilagent.storage.workspace import OptimizationWorkspace

    settings = CompilagentSettings(model_name="test", harness="nop")
    ws = OptimizationWorkspace(
        session_cwd=tmp_path, root_name=settings.workspace_dir_name
    ).ensure()
    store = TraceStore(ws.root).ensure()
    store.emit_kv(
        EventKind.SESSION_STARTED,
        run_id="run-A",
        payload={"workload_id": "wl1", "backend_id": "be1", "harness": "h1"},
    )
    time.sleep(0.001)
    store.emit_kv(
        EventKind.SESSION_FINISHED,
        run_id="run-A",
        payload={"successful_count": 2, "failed_attempts": 1, "leaderboard": []},
    )
    store.emit_kv(
        EventKind.SESSION_STARTED,
        run_id="run-B",
        payload={"workload_id": "wl2", "backend_id": "be1", "harness": "h1"},
    )
    store.emit_kv(
        EventKind.SESSION_FAILED,
        run_id="run-B",
        payload={"error_type": "Boom", "message": "kaboom"},
    )

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/runs")
        assert r.status_code == 200
        runs = {row["run_id"]: row for row in r.json()["runs"]}
        assert runs["run-A"]["status"] == "finished"
        assert runs["run-A"]["workload_id"] == "wl1"
        assert runs["run-B"]["status"] == "failed"


def test_run_events_endpoint(tmp_path):
    from compilagent.storage.workspace import OptimizationWorkspace

    settings = CompilagentSettings(model_name="test", harness="nop")
    ws = OptimizationWorkspace(
        session_cwd=tmp_path, root_name=settings.workspace_dir_name
    ).ensure()
    store = TraceStore(ws.root).ensure()
    store.emit_kv(
        EventKind.SESSION_STARTED, run_id="r1",
        payload={"workload_id": "w", "backend_id": "b", "harness": "h"},
    )
    store.emit_kv(EventKind.RUN_PROGRESS, run_id="r1", payload={"successful_count": 1})
    store.emit_kv(EventKind.SESSION_STARTED, run_id="r2",
                  payload={"workload_id": "x", "backend_id": "b", "harness": "h"})

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/runs/r1/events")
        assert r.status_code == 200
        kinds = [e["kind"] for e in r.json()["events"]]
        assert "session.started" in kinds
        assert "run.progress" in kinds
        assert all(e["run_id"] == "r1" for e in r.json()["events"])


def test_run_leaderboard_returns_latest(tmp_path):
    from compilagent.storage.workspace import OptimizationWorkspace

    settings = CompilagentSettings(model_name="test", harness="nop")
    ws = OptimizationWorkspace(
        session_cwd=tmp_path, root_name=settings.workspace_dir_name
    ).ensure()
    store = TraceStore(ws.root).ensure()
    store.emit_kv(
        EventKind.LEADERBOARD_UPDATED, run_id="rid",
        payload={"rows": [
            {"candidate_id": "baseline", "median_ms": 10.0, "speedup_vs_baseline": 1.0, "correctness_ok": True},
            {"candidate_id": "cand-1", "median_ms": 5.0, "speedup_vs_baseline": 2.0, "correctness_ok": True},
        ]},
    )
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/runs/rid/leaderboard")
        assert r.status_code == 200
        body = r.json()
        assert body["run_id"] == "rid"
        assert len(body["rows"]) == 2


# --------------------------------------------------------- artifacts


def test_artifact_preview_uses_registered_renderer(tmp_path):
    """A backend-shaped suffix is rendered by whatever the registry holds —
    no suffix table in the UI code itself."""

    target = tmp_path / ".compilagent" / "demo.custom_ir"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("HELLO custom_ir BODY", encoding="utf-8")

    from dataclasses import dataclass

    @dataclass(frozen=True, slots=True)
    class CustomRenderer:
        suffixes: tuple[str, ...] = (".custom_ir",)
        priority: int = 99

        def render(self, path, *, max_chars: int = 40_000):
            return ArtifactPreview(
                kind="code",
                language="custom",
                text=path.read_text(encoding="utf-8"),
            )

    artifact_renderer_registry.register(CustomRenderer())
    try:
        app = _make_app(tmp_path)
        with TestClient(app) as client:
            r = client.get("/api/artifacts/preview/demo.custom_ir")
            assert r.status_code == 200
            body = r.json()
            assert body["language"] == "custom"
            assert "HELLO custom_ir BODY" in body["text"]
    finally:
        artifact_renderer_registry.clear()
        from compilagent.observation.artifacts import build_default_registry

        for r in build_default_registry()._renderers:
            artifact_renderer_registry.register(r)


def test_artifact_preview_path_traversal_blocked(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/artifacts/preview/..%2F..%2F..%2Fetc%2Fpasswd")
        body = r.json()
        assert body.get("error") == "path_traversal" or not body.get("exists")


def test_artifact_download_serves_file(tmp_path):
    target = tmp_path / ".compilagent" / "out.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello", encoding="utf-8")
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/artifacts/out.txt")
        assert r.status_code == 200
        assert "hello" in r.text


def test_artifact_download_404_when_missing(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/artifacts/nope.txt")
        assert r.status_code == 404


# --------------------------------------------------------- gpu telemetry


def test_gpu_telemetry_returns_empty_when_nvidia_smi_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/telemetry/gpu")
        assert r.status_code == 200
        body = r.json()
        assert body["gpus"] == []


# --------------------------------------------------------- websocket


def test_websocket_accepts_with_live_flag(tmp_path):
    from compilagent.storage.workspace import OptimizationWorkspace

    settings = CompilagentSettings(model_name="test", harness="nop")
    ws_root = OptimizationWorkspace(
        session_cwd=tmp_path, root_name=settings.workspace_dir_name
    ).ensure()
    store = TraceStore(ws_root.root).ensure()
    # Pre-existing event before the WS connects with live=1; should NOT arrive.
    store.emit_kv(EventKind.SESSION_STARTED, run_id="rid",
                  payload={"workload_id": "x", "backend_id": "b", "harness": "h"})

    app = _make_app(tmp_path)
    with TestClient(app) as client, client.websocket_connect(
        "/ws?live=1&run_id=rid"
    ) as ws_client:
        # Drop a fresh event after the WS opened.
        store.emit_kv(EventKind.RUN_PROGRESS, run_id="rid",
                      payload={"successful_count": 1})
        data = ws_client.receive_text()
        ev = json.loads(data)
        assert ev["kind"] == "run.progress"
        assert ev["run_id"] == "rid"


def test_websocket_filters_by_run_id(tmp_path):
    from compilagent.storage.workspace import OptimizationWorkspace

    settings = CompilagentSettings(model_name="test", harness="nop")
    ws_root = OptimizationWorkspace(
        session_cwd=tmp_path, root_name=settings.workspace_dir_name
    ).ensure()
    store = TraceStore(ws_root.root).ensure()

    app = _make_app(tmp_path)
    with TestClient(app) as client, client.websocket_connect(
        "/ws?live=1&run_id=keep"
    ) as ws_client:
        store.emit_kv(EventKind.RUN_PROGRESS, run_id="drop",
                      payload={"successful_count": 0})
        store.emit_kv(EventKind.RUN_PROGRESS, run_id="keep",
                      payload={"successful_count": 5})
        data = ws_client.receive_text()
        ev = json.loads(data)
        assert ev["run_id"] == "keep"
        assert ev["payload"]["successful_count"] == 5


# --------------------------------------------------------- run trigger


def test_post_runs_workload_returns_run_id(tmp_path):
    spec = WorkloadSpec(
        id="ui_run_workload", title="t", description="",
        kind=WorkloadKind.KERNEL, backend_id="fake_be",
    )

    @register_workload(spec)
    def _build(s):  # pragma: no cover
        return WorkloadInstance(spec=s, forward=lambda: None)

    class _StubHarness:
        id = "stub"
        supported_providers: tuple[str, ...] = ("test",)

        async def run(self, request):
            from compilagent.harness.base import StreamEvent, StreamEventKind

            yield StreamEvent(kind=StreamEventKind.RUN_FINISHED, text="ok")

    class _StubBackend:
        id = "fake_be"
        artifact_stages: tuple[str, ...] = ()

        def device_capability(self):
            from compilagent.core.analysis import DeviceCapability
            return DeviceCapability(arch="cpu", capability_int=None, name="x",
                                    memory_total_bytes=None, memory_peak_bandwidth_gbps=None)

        def analyze(self, workload, *, baseline_artifacts):
            from compilagent.core.analysis import Analysis
            return Analysis()

        def derive_search_space(self, workload, analysis):
            from compilagent.core.search_space import SearchSpace
            return SearchSpace(workload_id=workload.id, backend_id=self.id, levers=())

        def validate_intervention(self, intervention):
            from compilagent.core.plan import ValidationResult
            return ValidationResult(ok=True)

        def interpret_plan(self, plan):
            return plan

        def apply_intervention(self, plan, intervention):
            from compilagent.core.plan import Plan
            return Plan(interventions=plan.interventions + (intervention,))

        def compile(self, workload, plan, *, artifact_dir, pass_callback=None):
            from compilagent.core.analysis import CompileResult
            return CompileResult(ok=True, elapsed_ms=0.0)

        def time_workload(self, workload, plan, *, warmup, repetitions, max_seconds=None):
            from compilagent.core.analysis import TimingResult
            return TimingResult(timings_ms=(1.0,), median_ms=1.0, p20_ms=1.0, p80_ms=1.0)

        def validate_correctness(self, workload, baseline, candidate, tolerance):
            from compilagent.core.analysis import CorrectnessResult
            return CorrectnessResult(ok=True)

        def reset_between_compiles(self, workload):
            return None

        def list_introspection_tools(self):
            return ()

        def list_artifact_renderers(self):
            return ()

        def infer_workload_family(self, workload):
            return None

    backend_registry.register("fake_be", _StubBackend)
    harness_registry.register("stub", _StubHarness)

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/runs/workload",
            json={"workload_id": "ui_run_workload", "harness": "stub", "max_candidates": 1},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["run_id"].startswith("ui-")
        assert body["harness"] == "stub"
        assert body["workload_id"] == "ui_run_workload"


def test_post_runs_workload_rejects_unknown_workload(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post("/api/runs/workload", json={"workload_id": "no_such", "harness": "stub"})
        assert r.status_code == 404
