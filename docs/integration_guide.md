# Integration Guide

This guide explains how to ship an integration for `compilagent` — your own
compiler, optimizer, agent harness, workload, artifact renderer, or
cross-run policy — without forking the repo.

The core (`src/compilagent/`) defines seven plug-in slots. Each slot is a
Python `Protocol` you implement, plus a registry your code calls. The core
itself imports nothing from any integration.

## Quick map: which slot do I want?

| You want to add… | Slot | Protocol | Registry |
|---|---|---|---|
| A new compiler / optimizer | `Backend` | `compilagent.Backend` | `compilagent.backend_registry` |
| A new agent runtime / LLM harness | `Harness` | `compilagent.Harness` | `compilagent.harness_registry` |
| A new compile target (kernel / model) | `Workload` | builder fn + `WorkloadSpec` | `compilagent.workload_registry` |
| A renderer for a backend-specific artifact suffix | `ArtifactRenderer` | `compilagent.ArtifactRenderer` | `compilagent.artifact_renderer_registry` |
| Cross-run memory / hint provider | `CandidatePolicy` | `compilagent.CandidatePolicy` | passed into `OptimizationSession(policy=...)` |
| Custom agent tools alongside the canonical 8 | `ToolDecl` | `compilagent.ToolDecl` | `Backend.list_introspection_tools()` |
| A new event sink (web socket fan-out, …) | `ObservationSink` | `compilagent.ObservationSink` | passed into `OptimizationSession(sink=...)` |

## Distribution: how does your code get loaded?

Two equivalent ways. Pick whichever fits your packaging.

### A. Direct import
Your package's `__init__.py` calls the appropriate registry at import
time:

```python
# my_package/__init__.py
from compilagent import backend_registry
from .backend import MyBackend

backend_registry.register("my_backend", MyBackend)
```

Users then call `import my_package` (or
`compilagent.import_modules(["my_package"])`) before constructing an
`OptimizationSession`.

### B. Setuptools entry points (recommended for PyPI packages)
Your `pyproject.toml` advertises one of the recognized groups:

```toml
[project.entry-points."compilagent.integrations"]
my_backend = "my_package"
```

Recognized groups (all behave identically — they import the named module
so its registration side effects run): `compilagent.integrations`,
`compilagent.backends`, `compilagent.harnesses`, `compilagent.workloads`.

`OptimizationSession.__init__` calls
`compilagent.load_entry_point_integrations()` once per process, so
`pip install your-package` is enough — the user does not need any glue
code.

Failures importing one entry point do not abort the others; the user can
debug by importing the module explicitly.

## Slot 1: `Backend` — a new compiler / optimizer

Two ways: implement the full `Backend` protocol structurally, or inherit
from `BackendBase` to skip the no-op-defaultable methods.

### Minimal example

```python
from collections.abc import Sequence
from pathlib import Path

from compilagent import (
    Analysis, BackendBase, CompileResult, CorrectnessResult,
    DeviceCapability, Plan, SearchSpace, TimingResult,
    ToleranceConfig, ValidationResult, WorkloadSpec,
)

class MyBackend(BackendBase):
    id = "my_backend"
    artifact_stages = ("source", "asm")

    def device_capability(self) -> DeviceCapability:
        return DeviceCapability(
            arch="cpu", capability_int=None, name="MyTarget",
            memory_total_bytes=None, memory_peak_bandwidth_gbps=None,
        )

    def analyze(self, workload, *, baseline_artifacts) -> Analysis:
        return Analysis(summary={"kind": workload.kind.value, "op_counts": {}})

    def derive_search_space(self, workload, analysis) -> SearchSpace:
        # Levers MUST come from analysis / device cap, never hand-coded.
        return SearchSpace(workload_id=workload.id, backend_id=self.id, levers=())

    def validate_intervention(self, intervention) -> ValidationResult:
        if intervention.target.kind in {"knob"}:
            return ValidationResult(ok=True)
        return ValidationResult(ok=False, errors=(f"unsupported kind {intervention.target.kind}",))

    def compile(self, workload, plan, *, artifact_dir, pass_callback=None) -> CompileResult:
        artifact = artifact_dir / "out.asm"
        artifact.write_text("...", encoding="utf-8")
        return CompileResult(ok=True, elapsed_ms=1.2, artifacts=(artifact,))

    def time_workload(self, workload, plan, *, warmup, repetitions, max_seconds=None) -> TimingResult:
        # Do real timing here.
        timings = (1.0,) * repetitions
        return TimingResult(timings_ms=timings, median_ms=1.0, p20_ms=1.0, p80_ms=1.0)

    def validate_correctness(self, workload, baseline, candidate, tolerance) -> CorrectnessResult:
        return CorrectnessResult(ok=True)
```

### Things to know
- **`CompileResult.artifacts` is the only path collection the session
  reads.** Backend-specific paths (FX dumps, IR per stage, ...) must end
  up in that tuple. Use `metadata` for anything else.
- **Compile failure ≠ exception.** Return `CompileResult(ok=False,
  diagnostics=...)`; the session handles failure-budget bookkeeping.
- **`pass_callback`** lets you stream a per-pass timeline into the UI;
  call it once per executed compile step with a `PassEvent`.
- **`reset_between_compiles`** is where backends like PyTorch put
  `torch._dynamo.reset()` / `torch.cuda.empty_cache()`. Pure backends
  return without doing anything (the default in `BackendBase`).
- **`infer_workload_family`** lets `CandidatePolicy` group similar
  workloads. Return `"matmul"`, `"reduction"`, etc., or `None`.

## Slot 2: `Harness` — a new agent runtime

```python
from collections.abc import AsyncIterator

from compilagent import (
    HarnessRunRequest, StreamEvent, StreamEventKind,
)

class MyHarness:
    id = "my_harness"
    supported_providers = ("anthropic", "openai")

    async def run(self, request: HarnessRunRequest) -> AsyncIterator[StreamEvent]:
        # 1. Translate request.toolset into your runtime's native tools.
        # 2. Drive the agent loop.
        # 3. For each tool call: result = request.toolset.by_name(name).handler(args)
        # 4. Yield StreamEvent(THINKING_*, TEXT_*, TOOL_CALL, TOOL_RESULT/ERROR).
        # 5. End with RUN_FINISHED or RUN_FAILED.
        yield StreamEvent(kind=StreamEventKind.RUN_FINISHED, text="done")
```

### Things to know
- **The harness owns the inner loop.** The session bootstraps and then
  calls `harness.run(request)`; the harness drives until the end and
  yields a final `RUN_FINISHED`/`RUN_FAILED`.
- **Vendor types stay inside the harness package.** Translation from
  `pydantic_ai.messages.*` / `claude_agent_sdk.*` to `StreamEvent` happens
  here, and only here.
- **Tool dispatch is the harness's job.** Call `handler(args)`; on
  `ValueError` yield `TOOL_ERROR` (do not propagate the exception — the
  agent should see it as a retryable error).
- **Read-only / destructive gating.** Respect `ToolDecl.read_only` if your
  runtime exposes permission modes (ACP prepare-tool, Claude SDK
  `allowed_tools`).

## Slot 3: `Workload` — a new compile target

Two decorator flavors:

- `register_workload(spec)` — strict; raises `ValueError` if `spec.id` is
  already registered.
- `register_workload_safely(spec)` — idempotent; silently skips a duplicate
  registration. Use this for **example workloads shipped by an integration**
  (so test harnesses reloading the integration module don't trigger a
  duplicate-id error). Both first-party integrations (`triton`,
  `torch_inductor`) use `register_workload_safely` for their bundled demos.

```python
from compilagent import (
    BenchmarkBudget, ToleranceConfig, WorkloadInstance, WorkloadKind,
    WorkloadSpec, register_workload,
)

_MY_SPEC = WorkloadSpec(
    id="my_kernel",
    title="My Kernel",
    description="Element-wise foo on 1D tensors.",
    kind=WorkloadKind.KERNEL,
    backend_id="my_backend",
    tolerance=ToleranceConfig(atol=1e-5, rtol=1e-4),
    budget=BenchmarkBudget(warmup=3, repetitions=10, max_seconds=60.0),
)

@register_workload(_MY_SPEC)
def build_my_workload(spec: WorkloadSpec) -> WorkloadInstance:
    # Materialise inputs and the forward callable here.
    return WorkloadInstance(
        spec=spec,
        forward=lambda: None,
        example_inputs=(),
        metadata={"source_path": __file__},
    )
```

The decorator registers `(spec, builder)` in the global
`workload_registry`. The session calls `registry.build(workload_id)` to
materialise an instance per run.

The registry exposes two helpers used by the observation UI's
`/api/workloads/{id}/source` endpoint:

```python
workload_registry.get_builder(workload_id)         # → the registered callable
workload_registry.get_builder_source(workload_id)  # → {language, source_path,
                                                   #    source, line_count}
```

`get_builder_source` reads the entire **module** the builder was defined in
(via `inspect.getsource(inspect.getmodule(builder))`), so the UI shows the
spec literal + helpers + the build function, not just the lambda body.
Workloads defined in a REPL or with no resolvable `__file__` round-trip as
`source=""`, `source_path=None` — the UI handles that gracefully.

### Shipping example workloads with a backend

A backend integration that wants to register a curated set of example
workloads for the observation UI follows the pattern used by
`compilagent.integrations.triton` and `compilagent.integrations.torch_inductor`:

```
src/compilagent/integrations/<your_backend>/
├── __init__.py            # registers the Backend + imports `examples`
├── backend.py
└── examples/
    ├── __init__.py        # tolerantly imports each demo module
    ├── demo_one.py        # @register_workload_safely(spec) def build_workload(...)
    └── demo_two.py
```

The integration's `__init__.py` does:

```python
from compilagent.core.backend import backend_registry
from .backend import MyBackend

if "my_backend" not in backend_registry.ids():
    backend_registry.register("my_backend", MyBackend)

from . import examples  # noqa — side-effect import registers workload specs
```

Each demo module's body imports its heavy deps (`torch`, `triton`,
`torchvision`, …) lazily inside `build_workload`, never at module top.
That keeps the spec literal importable on machines without those libs —
the UI sees the workload in its dropdown but a Start click fails with a
clear `RuntimeError("CUDA is required …")` instead of a silent crash at
import. The `examples/__init__.py` wraps each per-module import in
`contextlib.suppress(Exception)` so a single broken demo doesn't take the
others down.

The decorator registers `(spec, builder)` in the global
`workload_registry`. The session calls `registry.build(workload_id)` to
materialise an instance per run.

## Slot 4: `ArtifactRenderer` — a new artifact preview

```python
from dataclasses import dataclass, field
from pathlib import Path

from compilagent import ArtifactPreview, artifact_renderer_registry

@dataclass(frozen=True, slots=True)
class MlirRenderer:
    suffixes = (".ttgir", ".ttir", ".mlir", ".llir")
    priority = 50  # higher than the built-in fallback (10)

    def render(self, path: Path, *, max_chars: int = 40_000) -> ArtifactPreview:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... <truncated>"
        return ArtifactPreview(kind="code", language="mlir", text=text)

artifact_renderer_registry.register(MlirRenderer())
```

You can also return renderers from `Backend.list_artifact_renderers()`;
the session registers them automatically during bootstrap.

## Slot 5: `CandidatePolicy` — cross-run hints

```python
from collections.abc import Sequence

from compilagent import (
    CandidatePolicy, ExperimentLog, Intervention, PolicyHint, Target,
)

class ExperimentLogPolicy:
    name = "experiment_log"

    def __init__(self, log: ExperimentLog):
        self._log = log

    def consult(self, *, workload, analysis, family, arch) -> Sequence[PolicyHint]:
        rows = self._log.recall(
            workload_id=workload.id,
            backend_id=workload.backend_id,
            family=family, arch=arch, top_n=3,
        )
        hints: list[PolicyHint] = []
        for row in rows:
            ivs = tuple(
                Intervention(
                    target=Target(kind=iv["target"]["kind"], selector=iv["target"]["selector"]),
                    payload=iv["payload"],
                )
                for iv in row.get("interventions", [])
            )
            hints.append(PolicyHint(
                suggested_interventions=ivs,
                rationale=f"prior speedup {row.get('speedup', 1.0):.2f}×",
                confidence=0.5,
            ))
        return hints
```

Pass the instance into `OptimizationSession(policy=...)`.

## Slot 6: `ToolDecl` — backend-specific agent tools

```python
from compilagent import ToolDecl

def list_introspection_tools(self):
    def _list_my_passes(_args: dict) -> str:
        return json.dumps(["my_pass_a", "my_pass_b"])

    return (
        ToolDecl(
            name="list_my_passes",
            description="List the optimisation passes the my_backend supports.",
            args_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=_list_my_passes,
            read_only=True,
        ),
    )
```

Names must not collide with the canonical 8 session tools
(`inspect_workload`, `inspect_search_space`, `propose_candidate`,
`propose_candidates`, `run_candidate`, `run_candidates`,
`synthesize_findings`, `compare_runs`). The session calls
`Toolset.with_extra(...)` to merge.

## Slot 7: `ObservationSink` — a custom event destination

```python
from compilagent import ObservationEvent, ObservationSink

class WebSocketFanoutSink:
    def __init__(self, base: ObservationSink, push):
        self._base = base
        self._push = push

    def emit(self, event):
        self._base.emit(event)
        self._push(event.serialize())  # to your WS clients

    def emit_kv(self, kind, **kw):
        ev = ObservationEvent.make(kind, **kw)
        self.emit(ev)
```

Pass the instance into `OptimizationSession(sink=...)`.

## Surfacing diagnostics from a failed integration import

If your integration's `__init__.py` raises during the entry-point loader
import sweep (e.g. an optional dependency is missing), the failure is
captured rather than silently swallowed. The observation UI surfaces it at
`/api/workloads/diagnostics`. To consume the same data programmatically:

```python
from compilagent import load_entry_point_integrations, get_recent_load_failures

load_entry_point_integrations()
for failure in get_recent_load_failures():
    print(failure["module"], failure["error_type"], failure["message"])
```

`get_recent_load_failures()` returns a list of dicts shaped
`{module, group, error_type, message, traceback}` populated by the most
recent loader call. The list resets on the next idempotent load. This means
an integration that imports cleanly during testing but fails on a user's
box (e.g. CUDA missing) shows up in the UI with a precise traceback rather
than an empty workload list.

## What the core guarantees

The core honours these contracts in every Phase-1 release; integrations
written today will keep working as Phase 2+ ships:

- `Intervention.target.kind` is a free string. The core never validates it
  against an enum; backends define their own vocabulary.
- Backend / harness ids are plain strings. No `Literal[...]` types
  enumerate them.
- The session never imports from `compilagent.integrations`.
  No `if backend.id == "triton"` branches anywhere in `src/compilagent/`.
- `CompileResult.artifacts` + `metadata` are the only path / extras
  collections the session reads. Per-stage paths must be in the tuple.
- `StreamEvent` / `StreamEventKind` is the only event type that crosses
  the harness boundary. No vendor message types in core.
- `EventKind` is open: arbitrary string kinds (`"my.custom.event"`) are
  accepted by `ObservationSink.emit_kv`.

## Anti-patterns to avoid in your integration

- Don't import from `compilagent.integrations.*`. The core does not, and
  doing so couples your integration to others.
- Don't put torch / triton / vendor SDK imports at module top in code that
  the core might import. Do them lazily inside methods, or behind a
  `try`/`ImportError` so a missing dependency fails clearly when first
  used.
- Don't depend on `WorkloadSpec.metadata["…"]` keys defined by another
  integration. Use your own keys.
- Don't raise from `compile`/`time_workload`/`validate_correctness` for
  expected failure modes (compile error, OOM, drift). Return result with
  `ok=False` so the session does correct budget bookkeeping.

## See also

- `src/compilagent/core/backend.py` — the `Backend` protocol with full
  per-method contract docs.
- `src/compilagent/harness/base.py` — the `Harness` protocol.
- `src/compilagent/core/candidate_policy.py` — `CandidatePolicy`.
- `src/compilagent/observation/artifacts.py` — `ArtifactRenderer`.
- `tests/compilagent/test_session_with_fakes.py` — a complete in-tree
  example of a `Backend` + `Harness` + `Workload` driven through a session.
