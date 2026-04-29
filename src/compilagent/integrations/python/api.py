"""User-facing entry points: `optimize_module` and `optimize_kernel`.

Both build an in-memory `WorkloadSpec`, register a builder closure with the
core's `workload_registry`, drive an `OptimizationSession` against the
configured harness, and return an `OptimizationResult` that includes the
optimized drop-in callable when a validated candidate beat baseline.

Defaults:
  - `optimize_module` → `backend_id="torch_inductor"`.
  - `optimize_kernel` → `backend_id="triton"`.

Both accept a `backend_id` override so out-of-tree backends can be plugged
in without forking these functions.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from compilagent.bootstrap import import_modules, load_entry_point_integrations
from compilagent.core.backend import backend_registry
from compilagent.core.workload import (
    BenchmarkBudget,
    DtypePolicy,
    ShapePolicy,
    ToleranceConfig,
    WorkloadInstance,
    WorkloadKind,
    WorkloadSpec,
)
from compilagent.harness.base import HarnessRunRequest
from compilagent.harness.registry import harness_registry
from compilagent.session.session import OptimizationSession, run_session
from compilagent.settings import CompilagentSettings
from compilagent.storage.trace_store import TraceStore
from compilagent.storage.workspace import OptimizationWorkspace

from .result import OptimizationResult

_USER_PROMPT = (
    "Optimize the registered workload using the canonical 8-tool toolset. "
    "Inspect the workload + search space first, then propose 2-4 multi-"
    "intervention candidates, run them, synthesise findings, and propose "
    "more if budget remains. Stop when no further candidate is likely to "
    "beat the current best."
)


def _system_instructions(spec: WorkloadSpec, baseline_ms: float | None) -> str:
    return (
        "You are a compiler-heuristic researcher. Replace baseline compiler "
        "decisions with verified faster ones. Use the canonical session "
        "tools; never propose an intervention the backend will reject. "
        f"Workload: id={spec.id}, kind={spec.kind.value}, "
        f"backend_id={spec.backend_id}; baseline median = {baseline_ms} ms."
    )


def _ensure_integrations_loaded(extra: Sequence[str] = ()) -> None:
    """Import any out-of-tree integrations the env advertises + caller list."""

    load_entry_point_integrations()
    if extra:
        import_modules(list(extra))


def _harness_extra_from_settings(settings: CompilagentSettings) -> dict[str, Any]:
    extras: dict[str, Any] = dict(settings.harness_extra or {})
    if settings.anthropic_api_key is not None:
        extras.setdefault(
            "anthropic_api_key", settings.anthropic_api_key.get_secret_value()
        )
    if settings.mistral_api_key is not None:
        extras.setdefault(
            "mistral_api_key", settings.mistral_api_key.get_secret_value()
        )
    if settings.openai_api_key is not None:
        extras.setdefault(
            "openai_api_key", settings.openai_api_key.get_secret_value()
        )
    return extras


def _read_leaderboard_json(session: OptimizationSession) -> list[dict[str, Any]]:
    try:
        return json.loads(session.compare_runs())
    except Exception:  # noqa: BLE001
        return []


def _select_best(
    rows: Sequence[dict[str, Any]],
    *,
    candidates: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Pick the best validated candidate row + its session-side record."""

    qualifying = [
        r
        for r in rows
        if r.get("candidate_id")
        and r["candidate_id"] != "baseline"
        and isinstance(r.get("speedup_vs_baseline"), (int, float))
        and r["speedup_vs_baseline"] > 1.0
        and (r.get("correctness_ok") is None or r["correctness_ok"])
    ]
    qualifying.sort(key=lambda r: r["speedup_vs_baseline"] or 0.0, reverse=True)
    if not qualifying:
        return None, None
    best = qualifying[0]
    record = candidates.get(best["candidate_id"])
    return best, record


def _run_session(
    *,
    spec: WorkloadSpec,
    settings: CompilagentSettings,
    workspace: OptimizationWorkspace,
    harness_id: str,
    model_id: str,
    max_candidates: int,
    user_prompt: str,
) -> OptimizationResult:
    started = time.perf_counter()
    workspace.ensure()
    sink = TraceStore(workspace.root).ensure()
    session = OptimizationSession(
        workload_id=spec.id,
        run_id=f"run-{uuid.uuid4().hex[:12]}",
        workspace=workspace,
        sink=sink,
        max_candidates=max_candidates,
    )
    request = HarnessRunRequest(
        toolset=session.toolset,
        system_instructions=_system_instructions(spec, session.baseline_time.median_ms),
        user_prompt=user_prompt,
        model_id=model_id,
        reasoning_effort=settings.reasoning_effort,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        max_turns=int(settings.harness_extra.get("max_turns", 24)),
        extra=_harness_extra_from_settings(settings),
    )
    harness = harness_registry.get(harness_id)
    harness_result = asyncio.run(
        run_session(session=session, harness=harness, request=request)
    )

    leaderboard = _read_leaderboard_json(session)
    best, record = _select_best(leaderboard, candidates=session.candidates)
    optimized_callable = None
    best_plan = None
    if record is not None:
        compile_outcome = record.get("compile")
        if compile_outcome is not None:
            optimized_callable = getattr(compile_outcome, "compiled_callable", None)
        best_plan = record.get("plan")

    session.finalize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return OptimizationResult(
        workload_id=spec.id,
        backend_id=spec.backend_id,
        harness=harness_id,
        baseline_median_ms=session.baseline_time.median_ms,
        best_speedup=(best.get("speedup_vs_baseline") if best else None),
        best_candidate_id=(best.get("candidate_id") if best else None),
        best_median_ms=(best.get("median_ms") if best else None),
        correctness_ok=(best.get("correctness_ok") if best else None),
        max_abs_diff=None,
        final_text=harness_result.final_text,
        elapsed_ms=elapsed_ms,
        candidates=leaderboard,
        workspace_root=workspace.root,
        optimized_callable=optimized_callable,
        best_plan=best_plan,
    )


# ----------------------------------------------------------------- public API


def optimize_module(
    model: Any,
    example_inputs: tuple[Any, ...],
    *,
    max_candidates: int = 8,
    backend_id: str = "torch_inductor",
    harness: str | None = None,
    model_id: str | None = None,
    settings: CompilagentSettings | None = None,
    workspace: OptimizationWorkspace | None = None,
    user_prompt: str = _USER_PROMPT,
    integrations: Sequence[str] = (),
) -> OptimizationResult:
    """Drop-in replacement for `torch.compile` driven by an LLM agent.

    Validates `backend_id` exists in the registry; raises `KeyError` with a
    helpful message otherwise. Loads entry-point-advertised integrations
    once before the registry check so `pip install compilagent-acme` and
    then `optimize_module(..., backend_id="acme")` works with no glue code.
    """

    settings = settings or CompilagentSettings.from_env()
    _ensure_integrations_loaded(extra=tuple(integrations))

    if backend_id not in backend_registry.ids():
        known = backend_registry.ids()
        raise KeyError(
            f"Unknown backend `{backend_id}`. Registered: {known or '(none)'}. "
            f"Did you forget to `import compilagent.integrations.{backend_id}`?"
        )

    workload_id = f"py_module_{uuid.uuid4().hex[:10]}"
    spec = _build_module_spec(
        workload_id=workload_id,
        model=model,
        example_inputs=example_inputs,
        backend_id=backend_id,
        max_seconds=float(settings.max_benchmark_seconds),
    )

    @workload_registry_register(spec)
    def _build(_spec: WorkloadSpec) -> WorkloadInstance:
        return WorkloadInstance(
            spec=_spec,
            forward=_make_forward(model, example_inputs),
            example_inputs=example_inputs,
            metadata={
                "kind": "module",
                "module_class": type(model).__name__,
            },
        )

    return _run_session(
        spec=spec,
        settings=settings,
        workspace=workspace or OptimizationWorkspace(session_cwd=Path.cwd()),
        harness_id=(harness or settings.harness),
        model_id=(model_id or settings.model_name),
        max_candidates=max_candidates,
        user_prompt=user_prompt,
    )


def optimize_kernel(
    kernel: Any,
    args: tuple[Any, ...],
    *,
    grid: Callable[[dict[str, Any]], tuple[int, ...]],
    constexpr: dict[str, Any] | None = None,
    max_candidates: int = 8,
    backend_id: str = "triton",
    harness: str | None = None,
    model_id: str | None = None,
    settings: CompilagentSettings | None = None,
    workspace: OptimizationWorkspace | None = None,
    user_prompt: str = _USER_PROMPT,
    integrations: Sequence[str] = (),
) -> OptimizationResult:
    """Optimize a `@triton.jit` kernel under the configured agent."""

    settings = settings or CompilagentSettings.from_env()
    _ensure_integrations_loaded(extra=tuple(integrations))

    if backend_id not in backend_registry.ids():
        known = backend_registry.ids()
        raise KeyError(
            f"Unknown backend `{backend_id}`. Registered: {known or '(none)'}. "
            f"Did you forget to `import compilagent.integrations.{backend_id}`?"
        )

    workload_id = f"py_kernel_{uuid.uuid4().hex[:10]}"
    spec = _build_kernel_spec(
        workload_id=workload_id,
        kernel=kernel,
        constexpr=constexpr or {},
        backend_id=backend_id,
        max_seconds=float(settings.max_benchmark_seconds),
    )

    @workload_registry_register(spec)
    def _build(_spec: WorkloadSpec) -> WorkloadInstance:
        return WorkloadInstance(
            spec=_spec,
            forward=_make_kernel_forward(kernel, args, grid, constexpr or {}),
            example_inputs=args,
            metadata={
                "kind": "kernel",
                "kernel_symbol": _kernel_symbol(kernel),
                "source_path": _kernel_source_path(kernel),
            },
        )

    return _run_session(
        spec=spec,
        settings=settings,
        workspace=workspace or OptimizationWorkspace(session_cwd=Path.cwd()),
        harness_id=(harness or settings.harness),
        model_id=(model_id or settings.model_name),
        max_candidates=max_candidates,
        user_prompt=user_prompt,
    )


# --------------------------------------------------------- internal helpers


def workload_registry_register(spec: WorkloadSpec):
    """Wrap `register_workload` so each call gets a fresh, isolated registration.

    The user may invoke `optimize_module` multiple times in one process; each
    call gets a unique `workload_id`, so registration never collides.
    """

    from compilagent.core.workload_registry import register_workload

    return register_workload(spec)


def _build_module_spec(
    *,
    workload_id: str,
    model: Any,
    example_inputs: tuple[Any, ...],
    backend_id: str,
    max_seconds: float,
) -> WorkloadSpec:
    activation_dtype, param_dtype = _infer_dtypes(model, example_inputs)
    shape_policy = _infer_shape_policy(example_inputs)
    return WorkloadSpec(
        id=workload_id,
        title=type(model).__name__,
        description="In-memory PyTorch module wrapped by optimize_module.",
        kind=WorkloadKind.FULL_MODEL,
        backend_id=backend_id,
        dtype_policy=DtypePolicy(
            activation_dtype=activation_dtype,
            param_dtype=param_dtype,
            autocast=False,
        ),
        shape_policy=shape_policy,
        tolerance=ToleranceConfig(atol=1e-4, rtol=1e-3),
        budget=BenchmarkBudget(warmup=3, repetitions=10, max_seconds=max_seconds),
        metadata={
            "module_class": type(model).__name__,
        },
    )


def _build_kernel_spec(
    *,
    workload_id: str,
    kernel: Any,
    constexpr: dict[str, Any],
    backend_id: str,
    max_seconds: float,
) -> WorkloadSpec:
    return WorkloadSpec(
        id=workload_id,
        title=_kernel_symbol(kernel),
        description="In-memory Triton kernel wrapped by optimize_kernel.",
        kind=WorkloadKind.KERNEL,
        backend_id=backend_id,
        tolerance=ToleranceConfig(atol=1e-5, rtol=1e-4),
        budget=BenchmarkBudget(warmup=3, repetitions=10, max_seconds=max_seconds),
        metadata={
            "kernel_symbol": _kernel_symbol(kernel),
            "source_path": _kernel_source_path(kernel),
            "constexpr": dict(constexpr),
        },
    )


def _infer_dtypes(model: Any, example_inputs: tuple[Any, ...]) -> tuple[str, str]:
    activation = "fp32"
    param = "fp32"
    try:
        if example_inputs:
            t = example_inputs[0]
            dtype = getattr(t, "dtype", None)
            if dtype is not None:
                activation = _dtype_str(dtype)
        params = list(getattr(model, "parameters", lambda: [])())
        if params:
            dtype = getattr(params[0], "dtype", None)
            if dtype is not None:
                param = _dtype_str(dtype)
    except Exception:  # noqa: BLE001
        pass
    return activation, param


def _dtype_str(dtype: Any) -> str:
    name = str(dtype).lower()
    if "bfloat16" in name or "bf16" in name:
        return "bf16"
    if "float16" in name or "half" in name or "fp16" in name:
        return "fp16"
    return "fp32"


def _infer_shape_policy(example_inputs: tuple[Any, ...]) -> ShapePolicy:
    extra: dict[str, Any] = {}
    batch = 1
    if example_inputs:
        t = example_inputs[0]
        shape = getattr(t, "shape", None)
        if shape is not None:
            try:
                shape_tuple = tuple(int(x) for x in shape)
                if shape_tuple:
                    batch = shape_tuple[0]
                    extra["input_shape"] = list(shape_tuple)
            except Exception:  # noqa: BLE001
                pass
    return ShapePolicy(batch_size=batch, extra=extra)


def _make_forward(model: Any, example_inputs: tuple[Any, ...]) -> Callable[[], Any]:
    def _forward():
        return model(*example_inputs) if example_inputs else model()

    return _forward


def _make_kernel_forward(
    kernel: Any,
    args: tuple[Any, ...],
    grid: Callable[[dict[str, Any]], tuple[int, ...]],
    constexpr: dict[str, Any],
) -> Callable[[], Any]:
    def _forward():
        return kernel[grid(constexpr)](*args, **constexpr)

    return _forward


def _kernel_symbol(kernel: Any) -> str:
    return getattr(kernel, "__name__", None) or type(kernel).__name__


def _kernel_source_path(kernel: Any) -> str | None:
    try:
        path = inspect.getsourcefile(kernel)
        return path or None
    except TypeError:
        try:
            inner = getattr(kernel, "fn", None)
            return inspect.getsourcefile(inner) if inner is not None else None
        except Exception:  # noqa: BLE001
            return None
