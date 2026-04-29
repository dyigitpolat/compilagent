"""Drive `torch.compile` for an arbitrary `WorkloadInstance`.

The harness builds a custom Dynamo backend at compile time that:

  1. Applies the plan's FX-graph rewrites to the captured `GraphModule`.
  2. Enters context managers for inductor / dynamo / lowering / scheduler
     overrides scoped to this compile.
  3. Hands the (possibly mutated) graph to `compile_fx` to produce a callable.
  4. Captures the inductor-generated artifacts (output code, schedule logs)
     into `artifact_dir`.

The harness is import-safe on machines without torch — every torch import is
lazy and any failure during compilation produces a `CompileResult(ok=False)`
with a diagnostic message instead of crashing.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .interception import (
    patched_dynamo_config,
    patched_inductor_choices,
    patched_inductor_config,
    patched_lowering_registry,
    scheduler_passes,
)


@dataclass(slots=True)
class InductorPlan:
    """A backend-specific projection of `Plan` for the inductor harness.

    Built by `TorchInductorBackend.compile()` from the generic `Plan`. Each
    field is a typed slice the interception layer understands.
    """

    inductor_config: dict[str, Any] = field(default_factory=dict)
    dynamo_config: dict[str, Any] = field(default_factory=dict)
    lowering_overrides: dict[str, Callable[..., Any]] = field(default_factory=dict)
    fx_rewriters: list[Callable[[Any], Any]] = field(default_factory=list)
    pre_fusion_pass: Any | None = None
    post_fusion_pass: Any | None = None
    choices_handler: Any | None = None


@dataclass(slots=True)
class InductorCompileOutcome:
    """Summary of what `_drive_compile` produced for one workload+plan."""

    ok: bool
    elapsed_ms: float
    compiled_callable: Any | None = None
    output_code_path: Path | None = None
    schedule_log_path: Path | None = None
    fx_graph_path: Path | None = None
    diagnostics: str | None = None
    captured_logs: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _capture_torch_logs(artifact_dir: Path, channels: tuple[str, ...]) -> tuple[Any, dict[str, Path]]:
    """Attach a per-compile file handler to torch's loggers.

    Returns a context-manager-shaped object whose `__exit__` removes the handler,
    plus the dict mapping channel→file path.
    """

    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    handlers: list[tuple[logging.Logger, logging.Handler]] = []
    for ch in channels:
        logger = logging.getLogger(f"torch._inductor.{ch}") if ch != "dynamo" else logging.getLogger("torch._dynamo")
        # Best-effort wider net — channels like "output_code" attach to the
        # same root inductor logger, so keep `ch`-named files distinct.
        path = artifact_dir / f"{ch}.log"
        paths[ch] = path
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        # Force the channel to emit by setting the env vars torch._logging reads.
        # We pre-set TORCH_LOGS in the harness entrypoint; here we just attach.
        handlers.append((logger, handler))

    @contextlib.contextmanager
    def _ctx():
        try:
            yield paths
        finally:
            for logger, handler in handlers:
                try:
                    logger.removeHandler(handler)
                    handler.close()
                except Exception:  # noqa: BLE001
                    pass

    return _ctx(), paths


def _make_dynamo_backend(plan: InductorPlan, artifact_dir: Path) -> Callable[..., Any]:
    """Closure-bound backend used as `torch.compile(backend=...)`."""

    # Always-on inductor defaults that prevent autotune log spam from configs
    # that exceed the device's shared-memory budget. The plan's overrides win
    # if it explicitly sets these keys.
    autotune_safety_defaults = {
        "max_autotune_prune_choices_based_on_shared_mem": True,
    }
    inductor_overrides = {**autotune_safety_defaults, **plan.inductor_config}

    def backend(gm, example_inputs):  # gm: torch.fx.GraphModule
        for rewriter in plan.fx_rewriters:
            try:
                gm = rewriter(gm) or gm
            except Exception:  # noqa: BLE001
                continue
        try:
            (artifact_dir / "fx_graph.py").write_text(gm.print_readable(print_output=False), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        from torch._inductor.compile_fx import compile_fx  # type: ignore[import-not-found]

        with ExitStack() as stack:
            stack.enter_context(patched_inductor_config(inductor_overrides))
            stack.enter_context(patched_dynamo_config(plan.dynamo_config))
            stack.enter_context(patched_lowering_registry(plan.lowering_overrides))
            stack.enter_context(patched_inductor_choices(plan.choices_handler))
            stack.enter_context(
                scheduler_passes(
                    pre_fusion=plan.pre_fusion_pass,
                    post_fusion=plan.post_fusion_pass,
                )
            )
            return compile_fx(gm, example_inputs)

    return backend


def drive_compile(
    workload_instance: Any,
    plan: InductorPlan,
    *,
    artifact_dir: Path,
    log_channels: tuple[str, ...] = ("output_code", "schedule", "fusion", "recompiles"),
) -> InductorCompileOutcome:
    """Compile a workload through `torch.compile` with the supplied plan."""

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Tell torch._logging which channels to emit (they get attached by file
    # handlers below). This is set per-process; harmless if the value is
    # already set to something else (we restore on exit via env-var snapshot).
    prev_torch_logs = os.environ.get("TORCH_LOGS")
    os.environ["TORCH_LOGS"] = ",".join(log_channels)

    started = time.perf_counter()
    log_paths: dict[str, Path] = {}
    try:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            return InductorCompileOutcome(
                ok=False,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                diagnostics=f"torch import failed: {exc!r}",
            )

        log_ctx, log_paths = _capture_torch_logs(artifact_dir, log_channels)
        # Capture (and silence) the per-config autotune probe errors. Inductor
        # sweeps every Triton template config; the ones that exceed shared
        # memory raise `OutOfMemoryError` and get logged at ERROR — but they're
        # *probes*, not actual run failures. We collect them in a buffer the
        # caller can surface to the agent, and raise the logger's own level so
        # they stop spamming the server console.
        probe_warnings: list[str] = []
        probe_logger = logging.getLogger("torch._inductor.select_algorithm")
        prev_probe_level = probe_logger.level
        probe_logger.setLevel(logging.CRITICAL)

        class _ProbeBuffer(logging.Handler):
            def emit(self, record):
                try:
                    line = record.getMessage()
                except Exception:  # noqa: BLE001
                    return
                if "Required:" in line and "Hardware limit" in line or "No valid triton configs" in line:
                    probe_warnings.append(line.strip())

        probe_handler = _ProbeBuffer(level=logging.WARNING)
        probe_logger.addHandler(probe_handler)
        # Same for the dispatch / autotune cache loggers in case more spam shows up.
        with log_ctx:
            try:
                torch._dynamo.reset()
            except Exception:  # noqa: BLE001
                pass
            backend = _make_dynamo_backend(plan, artifact_dir)
            module = workload_instance.metadata.get("module")
            forward_callable = workload_instance.forward
            example_inputs = workload_instance.example_inputs
            try:
                if module is not None and hasattr(torch, "compile"):
                    compiled = torch.compile(module, backend=backend, dynamic=False)
                else:
                    compiled = torch.compile(forward_callable, backend=backend, dynamic=False)
                # Trigger compilation + capture inductor's generated python via
                # the canonical helper. This is the same surface inductor's own
                # tests use; produces a list of source strings (one per kernel
                # bundle generated by this run).
                from torch._inductor.utils import run_and_get_code  # type: ignore[import-not-found]
                if module is not None:
                    _, generated_sources = run_and_get_code(compiled, *example_inputs)
                else:
                    _, generated_sources = run_and_get_code(compiled)
                output_code_path: Path | None = None
                if generated_sources:
                    output_code_path = artifact_dir / "output_code.py"
                    output_code_path.write_text(
                        "\n\n# === next inductor module ===\n\n".join(generated_sources),
                        encoding="utf-8",
                    )
            except Exception as exc:  # noqa: BLE001
                return InductorCompileOutcome(
                    ok=False,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    diagnostics=f"compile failed: {type(exc).__name__}: {exc}",
                    captured_logs=log_paths,
                    warnings=probe_warnings[:64],
                )
            elapsed = (time.perf_counter() - started) * 1000.0
            return InductorCompileOutcome(
                ok=True,
                elapsed_ms=elapsed,
                compiled_callable=compiled,
                output_code_path=output_code_path,
                schedule_log_path=log_paths.get("schedule"),
                fx_graph_path=artifact_dir / "fx_graph.py" if (artifact_dir / "fx_graph.py").exists() else None,
                captured_logs=log_paths,
                warnings=probe_warnings[:64],
            )
    finally:
        # Restore TORCH_LOGS
        if prev_torch_logs is None:
            os.environ.pop("TORCH_LOGS", None)
        else:
            os.environ["TORCH_LOGS"] = prev_torch_logs
        try:
            probe_logger.removeHandler(probe_handler)
            probe_logger.setLevel(prev_probe_level)
        except Exception:  # noqa: BLE001
            pass
