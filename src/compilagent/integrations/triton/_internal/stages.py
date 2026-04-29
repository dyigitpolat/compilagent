from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pipeline import PassIntervention, PassResult, StagePipeline


@dataclass(frozen=True, slots=True)
class StageHookConfig:
    key_material: str
    artifact_dir: Path | None = None
    stage_names: tuple[str, ...] = ("ttir", "ttgir", "llir")
    label: str = "compilagent-stage-hook"
    replace_stages: tuple[str, ...] = ()
    """Stages to fully replace with our pass-by-pass executor (`ttir`, `ttgir`)."""
    interventions: tuple[PassIntervention, ...] = ()
    """Per-pass overrides applied when a stage is replaced."""

    @property
    def key(self) -> str:
        return f"{self.label}:{self.key_material}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class StageHookRecord:
    stage_name: str
    digest: str
    artifact_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


StageCallback = Callable[[str, Any, dict[str, Any]], None]
PassEventCallback = Callable[[str, PassResult], None]


def make_stage_inspection_hook(
    config: StageHookConfig,
    *,
    callback: StageCallback | None = None,
    records: list[StageHookRecord] | None = None,
    pass_callback: PassEventCallback | None = None,
):
    """Build a Triton-compatible `add_stages_inspection_hook` callable.

    Two modes:
    - default (observe): wraps the original stage function with timing/digest.
    - replace: when a stage name is in `config.replace_stages`, the upstream
      lambda is substituted by a `StagePipeline` that runs the stage pass-by-
      pass with optional `PassIntervention` overrides. Per-pass results stream
      through `pass_callback` if provided.
    """

    records = records if records is not None else []

    def inspect_stages_hook(
        self=None,
        stages=None,
        options=None,
        language=None,
        capability=None,
    ):
        if all(arg is None for arg in (self, stages, options, language, capability)):
            return config.key, config.digest
        if stages is None or self is None:
            return config.key, config.digest
        stage_names = tuple(stage for stage in config.stage_names if stage in stages)

        replace_set = {s for s in config.replace_stages if s in stages}
        if replace_set:
            num_warps = int(getattr(options, "num_warps", 4) or 4)
            num_stages = int(getattr(options, "num_stages", 3) or 3)
            num_ctas = int(getattr(options, "num_ctas", 1) or 1)
            interventions = {iv.pass_name: iv for iv in config.interventions}
            pipeline = StagePipeline(
                capability=int(capability),
                num_warps=num_warps,
                num_stages=num_stages,
                num_ctas=num_ctas,
                interventions=interventions,
                on_pass=pass_callback,
                capture_ir=True,
                dump_dir=(config.artifact_dir / "passes") if config.artifact_dir else None,
            )
            if "ttir" in replace_set:
                stages["ttir"] = lambda mod, metadata: pipeline.make_ttir(mod, metadata)
            if "ttgir" in replace_set:
                stages["ttgir"] = lambda mod, metadata: pipeline.make_ttgir(mod, metadata)

        for stage_name in stage_names:
            if stage_name in replace_set:
                continue  # already substituted with our executor
            original_stage = stages[stage_name]

            def make_stage_wrapper(src, metadata, *, _stage_name=stage_name, _original=original_stage):
                started = time.perf_counter()
                module = _original(src, metadata)
                elapsed_ms = (time.perf_counter() - started) * 1000
                artifact_path = _write_stage_artifact(config, _stage_name, module)
                record = StageHookRecord(
                    stage_name=_stage_name,
                    digest=config.digest,
                    artifact_path=artifact_path,
                    metadata={
                        "duration_ms": elapsed_ms,
                        "language": str(language),
                        "capability": capability,
                        "options": _safe_options(options),
                    },
                )
                records.append(record)
                if callback is not None:
                    callback(_stage_name, module, metadata)
                return module

            stages[stage_name] = make_stage_wrapper
        return config.key, config.digest

    return inspect_stages_hook


@contextmanager
def scoped_stage_hook(
    config: StageHookConfig,
    *,
    callback: StageCallback | None = None,
    records: list[StageHookRecord] | None = None,
    pass_callback: PassEventCallback | None = None,
) -> Iterator[list[StageHookRecord]]:
    """Install a Triton stage hook for one experiment and restore previous state."""

    try:
        from triton import knobs
    except ImportError as exc:
        raise RuntimeError("Triton is not importable; install it into the top-level env.") from exc

    record_list = records if records is not None else []
    previous_hook = knobs.runtime.add_stages_inspection_hook
    knobs.runtime.add_stages_inspection_hook = make_stage_inspection_hook(
        config,
        callback=callback,
        records=record_list,
        pass_callback=pass_callback,
    )
    try:
        yield record_list
    finally:
        knobs.runtime.add_stages_inspection_hook = previous_hook


def _write_stage_artifact(config: StageHookConfig, stage_name: str, module: Any) -> Path | None:
    if config.artifact_dir is None:
        return None
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = config.artifact_dir / f"{stage_name}-{config.digest[:12]}.mlir"
    path.write_text(str(module), encoding="utf-8")
    return path


def _safe_options(options: Any) -> dict[str, Any]:
    if options is None:
        return {}
    if hasattr(options, "__dict__"):
        return {
            key: _safe_value(value)
            for key, value in vars(options).items()
            if not key.lower().endswith("key")
        }
    return {"repr": repr(options)}


def _safe_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, tuple | list):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    return repr(value)
