from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import PassIntervention, PassResult
from .schemas import CompileArtifact, CompileRequest, CompileResult, KernelSpec
from .stages import StageHookConfig, scoped_stage_hook

_ALLOWED_DIAGNOSTIC_ENV = {"TRITON_DUMP_PTXAS_LOG", "TRITON_DUMP_MIR"}


@dataclass(slots=True)
class TritonCompileHarness:
    """Compile a Triton kernel under a `Plan` and capture per-stage artifacts.

    `artifact_dir` is required — the core session always passes one. The
    legacy fallback that consulted an `OptimizationWorkspace` was dropped
    so this internal type has no workspace dependency.
    """

    def compile_kernel(
        self,
        spec: KernelSpec,
        request: CompileRequest,
        *,
        artifact_dir: Path,
        use_stage_hook: bool = False,
        replace_stages: tuple[str, ...] = (),
        interventions: tuple[PassIntervention, ...] = (),
        pass_callback: Callable[[str, PassResult], None] | None = None,
    ) -> CompileResult:
        """Compile or warm up a Triton kernel and capture available artifacts.

        The harness supports kernels that expose a module-level function named
        by `KernelSpec.entrypoint`. If the function provides a custom
        `compilagent_compile(meta: dict[str, Any])` method, that hook is used.
        Otherwise, the function is called with no positional arguments and the
        request metadata as keyword arguments. GPU-heavy benchmark definitions
        can therefore hide launch/input setup inside their own compile hook.
        """

        if spec.path is None:
            return CompileResult(
                kernel_id=spec.id,
                candidate_id=request.candidate_id,
                ok=False,
                diagnostics="KernelSpec.path is required for compile execution.",
            )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env = {
            "TRITON_ALWAYS_COMPILE": "1" if request.force_recompile else "0",
            "TRITON_KERNEL_DUMP": "1" if request.dump_ir else "0",
            "TRITON_DUMP_DIR": str(artifact_dir),
            **_safe_diagnostic_env(request.diagnostic_env),
        }
        hook_config = (
            StageHookConfig(
                key_material=request.stage_hook_key or request.model_dump_json(exclude_none=True),
                artifact_dir=artifact_dir,
                replace_stages=tuple(replace_stages),
                interventions=tuple(interventions),
            )
            if use_stage_hook
            else None
        )
        stage_records = []
        try:
            with patched_environ(env):
                if hook_config is None:
                    handle = _execute_compile(spec, request.meta)
                else:
                    with scoped_stage_hook(hook_config, pass_callback=pass_callback) as records:
                        handle = _execute_compile(spec, request.meta)
                        stage_records = list(records)
            artifacts = _collect_artifacts(handle, artifact_dir)
            stage_metadata = [_stage_record_metadata(record) for record in stage_records]
            metadata = _extract_metadata(handle)
            if stage_metadata:
                metadata["stage_records"] = stage_metadata
            return CompileResult(
                kernel_id=spec.id,
                candidate_id=request.candidate_id,
                ok=True,
                source_hash=_source_hash(spec.path),
                metadata=metadata,
                artifacts=artifacts,
            )
        except Exception as exc:  # pragma: no cover - diagnostics path depends on Triton/GPU
            return CompileResult(
                kernel_id=spec.id,
                candidate_id=request.candidate_id,
                ok=False,
                source_hash=_source_hash(spec.path),
                diagnostics=f"{exc.__class__.__name__}: {exc}",
                artifacts=_collect_artifacts(None, artifact_dir),
            )


def _execute_compile(spec: KernelSpec, meta: dict[str, Any]) -> Any:
    module = _load_module(spec.path)
    entrypoint = getattr(module, spec.entrypoint)
    compile_hook = getattr(entrypoint, "compilagent_compile", None)
    if callable(compile_hook):
        return compile_hook(meta)
    return entrypoint(**meta)


def _load_module(path: Path) -> Any:
    """Load a kernel module by path.

    If the path lives inside an installed Python package, prefer
    `importlib.import_module(<package>.<name>)` so relative imports keep
    working. Otherwise fall back to spec_from_file_location with the parent
    directory placed on sys.path.
    """

    resolved = path.resolve()
    package_qualname = _resolve_package_module_name(resolved)
    if package_qualname is not None:
        return importlib.import_module(package_qualname)

    module_name = f"compilagent_kernel_{hashlib.sha256(str(resolved).encode()).hexdigest()[:12]}"
    parent = str(resolved.parent)
    added_to_path = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        added_to_path = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load kernel module: {resolved}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if added_to_path:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def _resolve_package_module_name(path: Path) -> str | None:
    """Return `pkg.sub.module` if `path` lives inside an importable package."""

    if path.suffix != ".py":
        return None
    parts: list[str] = [path.stem]
    parent = path.parent
    while (parent / "__init__.py").exists():
        parts.append(parent.name)
        parent = parent.parent
    if len(parts) == 1:
        return None  # not inside any package
    parts.reverse()
    return ".".join(parts)


def _collect_artifacts(handle: Any, artifact_dir: Path) -> list[CompileArtifact]:
    artifacts: list[CompileArtifact] = []
    asm = getattr(handle, "asm", None)
    if isinstance(asm, dict):
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            if stage in asm:
                artifacts.append(
                    CompileArtifact(
                        stage=stage,
                        inline_text=str(asm[stage]),
                        metadata={"source": "handle.asm"},
                    )
                )
    if artifact_dir.exists():
        for path in sorted(artifact_dir.rglob("*")):
            if path.is_file() and path.suffix in {".ttir", ".ttgir", ".llir", ".ptx", ".mlir"}:
                artifacts.append(
                    CompileArtifact(
                        stage=path.suffix.lstrip("."),
                        path=path,
                        metadata={
                            "source": "artifact_dir",
                            "size_bytes": path.stat().st_size,
                            "digest": _file_hash(path),
                        },
                    )
                )
    return artifacts


def _safe_diagnostic_env(values: dict[str, str]) -> dict[str, str]:
    return {key: str(value) for key, value in values.items() if key in _ALLOWED_DIAGNOSTIC_ENV}


def _stage_record_metadata(record: Any) -> dict[str, Any]:
    artifact_path = getattr(record, "artifact_path", None)
    return {
        "stage": getattr(record, "stage_name", "unknown"),
        "digest": getattr(record, "digest", None),
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        **_safe_metadata_value(getattr(record, "metadata", {})),
    }


def _extract_metadata(handle: Any) -> dict[str, Any]:
    metadata = getattr(handle, "metadata", None)
    if metadata is None:
        return {}
    if hasattr(metadata, "_asdict"):
        return {key: _safe_metadata_value(value) for key, value in metadata._asdict().items()}
    if isinstance(metadata, dict):
        return {str(key): _safe_metadata_value(value) for key, value in metadata.items()}
    return {"repr": repr(metadata)}


def _safe_metadata_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, tuple | list):
        return [_safe_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_metadata_value(item) for key, item in value.items()}
    return repr(value)


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def patched_environ(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
