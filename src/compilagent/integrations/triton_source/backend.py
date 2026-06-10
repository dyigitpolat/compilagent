"""`TritonSourceBackend` — the core `Backend` protocol over a KERNEL-SOURCE
decision space (ticket E1).

Unlike the `triton` backend (whose decision space is the MLIR pass pipeline
of an *unchanged* kernel), this backend's whole decision space is the kernel
source itself: the agent replaces the reference PyTorch module with a
self-contained python module defining `ModelNew` whose forward path runs
`@triton.jit` kernels.

Workload contract (`WorkloadSpec.metadata`):

  - ``reference_module_source`` (required): self-contained python source —
    `Model` nn.Module (no-arg `__init__`) + `get_inputs()` factory returning
    CUDA tensors (KernelBench L1 style).
  - ``banned_patterns``: per-workload banned-API list for gate g4 (see
    `_internal.lint`). E.g. `["torch.softmax", "softmax"]`.
  - ``input_shapes`` / ``input_dtypes`` / ``op_signature``: cheap analysis
    metadata recorded by `analyze()`.
  - ``sandbox_timeout_seconds`` (optional, default 240): hard subprocess
    timeout.
  - ``sandbox_warmup`` / ``sandbox_repetitions`` (optional, defaults
    25 / 100): the CUDA-event timing protocol inside the sandbox.

Intervention vocabulary: a single kind, ``source_replace``, payload
``{"module_source": "<full python module defining ModelNew>"}``. When a plan
carries several `source_replace` interventions the LAST one wins (a candidate
is one module).

Evaluation & caching: `compile()` runs the entire E2a evaluation in one
subprocess sandbox (gates g1/g2/g3/g5 + CUDA-event timing of candidate AND
reference; gate g4 — the banned-API AST lint — runs in-parent *before* the
subprocess is spawned, so a linted-out candidate costs zero GPU seconds).
The parsed sandbox result is cached on the backend keyed by the plan's
source hash AND stored in ``CompileResult.metadata["evaluation"]``, so
`time_workload` and `validate_correctness` serve from that cache WITHOUT
re-running anything on the GPU. The baseline plan (empty `Plan()`) times the
reference module itself in the same sandbox.

Budget-ledger note (E4): the sandbox only runs timing reps when every gate
passed, so "a timing invocation only happens for gate-passing candidates"
holds by construction — `time_workload` for a gate-failing candidate returns
an empty `TimingResult` straight from the cache.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from compilagent.core.analysis import (
    Analysis,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    PassCallback,
    PassEvent,
    TimingResult,
)
from compilagent.core.backend import BackendBase
from compilagent.core.plan import Intervention, Plan, ValidationResult
from compilagent.core.search_space import (
    DerivationEvidence,
    Lever,
    SearchSpace,
    StructuredJsonRange,
)
from compilagent.core.tool_decl import ToolDecl
from compilagent.core.workload import ToleranceConfig, WorkloadSpec

from ._internal.gpu_lease import pool_devices
from ._internal.lint import lint_banned_apis
from ._internal.sandbox import DEFAULT_TIMEOUT_SECONDS, run_sandboxed_eval
from ._internal.sandbox_runner import DEFAULT_REPETITIONS, DEFAULT_WARMUP

_VALID_INTERVENTION_KINDS = frozenset({"source_replace"})

_GATE_ORDER = (
    "g1_shape_dtype",
    "g2_allclose",
    "g3_no_alias_no_mutation",
    "g4_banned_api",
    "g5_determinism",
)


def _percentile(samples: Sequence[float], q: float) -> float | None:
    if not samples:
        return None
    srt = sorted(samples)
    idx = min(len(srt) - 1, max(0, int(round(q * (len(srt) - 1)))))
    return srt[idx]


class TritonSourceBackend(BackendBase):
    """Kernel-source backend: the agent's lever is the module source itself."""

    id: str = "triton_source"
    artifact_stages: tuple[str, ...] = ("module_source",)

    def __init__(self) -> None:
        # plan-key → parsed sandbox result. One backend instance lives for
        # one session, so the cache is naturally per-run.
        self._eval_cache: dict[str, dict[str, Any]] = {}
        # Most recent spec seen by compile/analyze — lets the no-arg
        # `read_reference_source` tool default to the active workload.
        self._last_spec: WorkloadSpec | None = None

    # ---- device --------------------------------------------------------------

    def device_capability(self) -> DeviceCapability:
        cap_int: int | None = None
        name = "cpu"
        mem_total: int | None = None
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                index = self._capability_probe_index(torch.cuda.device_count())
                major, minor = torch.cuda.get_device_capability(index)
                cap_int = major * 10 + minor
                props = torch.cuda.get_device_properties(index)
                name = props.name
                mem_total = int(getattr(props, "total_memory", 0) or 0)
        except Exception:  # noqa: BLE001
            pass
        arch = f"cuda:sm_{cap_int}" if cap_int is not None else "cpu"
        return DeviceCapability(
            arch=arch,
            capability_int=cap_int,
            name=name,
            memory_total_bytes=mem_total,
            memory_peak_bandwidth_gbps=None,
        )

    @staticmethod
    def _capability_probe_index(device_count: int) -> int:
        """Which CUDA index to probe for capability reporting (read-only,
        context-free). In GPU-pool mode the parent process is unpinned, so
        index 0 may not even belong to the pool — probe the first pool
        device instead (pool entries are physical indices, valid exactly
        when no `CUDA_VISIBLE_DEVICES` remapping is in effect)."""

        pool = pool_devices()
        if pool and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            try:
                index = int(pool[0])
            except ValueError:
                return 0
            if 0 <= index < device_count:
                return index
        return 0

    # ---- analysis (cheap, no GPU) --------------------------------------------

    def analyze(
        self,
        workload: WorkloadSpec,
        *,
        baseline_artifacts: Sequence[Path],
    ) -> Analysis:
        self._last_spec = workload
        meta = workload.metadata
        reference = str(meta.get("reference_module_source", "") or "")
        summary: dict[str, Any] = {
            "kind": workload.kind.value,
            "tensor_shapes": dict(meta.get("input_shapes", {}) or {}),
            "dtypes": list(meta.get("input_dtypes", []) or []),
            "op_counts": {},
            "op_signature": str(meta.get("op_signature", "") or workload.description),
            "reference_line_count": reference.count("\n") + 1 if reference else 0,
        }
        return Analysis(
            summary=summary,
            extra={
                "banned_patterns": list(meta.get("banned_patterns", []) or []),
            },
        )

    # ---- search space ---------------------------------------------------------

    def derive_search_space(
        self,
        workload: WorkloadSpec,
        analysis: Analysis,
    ) -> SearchSpace:
        reference = str(
            workload.metadata.get("reference_module_source", "") or ""
        )
        if not reference:
            return SearchSpace(workload_id=workload.id, backend_id=self.id)
        summary = analysis.summary or {}
        shapes = summary.get("tensor_shapes") or {}
        dtypes = summary.get("dtypes") or []
        banned = list(workload.metadata.get("banned_patterns", []) or [])
        evidence = DerivationEvidence(
            rule="triton_source.reference_module",
            signal=(
                f"reference implementation "
                f"({summary.get('reference_line_count', 0)} lines, op: "
                f"{summary.get('op_signature', '?')}) with input shapes "
                f"{shapes} and dtypes {dtypes}"
            ),
            citations=(),
        )
        lever = Lever(
            id="kernel_source",
            target_kind="source_replace",
            target_selector="kernel_source",
            range=StructuredJsonRange(
                examples=(
                    {
                        "module_source": (
                            "<complete python module defining `ModelNew`, a "
                            "drop-in replacement for the reference `Model` "
                            "with the same __init__/forward signatures, whose "
                            "forward path runs @triton.jit kernels you write>"
                        )
                    },
                ),
                schema_hint=(
                    'payload = {"module_source": str}. The module must be '
                    "self-contained (its own imports), define `ModelNew`, "
                    "keep identical parameter names/shapes so state_dict "
                    "copies, and must NOT call the banned torch APIs: "
                    f"{banned}."
                ),
            ),
            default=None,
            description=(
                "Free-form kernel source. Replace the reference PyTorch "
                "module with a custom Triton implementation (`ModelNew`)."
            ),
            evidence=evidence,
            backend_id=self.id,
        )
        return SearchSpace(
            workload_id=workload.id, backend_id=self.id, levers=(lever,)
        )

    # ---- intervention surface --------------------------------------------------

    def validate_intervention(self, intervention: Intervention) -> ValidationResult:
        kind = intervention.target.kind
        if kind not in _VALID_INTERVENTION_KINDS:
            return ValidationResult(
                ok=False,
                errors=(
                    f"unsupported target.kind `{kind}`; the triton_source "
                    f"backend accepts only {sorted(_VALID_INTERVENTION_KINDS)} "
                    'with payload {"module_source": "<python module defining '
                    'ModelNew>"}',
                ),
            )
        payload = intervention.payload
        if not isinstance(payload, dict) or not str(
            payload.get("module_source", "") or ""
        ).strip():
            return ValidationResult(
                ok=False,
                errors=(
                    "source_replace payload must be a dict with a non-empty "
                    "`module_source` string (the full python module).",
                ),
            )
        source = str(payload["module_source"])
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return ValidationResult(
                ok=False,
                errors=(
                    f"module_source has a syntax error at line {exc.lineno}: "
                    f"{exc.msg}",
                ),
            )
        has_model_new = any(
            isinstance(node, ast.ClassDef) and node.name == "ModelNew"
            for node in tree.body
        )
        if not has_model_new:
            return ValidationResult(
                ok=False,
                errors=(
                    "module_source must define a top-level class `ModelNew` "
                    "(drop-in replacement for the reference `Model`).",
                ),
            )
        return ValidationResult(ok=True)

    # ---- compile (sandboxed evaluation) ----------------------------------------

    def _candidate_source(self, plan: Plan) -> str | None:
        """Extract the candidate module source; last `source_replace` wins."""

        source: str | None = None
        for iv in plan.interventions:
            if iv.target.kind == "source_replace" and isinstance(iv.payload, dict):
                value = str(iv.payload.get("module_source", "") or "")
                if value.strip():
                    source = value
        return source

    def _plan_key(self, workload: WorkloadSpec, plan: Plan) -> str:
        source = self._candidate_source(plan)
        if source is None:
            return f"{workload.id}::baseline"
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        return f"{workload.id}::{digest}"

    def compile(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        artifact_dir: Path,
        pass_callback: PassCallback | None = None,
    ) -> CompileResult:
        self._last_spec = workload
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        reference = str(
            workload.metadata.get("reference_module_source", "") or ""
        )
        if not reference:
            return CompileResult(
                ok=False,
                diagnostics=(
                    "workload.metadata is missing `reference_module_source` — "
                    "required for the triton_source backend."
                ),
            )

        candidate = self._candidate_source(plan)
        if not plan.is_empty and candidate is None:
            return CompileResult(
                ok=False,
                diagnostics=(
                    "plan carries no `source_replace` intervention with a "
                    "non-empty `module_source` payload."
                ),
            )

        # Artifacts: the source is the only compile artifact of this backend.
        ref_path = artifact_dir / "reference_module.py"
        ref_path.write_text(reference, encoding="utf-8")
        artifacts: list[Path] = [ref_path]
        if candidate is not None:
            cand_path = artifact_dir / "candidate_module.py"
            cand_path.write_text(candidate, encoding="utf-8")
            artifacts.append(cand_path)

        gates: dict[str, Any] = {}

        # Gate g4 — banned-API AST lint, in-parent, before any GPU work.
        if candidate is not None:
            banned = list(workload.metadata.get("banned_patterns", []) or [])
            violations = lint_banned_apis(candidate, banned)
            if violations:
                message = " ".join(v.message for v in violations[:3])
                gates["g4_banned_api"] = {"ok": False, "message": message}
                self._emit_pass(pass_callback, "lint", "banned_api_lint", error=message)
                result = {
                    "compiled": False,
                    "gates": gates,
                    "error": f"banned-API lint failed: {message}",
                }
                self._eval_cache[self._plan_key(workload, plan)] = result
                return CompileResult(
                    ok=False,
                    artifacts=tuple(artifacts),
                    diagnostics=result["error"],
                    metadata={"evaluation": result, "gates": gates},
                )
            gates["g4_banned_api"] = {
                "ok": True,
                "message": f"no banned-API calls (patterns: {banned}).",
            }
            self._emit_pass(pass_callback, "lint", "banned_api_lint")

        # Sandboxed evaluation: gates g1/g2/g3/g5 + CUDA-event timing.
        meta = workload.metadata
        result = run_sandboxed_eval(
            reference_source=reference,
            candidate_source=candidate,
            artifact_dir=artifact_dir,
            atol=workload.tolerance.atol,
            rtol=workload.tolerance.rtol,
            timeout_seconds=float(
                meta.get("sandbox_timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
            ),
            warmup=int(meta.get("sandbox_warmup", DEFAULT_WARMUP)),
            repetitions=int(meta.get("sandbox_repetitions", DEFAULT_REPETITIONS)),
        )
        result["gates"] = {**gates, **(result.get("gates") or {})}
        self._emit_pass(
            pass_callback,
            "sandbox",
            "subprocess_eval",
            error=result.get("error"),
        )

        # Cache the parsed result so time_workload / validate_correctness can
        # serve it WITHOUT re-running the subprocess.
        self._eval_cache[self._plan_key(workload, plan)] = result

        ok = bool(result.get("compiled")) and not result.get("error")
        # Surface failing-gate verdicts through `CompileResult.warnings`:
        # `run_candidate`'s tool result includes `compile_warnings`, which is
        # the only session-provided channel that reaches the agent verbatim —
        # harnesses build "failing gate + message" feedback from it.
        gate_warnings = tuple(
            f"{name} FAILED: {verdict.get('message', '')}"
            for name in _GATE_ORDER
            for verdict in (result["gates"].get(name),)
            if verdict is not None and not verdict.get("ok", False)
        )
        return CompileResult(
            ok=ok,
            elapsed_ms=None,
            artifacts=tuple(artifacts),
            compiled_callable=None,
            diagnostics=result.get("error"),
            warnings=tuple(result.get("warnings") or ()) + gate_warnings,
            metadata={
                "evaluation": result,
                "gates": result["gates"],
                "timed_out": bool(result.get("timed_out")),
            },
        )

    @staticmethod
    def _emit_pass(
        pass_callback: PassCallback | None,
        stage: str,
        name: str,
        *,
        error: str | None = None,
    ) -> None:
        if pass_callback is None:
            return
        try:
            pass_callback(
                PassEvent(
                    stage=stage,
                    name=name,
                    duration_ms=0.0,
                    action="run",
                    error=error,
                )
            )
        except Exception:  # noqa: BLE001
            pass

    # ---- timing (served from the cached sandbox result) ------------------------

    def time_workload(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        warmup: int,
        repetitions: int,
        max_seconds: float | None = None,
    ) -> TimingResult:
        """Serve timing from the sandbox result cached by `compile`.

        The `warmup`/`repetitions` arguments are intentionally ignored: the
        sandbox already measured with its own fixed CUDA-event protocol
        (25 warmup / 100 reps / 10% trimmed mean by default) in the same
        process that passed the gates, and re-running would double-charge
        the hardware budget.
        """

        key = self._plan_key(workload, plan)
        result = self._eval_cache.get(key)
        if result is None:
            return TimingResult(
                timings_ms=(),
                median_ms=None,
                p20_ms=None,
                p80_ms=None,
                diagnostics=(
                    "no cached sandbox evaluation for this plan — "
                    "compile() must run first."
                ),
            )

        is_baseline = plan.is_empty or self._candidate_source(plan) is None
        times = result.get("ref_times_ms" if is_baseline else "cand_times_ms")
        if not times:
            gates = result.get("gates") or {}
            failing = [k for k in _GATE_ORDER if not gates.get(k, {}).get("ok", True)]
            return TimingResult(
                timings_ms=(),
                median_ms=None,
                p20_ms=None,
                p80_ms=None,
                diagnostics=(
                    result.get("error")
                    or (
                        f"candidate failed gate(s) {failing} and was NOT "
                        "timed (budget ledger: timing invocations are "
                        "charged only for gate-passing candidates)."
                    )
                ),
            )
        times = tuple(float(t) for t in times)
        return TimingResult(
            timings_ms=times,
            median_ms=_percentile(times, 0.5),
            p20_ms=_percentile(times, 0.2),
            p80_ms=_percentile(times, 0.8),
            profile_metrics={
                "trimmed_mean_ms": result.get(
                    "ref_ms" if is_baseline else "cand_ms"
                ),
                "ref_trimmed_mean_ms": result.get("ref_ms"),
                "speedup_vs_ref_in_sandbox": result.get("speedup_vs_ref"),
            },
        )

    # ---- correctness (served from the cached sandbox result) -------------------

    def validate_correctness(
        self,
        workload: WorkloadSpec,
        baseline: CompileResult,
        candidate: CompileResult,
        tolerance: ToleranceConfig,
    ) -> CorrectnessResult:
        """Per-gate verdicts come straight from `candidate.metadata` — the
        sandbox already ran all E2a gates during `compile`; nothing is
        re-executed here."""

        evaluation = (candidate.metadata or {}).get("evaluation")
        if not isinstance(evaluation, dict):
            return CorrectnessResult(
                ok=False,
                diagnostics=(
                    "no cached sandbox evaluation on the candidate "
                    "CompileResult — compile() must produce it."
                ),
            )
        gates: dict[str, Any] = evaluation.get("gates") or {}
        failing = [
            (k, gates[k].get("message", ""))
            for k in _GATE_ORDER
            if k in gates and not gates[k].get("ok", False)
        ]
        ok = bool(gates) and not failing
        diagnostics = (
            "all E2a gates passed (5 value-randomized trials, seeds 100-104)."
            if ok
            else "; ".join(f"{k}: {msg}" for k, msg in failing)
            or "sandbox returned no gate verdicts."
        )
        return CorrectnessResult(
            ok=ok,
            max_abs_diff=evaluation.get("max_abs_diff"),
            max_rel_diff=evaluation.get("max_rel_diff"),
            failed_at=failing[0][0] if failing else None,
            diagnostics=diagnostics,
            metadata={"gates": gates},
        )

    # ---- introspection ----------------------------------------------------------

    def list_introspection_tools(self) -> Sequence[ToolDecl]:
        backend = self

        def read_reference_source(workload_id: str = "") -> str:
            """Return the reference PyTorch module source for a workload."""

            spec: WorkloadSpec | None
            if workload_id:
                from compilagent.core.workload_registry import workload_registry

                spec = workload_registry.get_spec(workload_id)
            else:
                spec = backend._last_spec
            if spec is None:
                raise ValueError(
                    "no active workload; pass `workload_id` explicitly."
                )
            return json.dumps(
                {
                    "workload_id": spec.id,
                    "task_description": spec.description,
                    "reference_module_source": str(
                        spec.metadata.get("reference_module_source", "") or ""
                    ),
                    "banned_patterns": list(
                        spec.metadata.get("banned_patterns", []) or []
                    ),
                },
                indent=2,
            )

        return (
            ToolDecl(
                name="read_reference_source",
                description=(
                    "Return the reference PyTorch module source (`Model` + "
                    "`get_inputs()`), the task description, and the banned "
                    "torch APIs for the active workload. `workload_id` is "
                    "optional and defaults to the session's workload."
                ),
                args_schema={
                    "type": "object",
                    "properties": {
                        "workload_id": {
                            "type": "string",
                            "description": (
                                "Workload id; empty string → active workload."
                            ),
                            "default": "",
                        },
                    },
                    "additionalProperties": False,
                },
                handler=read_reference_source,
                read_only=True,
            ),
        )

    def infer_workload_family(self, workload: WorkloadSpec) -> str | None:
        text = f"{workload.id} {workload.title} {workload.description}".lower()
        if "matmul" in text or "gemm" in text:
            return "matmul"
        if "norm" in text:
            return "norm"
        if "softmax" in text:
            return "reduction"
        if "gelu" in text or "swish" in text or "silu" in text:
            return "elementwise"
        return None
