"""KernelBench → `triton_source` adapter: 24-task selection + conversion (D8).

Selects 12 level-1 + 12 level-2 KernelBench tasks, stratified by op family,
and converts each into a `triton_source` workload entry (no-arg `Model` +
CUDA fp32 `get_inputs`, banned-API patterns derived from the ops used).
Exclusion criteria, every one recorded in the JSON manifest:

  - ``cudnn_conv_bound``     — Conv* tasks (the PyTorch reference dispatches
    to cuDNN fast paths a Triton rewrite cannot meaningfully race).
  - ``inherently_inefficient_baseline`` — robust-kbench-style static check:
    the reference (or its input construction) materializes structure via
    ``torch.diag`` / ``tril`` / ``triu``, so "speedups" come from fixing the
    baseline, not from kernel skill.
  - ``adapter_unparseable_inputs`` — `get_inputs` is not a plain list of
    ``torch.rand/randn`` tensors with symbolic shapes (scalars, boolean
    masks, post-hoc symmetrization, ...). The D10 holdout generator needs a
    symbolic shape template, so these tasks cannot join the suite.
  - ``contaminated``         — robust-kbench dynamic criteria, measured by
    running the reference ONCE on a leased GPU (`gpu_lease`, pool from
    ``COMPILAGENT_GPU_POOL``, default "1"): all reference outputs inside
    [-0.01, 0.01]; output std < 0.01; input-impact < 0.01 (outputs barely
    move when inputs are re-randomized); near-identity (eval-mode reference
    returns its input, e.g. fresh BatchNorm running stats).
  - ``probe_error``          — the reference itself failed to run.

Selection among survivors is deterministic: round-robin over op families
(largest family first), ascending KernelBench index inside a family.

Usage (probes lease GPU 1 via the pool; never pins a busy device):

    COMPILAGENT_GPU_POOL=1 env/bin/python -m scripts.kernelbench_select \
        --kb-root ../baselines/kernelbench/KernelBench \
        --out src/compilagent/integrations/triton_source/kernelbench_manifest.json

The probe child (``--probe-child``) is this same module re-executed inside a
``CUDA_VISIBLE_DEVICES=<leased device>`` subprocess — the parent never
touches CUDA.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_KB_ROOT = REPO_ROOT.parent / "baselines" / "kernelbench" / "KernelBench"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "src"
    / "compilagent"
    / "integrations"
    / "triton_source"
    / "kernelbench_manifest.json"
)
DEFAULT_PROBE_CACHE = REPO_ROOT / "scripts" / "results" / "kb_probe_cache.json"

PROBE_MARKER = "KB_PROBE_JSON:"

#: robust-kbench-style contamination thresholds (see module docstring).
CONTAMINATION_THRESHOLDS = {
    "out_abs_max_max": 0.01,
    "out_std_min": 0.01,
    "input_impact_min": 0.01,
    "near_identity_atol": 1e-3,
}

#: Constructions that make the baseline inherently inefficient (dense
#: materialization of structure the task then throws away / exploits).
_INEFFICIENT_CONSTRUCTIONS = ("torch.diag", ".tril(", ".triu(", "torch.tril", "torch.triu")

# --------------------------------------------------------------- task scanning


@dataclass(frozen=True, slots=True)
class KbTask:
    """One KernelBench task file."""

    level: int
    index: int
    name: str  # e.g. "Gemm_Multiply_LeakyReLU"
    path: Path
    source: str

    @property
    def slug(self) -> str:
        slug = re.sub(r"[^0-9a-z]+", "_", self.name.lower()).strip("_")
        return re.sub(r"_+", "_", slug)

    @property
    def workload_id(self) -> str:
        return f"kb{self.level}_{self.index}_{self.slug}"

    @property
    def kb_file(self) -> str:
        return f"level{self.level}/{self.path.name}"


_FILE_RE = re.compile(r"^(\d+)_(.+)\.py$")


def scan_tasks(kb_root: Path, level: int) -> list[KbTask]:
    tasks: list[KbTask] = []
    for path in sorted((kb_root / f"level{level}").glob("*.py")):
        match = _FILE_RE.match(path.name)
        if not match:
            continue
        tasks.append(
            KbTask(
                level=level,
                index=int(match.group(1)),
                name=match.group(2).strip("_"),
                path=path,
                source=path.read_text(encoding="utf-8"),
            )
        )
    tasks.sort(key=lambda t: t.index)
    return tasks


# ------------------------------------------------------------- static checks


def conv_exclusion(task: KbTask) -> str | None:
    """Conv* tasks are cuDNN-bound — excluded wholesale (ticket D8)."""

    if "conv" in task.name.lower() or re.search(r"\bnn\.Conv|\bconv\d?d\(", task.source):
        return "cudnn_conv_bound: Conv reference dispatches to cuDNN fast paths"
    return None


def inefficient_exclusion(task: KbTask) -> str | None:
    """robust-kbench: baselines that materialize structure (diag/tril/triu)
    reward fixing the reference, not writing a faster kernel."""

    hits = sorted({p for p in _INEFFICIENT_CONSTRUCTIONS if p in task.source})
    if hits:
        return f"inherently_inefficient_baseline: uses {', '.join(hits)}"
    return None


# ------------------------------------------------------- shape-template parse


class TemplateError(ValueError):
    """`get_inputs` cannot be expressed as a symbolic shape template."""


@dataclass(slots=True)
class ShapeTemplate:
    """Symbolic input shapes: per-input factory + dim variable names, plus a
    var table. Vars referenced by `get_init_inputs` parameterize the *model*
    (parameter shapes) and are FIXED; the rest are free data axes the D10
    holdout generator may vary."""

    inputs: list[dict[str, Any]] = field(default_factory=list)
    vars: dict[str, dict[str, Any]] = field(default_factory=dict)

    def input_shapes(self) -> dict[str, list[int]]:
        return {
            entry["name"]: [int(self.vars[d]["size"]) for d in entry["dims"]]
            for entry in self.inputs
        }


def _const_int(node: ast.AST) -> int | None:
    """Fold a constant-int expression (e.g. ``16384 * 4``); None otherwise."""

    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp):
        left, right = _const_int(node.left), _const_int(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.FloorDiv) and right != 0:
            return left // right
    return None


def _module_constants(tree: ast.Module) -> dict[str, Any]:
    """Module-level `name = <int or tuple-of-int>` assignments."""

    consts: dict[str, Any] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _const_int(node.value)
        if value is not None:
            consts[target.id] = value
            continue
        if isinstance(node.value, ast.Tuple):
            elements = [_const_int(e) for e in node.value.elts]
            if all(e is not None for e in elements):
                consts[target.id] = tuple(elements)
    return consts


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _forward_arg_names(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    args = item.args
                    if args.vararg or args.kwonlyargs or args.kwarg:
                        raise TemplateError("forward uses *args/**kwargs")
                    return [a.arg for a in args.args[1:]]  # skip self
    raise TemplateError("no Model.forward found")


def _init_input_names(tree: ast.Module) -> set[str]:
    fn = _find_function(tree, "get_init_inputs")
    if fn is None:
        return set()
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}


def _tensor_factory(call: ast.Call, consts: dict[str, Any]) -> dict[str, Any]:
    """Interpret one ``torch.rand|randn(...)`` call → {factory, dims}."""

    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
        and func.attr in {"rand", "randn"}
    ):
        raise TemplateError("input is not a plain torch.rand/randn call")
    if call.keywords:
        raise TemplateError("torch.rand/randn call uses keyword arguments")
    dims: list[str] = []
    for arg in call.args:
        if isinstance(arg, ast.Name):
            size = consts.get(arg.id)
            if not isinstance(size, int):
                raise TemplateError(f"shape var `{arg.id}` is not a module-level int")
            dims.append(arg.id)
        elif isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name):
            shape = consts.get(arg.value.id)
            if not isinstance(shape, tuple):
                raise TemplateError(
                    f"starred shape `{arg.value.id}` is not a module-level tuple"
                )
            dims.extend(f"{arg.value.id}_{i}" for i in range(len(shape)))
        elif (value := _const_int(arg)) is not None:
            dims.append(f"c{value}")
        else:
            raise TemplateError("unsupported shape expression in torch.rand/randn")
    if not dims:
        raise TemplateError("0-d tensor input")
    return {"factory": func.attr, "dims": dims}


def parse_shape_template(source: str) -> ShapeTemplate:
    """Parse a KB task's `get_inputs` into a symbolic shape template.

    Supported form: optional ``name = torch.rand|randn(<vars/ints>)``
    assignments followed by ``return [<name or factory call>, ...]``.
    Anything else (scalars, masks, arithmetic on tensors) raises
    `TemplateError` → the task is excluded with reason
    ``adapter_unparseable_inputs``.
    """

    tree = ast.parse(source)
    consts = _module_constants(tree)
    fn = _find_function(tree, "get_inputs")
    if fn is None:
        raise TemplateError("no get_inputs function")

    assigned: dict[str, dict[str, Any]] = {}
    returned: list[dict[str, Any]] | None = None
    for stmt in fn.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # docstring
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
                assigned[target.id] = _tensor_factory(stmt.value, consts)
                continue
            raise TemplateError("unsupported assignment in get_inputs")
        if isinstance(stmt, ast.Return):
            if not isinstance(stmt.value, ast.List):
                raise TemplateError("get_inputs does not return a list literal")
            returned = []
            for element in stmt.value.elts:
                if isinstance(element, ast.Name):
                    if element.id not in assigned:
                        raise TemplateError(
                            f"returned name `{element.id}` is not a tensor assignment"
                        )
                    returned.append(assigned[element.id])
                elif isinstance(element, ast.Call):
                    returned.append(_tensor_factory(element, consts))
                else:
                    raise TemplateError("returned element is not a tensor")
            break
        raise TemplateError("unsupported statement in get_inputs")
    if returned is None:
        raise TemplateError("get_inputs has no return statement")

    arg_names = _forward_arg_names(tree)
    if len(arg_names) != len(returned):
        raise TemplateError(
            f"forward takes {len(arg_names)} args but get_inputs returns "
            f"{len(returned)} tensors"
        )

    init_names = _init_input_names(tree)
    template = ShapeTemplate()
    for name, entry in zip(arg_names, returned, strict=True):
        template.inputs.append({"name": name, **entry})
        for dim in entry["dims"]:
            if dim in template.vars:
                continue
            if dim.startswith("c") and dim[1:].isdigit():
                size: int = int(dim[1:])
                free = False  # literal axes: provenance unknown → keep fixed
            elif (base := re.match(r"(.+)_(\d+)$", dim)) and base.group(1) in consts:
                size = int(consts[base.group(1)][int(base.group(2))])
                free = base.group(1) not in init_names
            else:
                size = int(consts[dim])
                free = dim not in init_names
            if size < 1:
                raise TemplateError(f"axis `{dim}` has non-positive size {size}")
            template.vars[dim] = {"size": size, "free": free}
    return template


# ----------------------------------------------------------------- conversion


def adapt_reference_source(kb_source: str, header: str = "") -> str:
    """KB module → sandbox-contract reference module.

    The KB `Model.__init__` takes init args and `get_inputs` returns CPU
    tensors; the `triton_source` sandbox wants a no-arg `Model` and CUDA fp32
    inputs. Renaming (`Model` → `_KBModel`, `get_inputs` → `_kb_get_inputs`)
    keeps internal references (`super(_KBModel, self)`) consistent, and the
    subclass shim preserves parameter names so `load_state_dict` copies.
    """

    renamed = re.sub(r"\bModel\b", "_KBModel", kb_source)
    renamed = re.sub(r"\bget_inputs\b", "_kb_get_inputs", renamed)
    shim = textwrap.dedent(
        """
        # --- compilagent triton_source adapter shim (ticket D8) ---

        class Model(_KBModel):
            def __init__(self):
                super().__init__(*get_init_inputs())

        def get_inputs():
            return [
                t.cuda().float() if isinstance(t, torch.Tensor) else t
                for t in _kb_get_inputs()
            ]
        """
    ).strip()
    prefix = f"# {header}\n" if header else ""
    return f"{prefix}{renamed.rstrip()}\n\n{shim}\n"


# --------------------------------------------------------------- op families

_NORM_TOKENS = {"GroupNorm", "BatchNorm", "InstanceNorm", "LayerNorm", "RMSNorm"}
_REDUCTION_TOKENS = {
    "Sum", "Max", "Min", "Mean", "LogSumExp", "Softmax", "LogSoftmax",
    "GlobalAvgPool", "Argmax", "Argmin",
}
_POOL_TOKENS = {"MaxPool", "AvgPool"}
_ACTIVATION_TOKENS = {
    "ReLU", "LeakyReLU", "GELU", "Swish", "SiLU", "Sigmoid", "HardSigmoid",
    "Tanh", "HardTanh", "Hardtanh", "Mish", "ELU", "SELU", "Softplus",
    "Softsign", "HardSwish", "MinGPTNewGelu", "Activation",
}


def _name_tokens(name: str) -> list[str]:
    return [t for t in name.split("_") if t]


def op_family(task: KbTask) -> str:
    """Deterministic op-family label used for stratified selection."""

    lower = task.name.lower()
    tokens = set(_name_tokens(task.name))
    if task.level == 1:
        if "attention" in lower:
            return "attention"
        if "loss" in lower:
            return "loss"
        if "cumsum" in lower or "cumprod" in lower:
            return "scan"
        if "pool" in lower:
            return "pooling"
        if "softmax" in lower or tokens & {"Argmax", "Argmin"} or "reduction" in lower:
            return "reduction"
        if "norm" in lower:
            return "norm"
        if "matmul" in lower or "matrix" in lower or "gemm" in lower or "bmm" in lower:
            return "matmul"
        return "activation"
    # Level 2: everything non-conv is GEMM/Matmul/BMM + epilogue; stratify by
    # the heaviest epilogue ingredient (norm > reduction > pool > activation).
    if tokens & _NORM_TOKENS:
        return "gemm+norm"
    if tokens & _REDUCTION_TOKENS:
        return "gemm+reduction"
    if tokens & _POOL_TOKENS:
        return "gemm+pool"
    if tokens & _ACTIVATION_TOKENS:
        return "gemm+activation"
    return "gemm+pointwise"


# ----------------------------------------------------------- banned patterns

_MATMUL_BANNED = ["matmul", "mm", "bmm", "einsum", "@", "linear", "Linear"]

#: filename token → banned patterns for gate g4 (terminal-segment matching in
#: `_internal.lint`; `__init__` bodies and `tl.*` are exempt there).
_TOKEN_BANNED: dict[str, list[str]] = {
    "Gemm": _MATMUL_BANNED,
    "Matmul": _MATMUL_BANNED,
    "BMM": _MATMUL_BANNED,
    "matmul": _MATMUL_BANNED,
    "matrix": _MATMUL_BANNED,
    "multiplication": _MATMUL_BANNED,
    "Softmax": ["softmax", "log_softmax", "Softmax"],
    "LogSoftmax": ["log_softmax", "softmax", "LogSoftmax", "Softmax"],
    "ReLU": ["relu", "relu_", "ReLU"],
    "LeakyReLU": ["leaky_relu", "LeakyReLU"],
    "GELU": ["gelu", "GELU"],
    "MinGPTNewGelu": ["gelu", "GELU", "tanh", "Tanh"],
    "Sigmoid": ["sigmoid", "Sigmoid"],
    "HardSigmoid": ["hardsigmoid", "Hardsigmoid"],
    "Swish": ["silu", "SiLU", "sigmoid", "Sigmoid"],
    "SiLU": ["silu", "SiLU", "sigmoid", "Sigmoid"],
    "Tanh": ["tanh", "Tanh"],
    "HardTanh": ["hardtanh", "Hardtanh"],
    "Hardtanh": ["hardtanh", "Hardtanh"],
    "Mish": ["mish", "Mish"],
    "ELU": ["elu", "ELU"],
    "SELU": ["selu", "SELU"],
    "Softplus": ["softplus", "Softplus"],
    "Softsign": ["softsign", "Softsign"],
    "HardSwish": ["hardswish", "Hardswish"],
    "LayerNorm": ["layer_norm", "LayerNorm", "native_layer_norm"],
    "RMSNorm": ["rms_norm", "RMSNorm"],
    "GroupNorm": ["group_norm", "GroupNorm"],
    "BatchNorm": ["batch_norm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d"],
    "InstanceNorm": ["instance_norm", "InstanceNorm1d", "InstanceNorm2d"],
    "FrobeniusNorm": ["norm", "vector_norm", "matrix_norm", "normalize"],
    "L1Norm": ["norm", "vector_norm", "normalize"],
    "L2Norm": ["norm", "vector_norm", "normalize"],
    "Sum": ["sum"],
    "Mean": ["mean"],
    "Max": ["max", "amax"],
    "Min": ["min", "amin"],
    "LogSumExp": ["logsumexp"],
    "Argmax": ["argmax"],
    "Argmin": ["argmin"],
    "MaxPool": ["max_pool1d", "max_pool2d", "max_pool3d", "MaxPool1d", "MaxPool2d", "MaxPool3d"],
    "AvgPool": ["avg_pool1d", "avg_pool2d", "avg_pool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d"],
    "GlobalAvgPool": [
        "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
        "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "mean",
    ],
    "Pooling": ["max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d",
                "avg_pool3d", "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d",
                "AvgPool2d", "AvgPool3d"],
    "cumsum": ["cumsum", "cumsum_"],
    "cumprod": ["cumprod", "cumprod_"],
    "ScaledDotProductAttention": [
        "scaled_dot_product_attention", "softmax", "Softmax", *_MATMUL_BANNED,
    ],
    "Dropout": ["dropout", "Dropout"],
}


def derive_banned_patterns(task: KbTask) -> list[str]:
    """Union of banned patterns for every known op token in the task name."""

    patterns: list[str] = []
    for token in _name_tokens(task.name):
        for pattern in _TOKEN_BANNED.get(token, ()):
            if pattern not in patterns:
                patterns.append(pattern)
    return patterns


# -------------------------------------------------------------------- probes


def contamination_reasons(stats: dict[str, Any]) -> list[str]:
    """Pure classification of one probe result against the robust-kbench
    thresholds; empty list == clean."""

    t = CONTAMINATION_THRESHOLDS
    reasons: list[str] = []
    if stats["out_abs_max"] <= t["out_abs_max_max"]:
        reasons.append(
            f"reference outputs all inside [-0.01, 0.01] "
            f"(max |out| = {stats['out_abs_max']:.2e})"
        )
    if stats["out_std"] < t["out_std_min"]:
        reasons.append(f"output std {stats['out_std']:.2e} < {t['out_std_min']}")
    if stats["input_impact"] < t["input_impact_min"]:
        reasons.append(
            f"input-impact {stats['input_impact']:.2e} < {t['input_impact_min']} "
            "(outputs barely change when inputs are re-randomized)"
        )
    if stats.get("near_identity"):
        reasons.append(
            "near-identity: eval-mode reference output ≈ its input "
            "(e.g. fresh BatchNorm running stats)"
        )
    return reasons


def parse_probe_lines(stdout: str) -> dict[str, dict[str, Any]]:
    """Marker-prefixed JSON lines → {task id: probe result}."""

    results: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        if not line.startswith(PROBE_MARKER):
            continue
        try:
            payload = json.loads(line[len(PROBE_MARKER):])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("id"):
            results[str(payload["id"])] = payload
    return results


def _probe_child(payload_path: str) -> int:
    """Subprocess body: run each task's reference once, print stats lines.

    Runs pinned to the leased device (the parent injects
    ``CUDA_VISIBLE_DEVICES``); one line per task so a timeout loses only the
    unfinished tail.
    """

    import importlib.util

    import torch

    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    workdir = Path(payload["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    atol = float(payload.get("near_identity_atol", 1e-3))

    for task in payload["tasks"]:
        out: dict[str, Any] = {"id": task["id"], "ok": False}
        started = time.perf_counter()
        try:
            path = workdir / f"{task['id']}_ref.py"
            path.write_text(task["source"], encoding="utf-8")
            spec = importlib.util.spec_from_file_location(f"kb_probe_{task['id']}", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            torch.manual_seed(0)
            model = module.Model().cuda().eval()

            def _tensors(value: Any) -> list[Any]:
                if isinstance(value, torch.Tensor):
                    return [value]
                if isinstance(value, (tuple, list)):
                    return [t for t in value if isinstance(t, torch.Tensor)]
                return []

            with torch.no_grad():
                torch.manual_seed(100)
                inputs_a = module.get_inputs()
                outs_a = _tensors(model(*inputs_a))
                torch.manual_seed(101)
                inputs_b = module.get_inputs()
                outs_b = _tensors(model(*inputs_b))
                torch.cuda.synchronize()

            if not outs_a:
                raise RuntimeError("reference returned no tensors")
            flat = torch.cat([t.detach().float().reshape(-1) for t in outs_a])
            out["out_abs_max"] = float(flat.abs().max())
            out["out_std"] = float(flat.std(correction=0)) if flat.numel() > 1 else 0.0
            out["input_impact"] = max(
                float((a.float() - b.float()).abs().max())
                for a, b in zip(outs_a, outs_b, strict=True)
            )
            tensors_in = [t for t in inputs_a if isinstance(t, torch.Tensor)]
            out["near_identity"] = any(
                t.shape == i.shape
                and torch.allclose(t.float(), i.float(), atol=atol, rtol=atol)
                for t in outs_a
                for i in tensors_in
            )
            out["input_shapes"] = [list(t.shape) for t in tensors_in]
            out["output_shapes"] = [list(t.shape) for t in outs_a]
            out["output_dtypes"] = [str(t.dtype) for t in outs_a]
            out["param_shapes"] = {
                k: list(v.shape) for k, v in model.state_dict().items()
            }
            out["ok"] = True
            del model, inputs_a, inputs_b, outs_a, outs_b, flat
        except Exception as exc:  # noqa: BLE001 — per-task failure is data
            out["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        finally:
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        out["probe_wall_s"] = round(time.perf_counter() - started, 2)
        print(PROBE_MARKER + json.dumps(out), flush=True)
    return 0


def run_probes(
    tasks: list[KbTask],
    *,
    workdir: Path,
    chunk_size: int = 8,
    chunk_timeout: float = 900.0,
    lease_timeout: float = 3600.0,
) -> dict[str, dict[str, Any]]:
    """Probe every task's reference on a leased pool device.

    One lease per chunk (amortizes the torch import while keeping each hold
    short enough not to starve concurrently running experiment drivers).
    """

    from compilagent.integrations.triton_source._internal.gpu_lease import acquire

    results: dict[str, dict[str, Any]] = {}
    chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    for chunk_index, chunk in enumerate(chunks):
        payload = {
            "near_identity_atol": CONTAMINATION_THRESHOLDS["near_identity_atol"],
            "workdir": str(workdir / f"chunk{chunk_index}"),
            "tasks": [
                {
                    "id": t.workload_id,
                    "source": adapt_reference_source(t.source, header=t.kb_file),
                }
                for t in chunk
            ],
        }
        payload_path = workdir / f"chunk{chunk_index}.json"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        lease = acquire(timeout=lease_timeout)
        with lease:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": lease.device}
            print(
                f"[probe] chunk {chunk_index + 1}/{len(chunks)} "
                f"({len(chunk)} task(s)) on leased gpu{lease.device} "
                f"(waited {lease.wait_seconds:.1f}s)",
                flush=True,
            )
            stdout = ""
            try:
                proc = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "scripts.kernelbench_select",
                        "--probe-child",
                        str(payload_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=chunk_timeout,
                    env=env,
                    cwd=REPO_ROOT,
                )
                stdout = proc.stdout or ""
                stderr_tail = (proc.stderr or "")[-1000:]
            except subprocess.TimeoutExpired as exc:
                stdout = (
                    exc.stdout.decode() if isinstance(exc.stdout, bytes)
                    else (exc.stdout or "")
                )
                stderr_tail = f"chunk timed out after {chunk_timeout:.0f}s"
        parsed = parse_probe_lines(stdout)
        for task in chunk:
            results[task.workload_id] = parsed.get(
                task.workload_id,
                {
                    "id": task.workload_id,
                    "ok": False,
                    "error": f"no probe result (chunk failure: {stderr_tail})",
                },
            )
    return results


# ------------------------------------------------------- stratified selection


def stratified_select(tasks: list[KbTask], quota: int) -> list[KbTask]:
    """Deterministic round-robin over op families, largest family first;
    ascending KB index inside each family."""

    buckets: dict[str, list[KbTask]] = {}
    for task in tasks:
        buckets.setdefault(op_family(task), []).append(task)
    for bucket in buckets.values():
        bucket.sort(key=lambda t: t.index)
    order = sorted(buckets, key=lambda f: (-len(buckets[f]), f))
    selected: list[KbTask] = []
    rank = 0
    while len(selected) < quota and any(buckets.values()):
        for family in order:
            if len(selected) >= quota:
                break
            if rank < len(buckets[family]):
                selected.append(buckets[family][rank])
        rank += 1
    selected.sort(key=lambda t: t.index)
    return selected


# ------------------------------------------------------------ manifest build


def build_selected_entry(task: KbTask, probe: dict[str, Any]) -> dict[str, Any]:
    template = parse_shape_template(task.source)
    input_shapes = template.input_shapes()
    params = probe.get("param_shapes") or {}
    param_note = (
        " Reference parameters: "
        + ", ".join(f"`{k}` {tuple(v)}" for k, v in params.items())
        + "."
        if params
        else ""
    )
    shapes_str = ", ".join(f"{k}: f32{tuple(v)}" for k, v in input_shapes.items())
    pretty_op = task.name.replace("_", " ")
    return {
        "workload_id": task.workload_id,
        "level": task.level,
        "kb_index": task.index,
        "kb_name": task.name,
        "kb_file": task.kb_file,
        "family": op_family(task),
        "title": f"KernelBench L{task.level} #{task.index}: {pretty_op}",
        "description": (
            f"KernelBench level-{task.level} task `{task.kb_file}`: {pretty_op} "
            f"over {shapes_str}.{param_note}"
        ),
        "op_signature": f"{task.name}({shapes_str})",
        "banned_patterns": derive_banned_patterns(task),
        "input_shapes": input_shapes,
        "reference_module_source": adapt_reference_source(
            task.source, header=f"KernelBench {task.kb_file} (adapted, ticket D8)"
        ),
        "holdout": {
            "inputs": template.inputs,
            "vars": template.vars,
        },
        "probe": {
            k: probe.get(k)
            for k in (
                "out_abs_max",
                "out_std",
                "input_impact",
                "near_identity",
                "output_shapes",
                "output_dtypes",
                "probe_wall_s",
            )
        },
    }


def _exclusion(task: KbTask, reasons: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "workload_id": task.workload_id,
        "level": task.level,
        "kb_file": task.kb_file,
        "reasons": reasons,
        **extra,
    }


def select_level(
    tasks: list[KbTask],
    probes: dict[str, dict[str, Any]],
    quota: int,
) -> tuple[list[KbTask], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply dynamic criteria to probed tasks, then stratify.

    Returns (selected, excluded_records, eligible_not_selected_records).
    Static exclusions are handled by the caller (they are never probed).
    """

    survivors: list[KbTask] = []
    excluded: list[dict[str, Any]] = []
    for task in tasks:
        probe = probes.get(task.workload_id) or {"ok": False, "error": "never probed"}
        if not probe.get("ok"):
            excluded.append(
                _exclusion(task, [f"probe_error: {probe.get('error')}"], probe=probe)
            )
            continue
        reasons = contamination_reasons(probe)
        if reasons:
            excluded.append(
                _exclusion(
                    task,
                    [f"contaminated: {r}" for r in reasons],
                    probe={
                        k: probe.get(k)
                        for k in ("out_abs_max", "out_std", "input_impact", "near_identity")
                    },
                )
            )
            continue
        survivors.append(task)
    selected = stratified_select(survivors, quota)
    chosen = {t.workload_id for t in selected}
    not_selected = [
        _exclusion(t, ["eligible_not_selected: stratification quota reached"],
                   family=op_family(t))
        for t in survivors
        if t.workload_id not in chosen
    ]
    return selected, excluded, not_selected


# ----------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-root", default=str(DEFAULT_KB_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--per-level", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--chunk-timeout", type=float, default=900.0)
    parser.add_argument(
        "--probe-cache", default=str(DEFAULT_PROBE_CACHE),
        help="JSON file of cached probe results (probing is the slow, "
             "GPU-leasing part; re-runs reuse it)",
    )
    parser.add_argument(
        "--probe-child", default="",
        help="(internal) run the probe subprocess body on this payload file",
    )
    args = parser.parse_args(argv)

    if args.probe_child:
        return _probe_child(args.probe_child)

    # Probes lease from the pool; default to GPU 1 (this machine's rule:
    # GPU 0 is reserved, experiments share 1-3 through the lease layer).
    os.environ.setdefault("COMPILAGENT_GPU_POOL", "1")

    kb_root = Path(args.kb_root)
    excluded: list[dict[str, Any]] = []
    probe_pool: dict[int, list[KbTask]] = {}
    for level in (1, 2):
        probe_pool[level] = []
        for task in scan_tasks(kb_root, level):
            reason = conv_exclusion(task) or inefficient_exclusion(task)
            if reason is None:
                try:
                    parse_shape_template(task.source)
                except TemplateError as exc:
                    reason = f"adapter_unparseable_inputs: {exc}"
            if reason is not None:
                excluded.append(_exclusion(task, [reason]))
            else:
                probe_pool[level].append(task)

    to_probe = probe_pool[1] + probe_pool[2]
    print(
        f"{len(to_probe)} candidate task(s) to probe "
        f"(L1 {len(probe_pool[1])}, L2 {len(probe_pool[2])}); "
        f"{len(excluded)} statically excluded.",
        flush=True,
    )

    cache_path = Path(args.probe_cache)
    probes: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        probes = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"loaded {len(probes)} cached probe result(s) from {cache_path}")
    missing = [t for t in to_probe if t.workload_id not in probes]
    if missing:
        fresh = run_probes(
            missing,
            workdir=cache_path.parent / "kb_probe_work",
            chunk_size=args.chunk_size,
            chunk_timeout=args.chunk_timeout,
        )
        probes.update(fresh)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(probes, indent=2), encoding="utf-8")
        print(f"probe cache updated → {cache_path}")

    selected_entries: list[dict[str, Any]] = []
    eligible_not_selected: list[dict[str, Any]] = []
    for level in (1, 2):
        selected, dyn_excluded, not_selected = select_level(
            probe_pool[level], probes, args.per_level
        )
        excluded.extend(dyn_excluded)
        eligible_not_selected.extend(not_selected)
        for task in selected:
            selected_entries.append(build_selected_entry(task, probes[task.workload_id]))
        print(
            f"level {level}: selected {len(selected)} / "
            f"{len(probe_pool[level])} probed candidates"
        )
        for task in selected:
            print(f"  {task.workload_id}  [{op_family(task)}]")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kb_root": str(kb_root),
        "criteria": {
            "static": [
                "cudnn_conv_bound",
                "inherently_inefficient_baseline",
                "adapter_unparseable_inputs",
            ],
            "dynamic_thresholds": CONTAMINATION_THRESHOLDS,
            "selection": "stratified round-robin by op family, largest family "
                         "first, ascending KB index within a family",
        },
        "selected": selected_entries,
        "eligible_not_selected": eligible_not_selected,
        "excluded": excluded,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"manifest → {out_path}  (selected {len(selected_entries)}, "
        f"excluded {len(excluded)}, eligible-not-selected "
        f"{len(eligible_not_selected)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
