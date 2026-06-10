"""Subprocess entry point that evaluates one `triton_source` candidate.

Executed as

    python -m compilagent.integrations.triton_source._internal.sandbox_runner \
        <payload.json>

inside a fresh process (CUDA_VISIBLE_DEVICES inherited from the parent) so a
hanging / OOMing candidate can be killed by the parent's subprocess timeout
without taking the session down. The payload JSON carries:

  - ``reference_source``: self-contained python module — `Model` nn.Module +
    `get_inputs()` factory (KernelBench L1 style).
  - ``candidate_source``: python module defining `ModelNew` (same
    `__init__`/`forward` signature). Absent/None → **baseline mode**: only
    the reference is timed (used for the empty-`Plan()` baseline compile).
  - ``atol`` / ``rtol``: fp32 tolerance (per-dtype table below for halves).
  - ``warmup`` / ``repetitions``: timing protocol (defaults 25 / 100).
  - ``trial_seeds``: value-randomization seeds (default 100..104).

Protocol (candidate mode):

  1. exec reference + candidate, `ModelNew().cuda().eval()`,
     `load_state_dict(ref.state_dict(), strict=False)`.
  2. 5 value-randomized trials (fresh `get_inputs()` per seed) with gates:
       g1  shape + dtype match
       g2  allclose under per-dtype tolerance (fp32: atol 1e-4, rtol 1e-3)
       g3  anti-aliasing — output must not share storage with any input, and
           inputs must be bit-identical before/after the candidate call
       g5  determinism — two candidate runs on the same seed/inputs are
           bitwise identical or within 1e-6
     (g4, the banned-API AST lint, runs in the parent *before* this
     subprocess is spawned — see `_internal.lint`.)
  3. CUDA-event timing of candidate AND reference — but **only when every
     gate passed** (budget-ledger rule: timing invocations are charged only
     for gate-passing candidates). 25 warmup launches, 100 timed reps,
     trimmed mean discarding 10% at each tail.

Result is a single JSON object on stdout behind the
``COMPILAGENT_SANDBOX_JSON:`` marker. Every failure mode is data (never a
non-zero crash the parent can't parse — exceptions are folded into
``error``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

MARKER = "COMPILAGENT_SANDBOX_JSON:"

DEFAULT_WARMUP = 25
DEFAULT_REPETITIONS = 100
DEFAULT_TRIM_FRACTION = 0.10
DEFAULT_TRIAL_SEEDS = (100, 101, 102, 103, 104)
DETERMINISM_ATOL = 1e-6


def _gate(ok: bool, message: str) -> dict[str, Any]:
    return {"ok": bool(ok), "message": message}


def _load_module(source: str, name: str, workdir: Path) -> dict[str, Any]:
    """Materialise `source` as a real .py file and import it by path.

    Triton 3.x refuses `@triton.jit` functions defined via `exec` with a
    synthetic filename ("@jit functions should be defined in a Python
    file"), so candidate/reference modules are written to disk first.
    """

    path = workdir / f"{name}_module.py"
    path.write_text(source, encoding="utf-8")
    module_name = f"compilagent_sandbox_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return vars(module)


def _tolerance_for(dtype: Any, torch: Any, atol: float, rtol: float) -> tuple[float, float]:
    """Per-dtype tolerance. fp32 comes from the payload (ticket E2a: atol
    1e-4 / rtol 1e-3); halves use conservative placeholders pending the
    E2b calibrated envelopes."""

    if dtype in (torch.float16,):
        return (1e-2, 1e-2)
    if dtype in (torch.bfloat16,):
        return (2e-2, 2e-2)
    if dtype in (torch.float64,):
        return (1e-7, 1e-6)
    return (atol, rtol)


def _as_tensor_list(out: Any, torch: Any) -> list[Any]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    return []


def _storage_ptr(tensor: Any) -> int:
    return tensor.untyped_storage().data_ptr()


def _bench(model: Any, inputs: list[Any], torch: Any, *, warmup: int, reps: int) -> dict[str, Any]:
    """CUDA-event timing: trimmed mean, 10% discarded at each tail."""

    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    samples.sort()
    k = int(len(samples) * DEFAULT_TRIM_FRACTION)
    trimmed = samples[k : len(samples) - k] if len(samples) > 2 * k else samples
    return {
        "mean_ms": sum(trimmed) / len(trimmed),
        "times_ms": trimmed,
    }


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "compiled": False,
        "gates": {},
        "max_abs_diff": None,
        "max_rel_diff": None,
        "cand_ms": None,
        "ref_ms": None,
        "speedup_vs_ref": None,
        "cand_times_ms": None,
        "ref_times_ms": None,
        "error": None,
        "warnings": [],
    }
    gates = out["gates"]

    warmup = int(payload.get("warmup", DEFAULT_WARMUP))
    reps = int(payload.get("repetitions", DEFAULT_REPETITIONS))
    seeds = [int(s) for s in payload.get("trial_seeds", DEFAULT_TRIAL_SEEDS)]
    atol = float(payload.get("atol", 1e-4))
    rtol = float(payload.get("rtol", 1e-3))
    candidate_source = payload.get("candidate_source")

    if not torch.cuda.is_available():
        out["error"] = "CUDA is not available inside the sandbox subprocess."
        return out

    workdir = Path(payload.get("workdir") or tempfile.mkdtemp(prefix="compilagent-sandbox-"))
    workdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    ref_ns = _load_module(payload["reference_source"], "reference", workdir)
    ref = ref_ns["Model"]().cuda().eval()
    get_inputs = ref_ns["get_inputs"]

    if candidate_source is None:
        # Baseline mode: time the reference module itself.
        inputs = get_inputs()
        with torch.no_grad():
            ref(*inputs)
            torch.cuda.synchronize()
        out["compiled"] = True
        bench = _bench(ref, inputs, torch, warmup=warmup, reps=reps)
        out["ref_ms"] = bench["mean_ms"]
        out["ref_times_ms"] = bench["times_ms"]
        gates["g1_shape_dtype"] = _gate(True, "baseline (reference) — gate vacuous")
        gates["g2_allclose"] = _gate(True, "baseline (reference) — gate vacuous")
        gates["g3_no_alias_no_mutation"] = _gate(True, "baseline (reference) — gate vacuous")
        gates["g5_determinism"] = _gate(True, "baseline (reference) — gate vacuous")
        return out

    cand_ns = _load_module(candidate_source, "candidate", workdir)
    if "ModelNew" not in cand_ns:
        out["error"] = "candidate module does not define `ModelNew`."
        return out
    torch.manual_seed(0)
    cand = cand_ns["ModelNew"]().cuda().eval()
    try:
        cand.load_state_dict(ref.state_dict(), strict=False)
    except Exception as exc:  # noqa: BLE001
        out["warnings"].append(
            f"load_state_dict(strict=False) failed: {type(exc).__name__}: {exc}"
        )

    max_abs = 0.0
    max_rel = 0.0
    g1_ok, g2_ok, g3_ok = True, True, True
    g1_msg = g2_msg = g3_msg = ""
    for seed in seeds:
        torch.manual_seed(seed)
        inputs = get_inputs()
        with torch.no_grad():
            r_out = ref(*inputs)
            torch.cuda.synchronize()
            input_tensors = [t for t in inputs if isinstance(t, torch.Tensor)]
            snapshots = [t.clone() for t in input_tensors]
            c_out = cand(*inputs)
            torch.cuda.synchronize()
        out["compiled"] = True

        r_list = _as_tensor_list(r_out, torch)
        c_list = _as_tensor_list(c_out, torch)
        if len(r_list) != len(c_list) or not r_list:
            g1_ok = False
            g1_msg = (
                f"seed {seed}: output structure mismatch "
                f"(reference {len(r_list)} tensor(s), candidate {len(c_list)})."
            )
            break
        mismatch = next(
            (
                (r, c)
                for r, c in zip(r_list, c_list)
                if r.shape != c.shape or r.dtype != c.dtype
            ),
            None,
        )
        if mismatch is not None:
            r, c = mismatch
            g1_ok = False
            g1_msg = (
                f"seed {seed}: shape/dtype mismatch — reference "
                f"{tuple(r.shape)}/{r.dtype}, candidate {tuple(c.shape)}/{c.dtype}."
            )
            break

        input_ptrs = {_storage_ptr(t) for t in input_tensors}
        aliased = [c for c in c_list if _storage_ptr(c) in input_ptrs]
        if aliased:
            g3_ok = False
            g3_msg = (
                f"seed {seed}: candidate output aliases an input buffer "
                "(shares storage). Allocate a fresh output tensor "
                "(e.g. torch.empty_like) instead of returning/reusing an input."
            )
            break
        mutated = [
            i
            for i, (t, snap) in enumerate(zip(input_tensors, snapshots))
            if not torch.equal(t, snap)
        ]
        if mutated:
            g3_ok = False
            g3_msg = (
                f"seed {seed}: candidate mutated input tensor(s) "
                f"{mutated} in place — inputs must be bit-identical before "
                "and after the candidate call."
            )
            break

        for r, c in zip(r_list, c_list):
            a, t = _tolerance_for(r.dtype, torch, atol, rtol)
            r32, c32 = r.float(), c.float()
            diff = (r32 - c32).abs()
            abs_d = float(diff.max())
            rel_d = float((diff / r32.abs().clamp_min(1e-12)).max())
            max_abs = max(max_abs, abs_d)
            max_rel = max(max_rel, rel_d)
            if not torch.allclose(r32, c32, atol=a, rtol=t):
                g2_ok = False
                g2_msg = (
                    f"seed {seed}: output drifted outside tolerance "
                    f"(atol={a}, rtol={t}): max_abs_diff={abs_d:.3e}, "
                    f"max_rel_diff={rel_d:.3e}."
                )
                break
        if not g2_ok:
            break

    out["max_abs_diff"] = max_abs
    out["max_rel_diff"] = max_rel
    gates["g1_shape_dtype"] = _gate(
        g1_ok, g1_msg or f"shape/dtype matched on {len(seeds)} randomized trials."
    )
    gates["g2_allclose"] = _gate(
        g2_ok,
        g2_msg
        or (
            f"allclose within tolerance on {len(seeds)} randomized trials "
            f"(max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e})."
        ),
    )
    gates["g3_no_alias_no_mutation"] = _gate(
        g3_ok, g3_msg or "no input aliasing; no in-place input mutation."
    )

    # g5 — determinism: same seed, same inputs, two runs.
    g5_ok, g5_msg = True, ""
    if g1_ok and g3_ok:
        torch.manual_seed(seeds[0])
        inputs = get_inputs()
        with torch.no_grad():
            first = _as_tensor_list(cand(*inputs), torch)
            torch.cuda.synchronize()
            second = _as_tensor_list(cand(*inputs), torch)
            torch.cuda.synchronize()
        for a_t, b_t in zip(first, second):
            if torch.equal(a_t, b_t):
                continue
            drift = float((a_t.float() - b_t.float()).abs().max())
            if drift > DETERMINISM_ATOL:
                g5_ok = False
                g5_msg = (
                    f"two runs on seed {seeds[0]} disagree by {drift:.3e} "
                    f"(> {DETERMINISM_ATOL}); the kernel is nondeterministic."
                )
                break
        g5_msg = g5_msg or "two same-seed runs bitwise identical (or within 1e-6)."
    else:
        g5_ok, g5_msg = False, "skipped: an earlier gate already failed."
    gates["g5_determinism"] = _gate(g5_ok, g5_msg)

    # Timing only for gate-passing candidates (budget-ledger rule).
    if all(g["ok"] for g in gates.values()):
        torch.manual_seed(0)
        inputs = get_inputs()
        cand_bench = _bench(cand, inputs, torch, warmup=warmup, reps=reps)
        ref_bench = _bench(ref, inputs, torch, warmup=warmup, reps=reps)
        out["cand_ms"] = cand_bench["mean_ms"]
        out["ref_ms"] = ref_bench["mean_ms"]
        out["cand_times_ms"] = cand_bench["times_ms"]
        out["ref_times_ms"] = ref_bench["times_ms"]
        if out["cand_ms"]:
            out["speedup_vs_ref"] = out["ref_ms"] / out["cand_ms"]
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(f"{MARKER}" + json.dumps({"error": "usage: sandbox_runner <payload.json>"}))
        return 0
    try:
        payload = json.loads(open(argv[0], encoding="utf-8").read())
    except Exception as exc:  # noqa: BLE001
        print(MARKER + json.dumps({"error": f"payload unreadable: {exc!r}"}))
        return 0
    try:
        result = evaluate(payload)
    except Exception as exc:  # noqa: BLE001
        result = {
            "compiled": False,
            "gates": {},
            "error": f"{type(exc).__name__}: {exc}"[:4000],
        }
    print(MARKER + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
