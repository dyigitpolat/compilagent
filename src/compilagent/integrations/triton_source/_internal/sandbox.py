"""Parent-side driver for the `triton_source` subprocess sandbox.

`run_sandboxed_eval` writes the evaluation payload next to the candidate
artifacts, spawns `sandbox_runner` in a fresh interpreter (inheriting the
parent environment — so `CUDA_VISIBLE_DEVICES` pinning flows through), and
parses the marker-prefixed JSON the runner prints.

Hard timeout: `subprocess.run(timeout=...)` kills the child on expiry, so a
candidate that hangs (`while True: pass`), deadlocks, or OOMs can never take
the session down — the failure comes back as an error dict the backend folds
into `CompileResult(ok=False, ...)`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .sandbox_runner import (
    DEFAULT_REPETITIONS,
    DEFAULT_TRIAL_SEEDS,
    DEFAULT_WARMUP,
    MARKER,
)

DEFAULT_TIMEOUT_SECONDS = 240.0

_RUNNER_MODULE = "compilagent.integrations.triton_source._internal.sandbox_runner"


def _failure(error: str, *, timed_out: bool = False) -> dict[str, Any]:
    return {
        "compiled": False,
        "gates": {},
        "cand_ms": None,
        "ref_ms": None,
        "speedup_vs_ref": None,
        "error": error,
        "timed_out": timed_out,
    }


def run_sandboxed_eval(
    *,
    reference_source: str,
    candidate_source: str | None,
    artifact_dir: Path,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    warmup: int = DEFAULT_WARMUP,
    repetitions: int = DEFAULT_REPETITIONS,
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
) -> dict[str, Any]:
    """Evaluate one candidate (or the baseline, when `candidate_source` is
    None) in a sandboxed subprocess; never raises for expected failures."""

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_source": reference_source,
        "candidate_source": candidate_source,
        "workdir": str(artifact_dir),
        "atol": atol,
        "rtol": rtol,
        "warmup": warmup,
        "repetitions": repetitions,
        "trial_seeds": list(trial_seeds),
    }
    payload_path = artifact_dir / "sandbox_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", _RUNNER_MODULE, str(payload_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return _failure(
            f"sandbox subprocess exceeded the {timeout_seconds:.0f}s hard "
            "timeout (hang, deadlock, or pathological kernel) and was killed.",
            timed_out=True,
        )
    except Exception as exc:  # noqa: BLE001
        return _failure(f"sandbox subprocess failed to launch: {exc!r}")

    (artifact_dir / "sandbox_stdout.log").write_text(
        proc.stdout or "", encoding="utf-8"
    )
    (artifact_dir / "sandbox_stderr.log").write_text(
        proc.stderr or "", encoding="utf-8"
    )

    for line in (proc.stdout or "").splitlines():
        if line.startswith(MARKER):
            try:
                result = json.loads(line[len(MARKER):])
            except json.JSONDecodeError as exc:
                return _failure(f"sandbox emitted unparseable JSON: {exc}")
            result.setdefault("timed_out", False)
            return result

    stderr_tail = (proc.stderr or "")[-2000:]
    return _failure(
        "sandbox produced no result JSON "
        f"(exit code {proc.returncode}); stderr tail: {stderr_tail}"
    )
