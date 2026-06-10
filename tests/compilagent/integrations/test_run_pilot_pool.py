"""Driver-level tests for `scripts.run_pilot` pool mode (CPU-only).

`--dry-run` swaps episode execution for a 0.1 s stub row, so these tests
exercise the real spawn/queue/append/resume plumbing — N worker processes,
flock-serialized JSONL appends — without GPUs, torch, or LLM keys. Every
subprocess runs with CUDA_VISIBLE_DEVICES="" as a belt-and-braces guard.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

#: 2 harnesses × 3 workloads × 1 budget × 2 seeds = 12 cells.
_GRID = [
    "--harnesses", "h_a,h_b",
    "--workloads", "w1,w2,w3",
    "--budgets", "8",
    "--seeds", "13,42",
]


def _driver_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # nothing here may ever touch a GPU
    return env


def _run_driver(out: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_pilot", *args, "--out", str(out)],
        cwd=REPO_ROOT,
        env=_driver_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )


def _rows(out: Path) -> list[dict]:
    lines = out.read_text(encoding="utf-8").splitlines()
    assert all(line.strip() for line in lines), "blank/torn line in JSONL"
    return [json.loads(line) for line in lines]  # every line must parse


# --------------------------------------------- (d) pool-mode dry-run drains


def test_pool_dry_run_8_workers_complete_all_cells_exactly_once(tmp_path):
    out = tmp_path / "pool.jsonl"
    proc = _run_driver(
        out,
        [*_GRID, "--gpu-pool", "1,2,3", "--episode-workers", "8", "--dry-run"],
    )
    assert proc.returncode == 0, proc.stderr
    rows = _rows(out)
    assert len(rows) == 12
    keys = {r["key"] for r in rows}
    assert len(keys) == 12, "duplicate cell keys in pool-mode output"
    assert all(r["completion_reason"] == "dry_run" for r in rows)
    assert all(r["gpu"] == "pool:1,2,3" for r in rows)
    # No worker may exit before the queue drains (8 workers > 12 cells / 8).
    assert "exited non-zero" not in proc.stderr


def test_pool_dry_run_resumes_by_skipping_completed_keys(tmp_path):
    out = tmp_path / "pool.jsonl"
    args = [*_GRID, "--gpu-pool", "1,2,3", "--episode-workers", "8", "--dry-run"]
    assert _run_driver(out, args).returncode == 0
    rerun = _run_driver(out, args)
    assert rerun.returncode == 0, rerun.stderr
    assert "Nothing to do" in rerun.stdout
    assert len(_rows(out)) == 12  # unchanged: nothing re-ran or duplicated


# ------------------------------------------------------------ CLI guard rails


def test_gpus_and_gpu_pool_are_mutually_exclusive(tmp_path):
    proc = _run_driver(
        tmp_path / "x.jsonl",
        [*_GRID, "--gpus", "1", "--gpu-pool", "2,3", "--dry-run"],
    )
    assert proc.returncode == 2
    assert "mutually exclusive" in proc.stderr


def test_episode_workers_requires_gpu_pool(tmp_path):
    proc = _run_driver(
        tmp_path / "x.jsonl",
        [*_GRID, "--gpus", "1", "--episode-workers", "4", "--dry-run"],
    )
    assert proc.returncode == 2
    assert "--gpu-pool" in proc.stderr


def test_gpu_zero_is_rejected_in_the_pool(tmp_path):
    proc = _run_driver(
        tmp_path / "x.jsonl",
        [*_GRID, "--gpu-pool", "0,1", "--dry-run"],
    )
    assert proc.returncode == 2
    assert "GPU 0 is reserved" in proc.stderr


# ------------------------------- (e) JSONL append safety under 8 hammerers


_HAMMER_CHILD = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from scripts.run_pilot import _append_row

    out = Path(sys.argv[1]); writer = sys.argv[2]; n = int(sys.argv[3])
    for i in range(n):
        _append_row(out, {
            "key": f"{writer}:{i}",
            "writer": writer,
            "i": i,
            # Big enough that a torn write would straddle pipe/page buffers.
            "pad": "x" * 4096,
        })
    """
)


def test_concurrent_appends_from_8_processes_never_tear_lines(tmp_path):
    out = tmp_path / "hammer.jsonl"
    writers, per_writer = 8, 40
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _HAMMER_CHILD, str(out), f"w{i}", str(per_writer)],
            cwd=REPO_ROOT,  # puts the repo root (and so `scripts`) on sys.path
            env=_driver_env(),
        )
        for i in range(writers)
    ]
    for p in procs:
        assert p.wait(timeout=120) == 0
    rows = _rows(out)  # every line parses — no interleaving, no tearing
    assert len(rows) == writers * per_writer
    keys = {r["key"] for r in rows}
    assert keys == {f"w{i}:{j}" for i in range(writers) for j in range(per_writer)}
