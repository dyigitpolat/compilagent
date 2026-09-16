#!/usr/bin/env python3
"""Loose-verifier replay: what a stock checker would have admitted.

The paper's five gates rejected every candidate in this population before
timing. This GPU-only replay (no LLM calls) re-evaluates each of them under
a KernelBench-style loose check -- shape/dtype equality and
``allclose(atol=rtol=1e-2)`` on the same five value-randomised trial seeds,
with NO anti-aliasing/mutation gate (g3), NO banned-API lint (g4) and NO
determinism gate (g5) -- and times every candidate that check admits. The
verifier axis (D5) thereby gets a second, measured setting: how many
rejected candidates a stock checker admits, and the apparent speedups it
would post for them.

Population: every candidate of the kernel-source grids (``t1lite_or.jsonl``
and ``kb24_grid.jsonl``) that compiled but failed at least one gate
(``compile_ok`` true, ``correctness_ok`` not true), taken over the good rows
of each ledger -- no error, completion_reason not in {None, harness_failed,
driver_error}, last row per key wins (the aggregate_all.py convention).
Sources and the exact evaluation payload (reference module, warmup,
repetitions, trial seeds) come from the run's stored artifacts under
``.compilagent/workloads/<wl>/runs/<run>/candidates/<cid>/``; only the
tolerance and the gate policy change. Rows are keyed
``workload|run_id|candidate_id`` and the pass is resumable.

Usage::

    COMPILAGENT_GPU_POOL=1 env/bin/python -m scripts.replay_loose_verifier            # full pass
    COMPILAGENT_GPU_POOL=1 env/bin/python -m scripts.replay_loose_verifier --limit 20 --out /tmp/smoke.jsonl
    env/bin/python -m scripts.replay_loose_verifier --dry-run                           # population only
    env/bin/python -m scripts.replay_loose_verifier --summary                           # statistics
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_pilot import _append_row  # noqa: E402

RESULTS = REPO_ROOT / "scripts" / "results"

#: (ledger label, file) -- the 420-episode kernel-source grids the paper's
#: gate-rejection paragraph counts over.
LEDGERS: tuple[tuple[str, str], ...] = (
    ("headline", "t1lite_or.jsonl"),
    ("kb24", "kb24_grid.jsonl"),
)

#: The stock-checker emulation: KernelBench's evaluator compares five
#: random-input trials under atol/rtol 1e-2 and has no aliasing, lint or
#: determinism check.
LOOSE_ATOL = 1e-2
LOOSE_RTOL = 1e-2
SKIPPED_GATES = ("g3_no_alias_no_mutation", "g5_determinism")
REQUIRED_GATES = ("g1_shape_dtype", "g2_allclose")
ALL_GATES = ("g1_shape_dtype", "g2_allclose", "g3_no_alias_no_mutation", "g5_determinism")


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------


def load_good_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Good rows keyed by cell key, last one winning (aggregate_all.py)."""

    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("error") or row.get("completion_reason") in (
            None,
            "harness_failed",
            "driver_error",
        ):
            continue
        out[row["key"]] = row
    return out


def failing_gates(candidate: dict[str, Any]) -> list[str]:
    gates = candidate.get("gates") or {}
    return sorted(
        name
        for name, verdict in gates.items()
        if isinstance(verdict, dict) and verdict.get("ok") is False
    )


def is_rejected(candidate: dict[str, Any]) -> bool:
    """Compiled, but not accepted by the gates."""

    return bool(candidate.get("compile_ok")) and candidate.get("correctness_ok") is not True


def population(ledger: str, rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in rows.values():
        for cand in row.get("candidates") or ():
            if not is_rejected(cand):
                continue
            items.append(
                {
                    "key": f"{row['workload']}|{row['run_id']}|{cand['candidate_id']}",
                    "ledger": ledger,
                    "harness": row.get("harness"),
                    "workload": row["workload"],
                    "seed": row.get("seed"),
                    "budget": row.get("budget"),
                    "model_id": row.get("model_id"),
                    "run_id": row["run_id"],
                    "candidate_id": cand["candidate_id"],
                    "original_failing_gates": failing_gates(cand),
                }
            )
    items.sort(key=lambda it: (it["ledger"], it["workload"], it["run_id"], it["candidate_id"]))
    return items


def full_population(ledger_paths: dict[str, Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for ledger, path in ledger_paths.items():
        items.extend(population(ledger, load_good_rows(path)))
    return items


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def candidate_dir(workspace: Path, item: dict[str, Any]) -> Path:
    return (
        workspace
        / "workloads"
        / item["workload"]
        / "runs"
        / item["run_id"]
        / "candidates"
        / item["candidate_id"]
    )


def replay_one(
    item: dict[str, Any], *, workspace: Path, artifacts_root: Path, timeout: float
) -> dict[str, Any]:
    from compilagent.integrations.triton_source._internal.sandbox import (
        run_sandboxed_eval,
    )

    base = {**item, "timestamp": time.time()}
    cdir = candidate_dir(workspace, item)
    payload_path = cdir / "sandbox_payload.json"
    if not payload_path.exists():
        return {**base, "error": f"missing artifact {payload_path}"}
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not payload.get("candidate_source"):
        return {**base, "error": "stored payload carries no candidate source"}

    t0 = time.perf_counter()
    result = run_sandboxed_eval(
        reference_source=payload["reference_source"],
        candidate_source=payload["candidate_source"],
        artifact_dir=artifacts_root / item["workload"] / item["run_id"] / item["candidate_id"],
        atol=LOOSE_ATOL,
        rtol=LOOSE_RTOL,
        timeout_seconds=timeout,
        warmup=int(payload.get("warmup", 25)),
        repetitions=int(payload.get("repetitions", 100)),
        trial_seeds=tuple(int(s) for s in payload.get("trial_seeds", (100, 101, 102, 103, 104))),
        skip_gates=SKIPPED_GATES,
    )
    wall = time.perf_counter() - t0
    gates = result.get("gates") or {}
    loose_gates = {g: gates.get(g, {}).get("ok") for g in ALL_GATES if g in gates}
    admitted = (
        bool(result.get("compiled"))
        and not result.get("error")
        and all(gates.get(g, {}).get("ok") is True for g in REQUIRED_GATES)
    )
    speedup = result.get("speedup_vs_ref") if admitted else None
    lease = result.get("gpu_lease") or {}
    return {
        **base,
        "loose_admitted": admitted,
        "loose_gates": loose_gates,
        "apparent_speedup": speedup,
        "cand_ms": result.get("cand_ms") if admitted else None,
        "ref_ms": result.get("ref_ms") if admitted else None,
        "max_abs_diff": result.get("max_abs_diff"),
        "max_rel_diff": result.get("max_rel_diff"),
        "error": result.get("error"),
        "timed_out": bool(result.get("timed_out")),
        "gpu_lease_wait_s": lease.get("lease_wait_s"),
        "wall_s": round(wall, 3),
    }


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def load_replay_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Replay rows keyed by candidate key; a later row supersedes an
    earlier one, an error row never supersedes a good one."""

    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error") and row["key"] in out and not out[row["key"]].get("error"):
            continue
        out[row["key"]] = row
    return out


def summarize(rows: list[dict[str, Any]], population_size: int | None = None) -> dict[str, Any]:
    good = [r for r in rows if not r.get("error")]
    errors = [r for r in rows if r.get("error")]
    admitted = [r for r in good if r.get("loose_admitted")]
    speedups = [
        float(r["apparent_speedup"])
        for r in admitted
        if isinstance(r.get("apparent_speedup"), (int, float))
    ]

    def by(field: str, transform=lambda v: v) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in good:
            k = str(transform(r.get(field)))
            slot = out.setdefault(k, {"evaluated": 0, "admitted": 0, "gt1": 0})
            slot["evaluated"] += 1
            if r.get("loose_admitted"):
                slot["admitted"] += 1
                if isinstance(r.get("apparent_speedup"), (int, float)) and r["apparent_speedup"] > 1.0:
                    slot["gt1"] += 1
        return dict(sorted(out.items()))

    return {
        "population_size": population_size,
        "evaluated": len(good),
        "errors": len(errors),
        "admitted": len(admitted),
        "admitted_fraction": (len(admitted) / len(good)) if good else None,
        "by_original_failing_gates": by("original_failing_gates", lambda v: "+".join(v or ())),
        "by_harness": by("harness"),
        "by_ledger": by("ledger"),
        "apparent_speedup": {
            "n": len(speedups),
            "max": max(speedups) if speedups else None,
            "median": statistics.median(speedups) if speedups else None,
            "count_gt_1.0": sum(1 for s in speedups if s > 1.0),
            "count_gt_1.2": sum(1 for s in speedups if s > 1.2),
            "count_ge_2.0": sum(1 for s in speedups if s >= 2.0),
        },
        # A stock fast_1 win: admitted by the loose check and faster than
        # the reference.
        "stock_fast1_wins": sum(1 for s in speedups if s > 1.0),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------



def _completed_keys(out_path: Path) -> set[str]:
    """Keys already evaluated: rows whose ``error`` is empty. (`run_pilot`'s
    helper tests for the absence of the key; replay rows always carry
    ``error: null``, so that test would re-run everything.)"""

    done: set[str] = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("key") and not row.get("error"):
            done.add(str(row["key"]))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default=str(REPO_ROOT / ".compilagent"))
    parser.add_argument(
        "--artifacts", default=str(RESULTS / "loose_replay_artifacts"),
        help="where each replayed candidate's sandbox files are written",
    )
    parser.add_argument("--out", default=str(RESULTS / "loose_verifier_replay.jsonl"))
    parser.add_argument(
        "--ledger", action="append", default=None, metavar="LABEL=FILE",
        help="override the ledgers (default: headline=t1lite_or.jsonl, kb24=kb24_grid.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N evaluations")
    parser.add_argument("--timeout", type=float, default=240.0, help="sandbox hard timeout (s)")
    parser.add_argument("--dry-run", action="store_true", help="list the population, run nothing")
    parser.add_argument("--summary", action="store_true", help="summarise --out and exit")
    args = parser.parse_args(argv)

    if args.ledger:
        ledger_paths = {}
        for spec in args.ledger:
            label, _, file = spec.partition("=")
            ledger_paths[label] = Path(file) if "/" in file else RESULTS / file
    else:
        ledger_paths = {label: RESULTS / file for label, file in LEDGERS}

    items = full_population(ledger_paths)
    out_path = Path(args.out)

    if args.summary:
        rows = list(load_replay_rows(out_path).values())
        print_summary(summarize(rows, population_size=len(items)))
        return 0

    per_ledger = Counter(it["ledger"] for it in items)
    per_gates = Counter("+".join(it["original_failing_gates"]) for it in items)
    print(f"population: {len(items)} gate-rejected candidates {dict(per_ledger)}", flush=True)
    print(f"by original failing gates: {dict(per_gates.most_common())}", flush=True)
    if args.dry_run:
        return 0

    # GPU-only via leases; GPU 0 is reserved on this machine.
    os.environ.setdefault("COMPILAGENT_GPU_POOL", "1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(out_path)
    workspace = Path(args.workspace)
    artifacts_root = Path(args.artifacts)

    evaluated = 0
    admitted = 0
    t_start = time.perf_counter()
    for item in items:
        if item["key"] in done:
            continue
        if args.limit and evaluated >= args.limit:
            break
        row = replay_one(item, workspace=workspace, artifacts_root=artifacts_root, timeout=args.timeout)
        _append_row(out_path, row)
        evaluated += 1
        admitted += bool(row.get("loose_admitted"))
        print(
            f"[{evaluated}] {item['key']} orig={'+'.join(item['original_failing_gates']) or '-'} "
            f"admitted={row.get('loose_admitted')} speedup={row.get('apparent_speedup')} "
            f"err={str(row.get('error'))[:60] if row.get('error') else None} wall={row.get('wall_s')}s",
            flush=True,
        )
    elapsed = time.perf_counter() - t_start
    print(
        f"evaluated {evaluated} (admitted {admitted}) in {elapsed:.0f}s"
        f"{f' ({elapsed / evaluated:.1f}s each)' if evaluated else ''}; "
        f"{len(done)} already done → {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
