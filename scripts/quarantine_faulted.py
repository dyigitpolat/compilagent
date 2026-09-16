"""Quarantine the episodes the 2026-06-11 sandbox fault touched, so that
`run_pilot` re-runs them.

The fault
---------
Commit 32e118a made ``triton_source/__init__.py`` import
``kernelbench_workloads``, which read ``kernelbench_manifest.json`` at import
time; the manifest was written at 02:27 local, some minutes after the import
landed. Between roughly 02:20 and 02:27 every sandbox subprocess (the runner
lives under that package, so ``python -m ...sandbox_runner`` executes the
package ``__init__`` first) died with ``FileNotFoundError`` before evaluating
anything, and ``sandbox.py`` returned the death as a candidate compile
failure. Four grids were running with ~42 episode workers, so a seven-minute
outage hit 77 in-flight episodes. The guards that close the hole are in
``sandbox.py`` (``SandboxInfrastructureError``, ``COMPILAGENT_SANDBOX_CHILD``).

Two channels of contamination, two quarantine criteria
------------------------------------------------------
1. ``sandbox_import_fault`` — a row is quarantined when any candidate's
   diagnostics carry the runner's manifest traceback. Every such candidate
   was booked as a compile failure, so the episode's search was steered by
   false feedback and its failure allowance was burnt by the fault.
2. ``memory_rule_exposure`` — CASCADE's skill memory distilled the false
   failures into one constraint rule ("avoid this failure mode: sandbox
   produced no result JSON ...") that reached frequency 361, the highest in
   the store, so every memory-bearing CASCADE episode that *started* after
   the fault began and before the rule was purged had that rule injected at
   the top of its prompt. Those episodes are quarantined as well and re-run
   from the purged store.

Rows are moved (not deleted) into ``<ledger>.quarantined_20260611.jsonl``
with a ``quarantine_reason`` field; ``run_pilot`` then treats the cells as
never run. Error rows (``"error"`` key) are left alone: the driver retries
them by itself.

Usage::

    env/bin/python -m scripts.quarantine_faulted --dry-run    # list only
    env/bin/python -m scripts.quarantine_faulted              # move rows, purge the rule
    env/bin/python -m scripts.quarantine_faulted --check      # exit 1 if any remain
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "scripts" / "results"
MEMORY = REPO_ROOT / ".compilagent" / "memory"
RULES_PATH = MEMORY / "skill_rules.json"
PURGED_RULES_PATH = MEMORY / "skill_rules.purged_20260611.json"
KEYS_PATH = RESULTS / "quarantine_20260611_keys.txt"

LEDGERS = (
    "t1lite_or.jsonl",
    "ablation_bundles.jsonl",
    "depth_sweep.jsonl",
    "stability_qwen.jsonl",
    "kb24_grid.jsonl",
)

#: Local wall-clock (this machine, UTC+8) start of the fault window: the first
#: faulted candidate evaluation is at 02:2x; episode START times are compared
#: against this for the memory-exposure criterion.
FAULT_START = dt.datetime(2026, 6, 11, 2, 20).timestamp()

CRASH_TEXT = "sandbox produced no result JSON"
MANIFEST_TEXT = "kernelbench_manifest.json"
RULE_TEXT = "sandbox produced no result JSON"

QUARANTINE_SUFFIX = ".quarantined_20260611.jsonl"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Rewrite a ledger atomically under its sidecar flock (the same lock
    `run_pilot._append_row` takes), so a concurrent appender never
    interleaves with the rewrite."""

    lock_path = path.with_name(path.name + ".lock")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(lock_path, "a", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            tmp.write_text(
                "".join(json.dumps(r, default=str) + "\n" for r in rows),
                encoding="utf-8",
            )
            tmp.replace(path)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _crash_kinds(row: dict[str, Any]) -> Counter:
    """Classify every no-JSON crash among the row's candidates."""

    kinds: Counter = Counter()
    for cand in row.get("candidates") or ():
        diag = str(cand.get("diagnostics") or "")
        if CRASH_TEXT not in diag:
            continue
        kinds["manifest" if MANIFEST_TEXT in diag else "other"] += 1
    return kinds


def _purge_time() -> float | None:
    if PURGED_RULES_PATH.exists():
        return float(json.loads(PURGED_RULES_PATH.read_text())["purged_at"])
    return None


def _exposed(row: dict[str, Any], window_end: float) -> bool:
    if row.get("harness") != "cascade" or row.get("policy", "null") == "null":
        return False
    start = float(row["timestamp"]) - float(row.get("wallclock_s") or 0.0)
    return FAULT_START < start and float(row["timestamp"]) <= window_end


def classify(
    rows: list[dict[str, Any]], window_end: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    """Split rows into (kept, quarantined-with-reason) and count crash kinds."""

    kept, quarantined = [], []
    other_crashes: Counter = Counter()
    for row in rows:
        if "error" in row:  # the driver retries these on its own
            kept.append(row)
            continue
        reasons = []
        kinds = _crash_kinds(row)
        if kinds["manifest"]:
            reasons.append("sandbox_import_fault")
        if kinds["other"]:
            other_crashes[row["key"]] += kinds["other"]
        if _exposed(row, window_end):
            reasons.append("memory_rule_exposure")
        if reasons:
            quarantined.append({**row, "quarantine_reason": reasons})
        else:
            kept.append(row)
    return kept, quarantined, other_crashes


def _purge_rules(dry_run: bool) -> list[dict[str, Any]]:
    data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    bad = [r for r in data["rules"] if RULE_TEXT in str(r.get("text", ""))]
    if not bad or dry_run:
        return bad
    data["rules"] = [r for r in data["rules"] if RULE_TEXT not in str(r.get("text", ""))]
    tmp = RULES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(RULES_PATH)
    PURGED_RULES_PATH.write_text(
        json.dumps(
            {
                "purged_at": time.time(),
                "purged_at_local": dt.datetime.now().isoformat(timespec="seconds"),
                "rules_sha_after": _sha(RULES_PATH),
                "removed": bad,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report, touch nothing")
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any ledger still holds a faulted or exposed row",
    )
    args = ap.parse_args(argv)
    dry_run = args.dry_run or args.check

    window_end = _purge_time() or time.time()
    total = Counter()
    all_keys: list[str] = []
    remaining = 0
    for name in LEDGERS:
        path = RESULTS / name
        rows = _read_rows(path)
        kept, quarantined, other = classify(rows, window_end)
        by_reason = Counter(r for q in quarantined for r in q["quarantine_reason"])
        by_harness = Counter(q["harness"] for q in quarantined)
        print(
            f"{name:26s} rows {len(rows):4d}  quarantine {len(quarantined):3d}  "
            f"{dict(by_reason)}  {dict(by_harness)}"
        )
        if other:
            print(f"  !! {sum(other.values())} no-JSON crashes of another kind in {len(other)} rows: "
                  f"{list(other)[:3]}")
        for q in quarantined:
            total[tuple(q["quarantine_reason"])] += 1
            all_keys.append(f"{name}\t{'+'.join(q['quarantine_reason'])}\t{q['key']}")
        remaining += len(quarantined)
        if not dry_run and quarantined:
            _append_rows(path.with_name(name.replace(".jsonl", QUARANTINE_SUFFIX)), quarantined)
            _write_rows(path, kept)
    print(f"\nquarantine total: {remaining} episodes  by reason set: {dict(total)}")

    bad_rules = _purge_rules(dry_run)
    print(f"fault-derived skill rules {'found' if dry_run else 'purged'}: {len(bad_rules)}"
          + (f" (frequency {[r.get('frequency') for r in bad_rules]})" if bad_rules else ""))

    if args.check:
        ok = remaining == 0 and not bad_rules
        print("CHECK", "OK" if ok else "FAILED")
        return 0 if ok else 1
    if not dry_run:
        KEYS_PATH.write_text("\n".join(all_keys) + "\n", encoding="utf-8")
        print(f"keys written to {KEYS_PATH}; rules store sha {_sha(RULES_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
