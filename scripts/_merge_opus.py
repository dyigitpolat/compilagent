"""Splice opus remaining-2 rows into the 10-row premerge state."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
BASE = RESULTS / "suite_results_opus_premerge.json"
NEW = RESULTS / "suite_results_opus_remaining.json"
OUT = RESULTS / "suite_results_opus.json"


def main() -> int:
    if not BASE.exists() or not NEW.exists():
        return 1
    base = json.loads(BASE.read_text())
    new = json.loads(NEW.read_text())
    by_name = {r["name"]: r for r in base.get("rows", [])}
    for r in new.get("rows", []):
        by_name[r["name"]] = r
    merged = dict(base)
    merged["rows"] = list(by_name.values())
    merged["elapsed_seconds"] = (
        (base.get("elapsed_seconds") or 0.0) + (new.get("elapsed_seconds") or 0.0)
    )
    merged["model_id"] = base.get("model_id") or new.get("model_id")
    OUT.write_text(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
