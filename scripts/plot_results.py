"""Plot per-candidate speedup from `scripts/results/suite_results.json`.

Produces a horizontal bar chart of `best_speedup` for each of the 12
candidates, with a 1× baseline reference line. Modules and kernels are
color-coded. Saves PNG to `scripts/results/suite_speedups.png`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = REPO_ROOT / "scripts" / "results" / "suite_results.json"
DEFAULT_PNG = REPO_ROOT / "scripts" / "results" / "suite_speedups.png"


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing results file: {path}. Run `python -m scripts.run_suite` first.")
    return json.loads(path.read_text())


def _plot(payload: dict, out_path: Path) -> None:
    rows = list(payload.get("rows", []))
    rows.sort(key=lambda r: (r.get("kind", ""), r.get("name", "")))

    names = [r["name"] for r in rows]
    kinds = [r.get("kind", "?") for r in rows]
    speedups = [
        r.get("best_speedup") if isinstance(r.get("best_speedup"), (int, float)) else 1.0
        for r in rows
    ]
    improved = [bool(r.get("improved")) for r in rows]
    errored = ["error" in r for r in rows]

    colors = []
    for kind, ok, err in zip(kinds, improved, errored):
        if err:
            colors.append("#b0b0b0")
        elif kind == "kernel":
            colors.append("#1f77b4" if ok else "#9ec5e8")
        else:
            colors.append("#d62728" if ok else "#f5b3b3")

    fig_height = max(4.5, 0.45 * len(names) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = list(range(len(names)))
    ax.barh(y, speedups, color=colors, edgecolor="#333", linewidth=0.5)
    ax.axvline(1.0, color="#444", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  ({k})" for n, k in zip(names, kinds)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("speedup vs baseline (×)")

    model_id = payload.get("model_id", "?")
    harness = payload.get("harness", "?")
    gpu = payload.get("gpu", "?")
    elapsed = payload.get("elapsed_seconds")
    title = f"compilagent suite — {harness} · {model_id}"
    if isinstance(elapsed, (int, float)):
        title += f" · {elapsed/60:.1f} min"
    ax.set_title(f"{title}\n{gpu}", fontsize=10)

    for yi, sp, err in zip(y, speedups, errored):
        if err:
            ax.text(0.02, yi, "error", va="center", fontsize=8, color="#444")
        else:
            ax.text(sp + 0.02, yi, f"{sp:.2f}×", va="center", fontsize=8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="kernel (improved)"),
        plt.Rectangle((0, 0), 1, 1, color="#9ec5e8", label="kernel (no win)"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="module (improved)"),
        plt.Rectangle((0, 0), 1, 1, color="#f5b3b3", label="module (no win)"),
        plt.Rectangle((0, 0), 1, 1, color="#b0b0b0", label="error"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PNG
    payload = _load(json_path)
    _plot(payload, out_path)
    print(f"Saved plot → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
