"""Single-bar-per-workload speedup plot, grouped by backend.

Usage:
    env/bin/python scripts/experiments/plot_workload_grid.py runs/study/<dir>/results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_jsonl", type=Path)
    args = p.parse_args()
    rows = [
        json.loads(l) for l in args.results_jsonl.read_text().splitlines()
        if l.strip()
    ]
    rows.sort(key=lambda r: (r["backend_id"], -(r.get("best_speedup") or 0)))
    labels, speedups, colors = [], [], []
    backend_color = {"triton": "#22c55e", "torch_inductor": "#0ea5e9"}
    for r in rows:
        sp = r.get("best_speedup")
        if not isinstance(sp, (int, float)):
            continue
        labels.append(r["workload_id"])
        speedups.append(float(sp))
        colors.append(backend_color.get(r["backend_id"], "#64748b"))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    bars = ax.barh(range(len(labels)), speedups, color=colors,
                   edgecolor="#1e293b", linewidth=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#64748b", linestyle=":", linewidth=0.8)
    ax.set_xlabel("speedup vs. baseline (1.0× = no change)")
    ax.set_xlim(left=0.95, right=max(1.20, max(speedups) + 0.02))
    ax.set_title(
        "Mistral large × pydantic-ai × 8 trials — best validated speedup per workload",
        fontsize=11,
    )

    # Backend legend.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=backend_color["torch_inductor"], label="torch_inductor"),
        plt.Rectangle((0, 0), 1, 1, color=backend_color["triton"], label="triton"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    # Per-bar labels: speedup + success ratio.
    for i, (bar, r) in enumerate(zip(bars, [r for r in rows if isinstance(r.get("best_speedup"), (int, float))])):
        sp = bar.get_width()
        succ = f"{r['successful_count']}/{r['max_candidates']}"
        ax.text(sp + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{sp:.3f}× ({succ})", va="center", fontsize=8, color="#1e293b")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out = args.results_jsonl.parent / "speedup_per_workload.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
