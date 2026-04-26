"""Side-by-side bar chart: Mistral large vs. Claude Opus 4.7 on every workload.

Usage:
    env/bin/python scripts/experiments/plot_mistral_vs_opus.py \\
        --mistral runs/study/all-mistral-t8-merged/results.jsonl \\
        --opus    runs/study/all-opus-t8/results.jsonl \\
        --out     runs/study/mistral_vs_opus_t8.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["workload_id"]] = r
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mistral", type=Path, required=True)
    p.add_argument("--opus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    m = load(args.mistral)
    o = load(args.opus)
    workloads = sorted(set(m.keys()) | set(o.keys()))
    # Drop workloads where either model hit a hard error (compile timeout / OOM
    # / infrastructure failure) — they're not meaningful in a head-to-head.
    workloads = [
        w for w in workloads
        if not (m.get(w) or {}).get("error")
        and not (o.get(w) or {}).get("error")
    ]

    # Sort by Opus speedup descending; workloads with no Opus data sink to the
    # bottom (sorted by Mistral within that group).
    def _opus_key(w: str) -> tuple[int, float]:
        sp_o = (o.get(w) or {}).get("best_speedup")
        sp_m = (m.get(w) or {}).get("best_speedup")
        if isinstance(sp_o, (int, float)):
            return (0, -float(sp_o))                       # primary: Opus desc
        return (1, -(float(sp_m) if isinstance(sp_m, (int, float)) else 0.0))
    workloads.sort(key=_opus_key)

    n = len(workloads)
    y = np.arange(n)
    h = 0.4
    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * n + 1.5)), dpi=150)

    def _bar_value(d: dict) -> float | None:
        v = d.get("best_speedup")
        if isinstance(v, (int, float)):
            return float(v)
        # Cell ran cleanly but no candidate beat baseline — show as 1.000×
        # (not "no data") so the comparison stays apples-to-apples.
        if d and not d.get("error"):
            return 1.0
        return None

    def _row_label(workload: str) -> str:
        backend = (m.get(workload) or o.get(workload) or {}).get("backend_id", "")
        return f"{workload}  ({backend})"

    m_vals = [_bar_value(m.get(w) or {}) for w in workloads]
    o_vals = [_bar_value(o.get(w) or {}) for w in workloads]

    bars_m = ax.barh(
        y - h / 2,
        [v if v is not None else 0 for v in m_vals],
        height=h, color="#0ea5e9", edgecolor="#1e293b", linewidth=0.6,
        label="Mistral large",
    )
    bars_o = ax.barh(
        y + h / 2,
        [v if v is not None else 0 for v in o_vals],
        height=h, color="#f97316", edgecolor="#1e293b", linewidth=0.6,
        label="Claude Opus 4.7",
    )

    for bar, v, r in zip(bars_m, m_vals, [m.get(w) for w in workloads]):
        if v is None:
            continue
        succ = f"{r.get('successful_count','?')}/{r.get('max_candidates','?')}"
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}× ({succ})", va="center", fontsize=7,
                color="#0c4a6e")
    for bar, v, r in zip(bars_o, o_vals, [o.get(w) for w in workloads]):
        if v is None:
            label_x = 1.005
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                    "no data", va="center", fontsize=7, color="#9a3412")
            continue
        succ = f"{r.get('successful_count','?')}/{r.get('max_candidates','?')}"
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}× ({succ})", va="center", fontsize=7,
                color="#9a3412")

    ax.axvline(1.0, color="#dc2626", linestyle="-", linewidth=2.0, zorder=10)
    ax.set_yticks(y)
    ax.set_yticklabels([_row_label(w) for w in workloads], fontsize=9)
    # Highlight workloads where Opus beat Mistral: bold + red label.
    for tick, w, sp_m, sp_o in zip(ax.get_yticklabels(), workloads, m_vals, o_vals):
        if (
            isinstance(sp_o, (int, float)) and isinstance(sp_m, (int, float))
            and sp_o > sp_m
        ):
            tick.set_color("#dc2626")
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    finite_vals = [v for v in (m_vals + o_vals) if v is not None]
    right = max([1.20] + [v + 0.04 for v in finite_vals])
    ax.set_xlim(left=0.95, right=right)
    ax.set_xlabel("speedup vs. baseline (1.0× = no change)")
    ax.set_title(
        "Mistral large vs. Claude Opus 4.7 — pydantic-ai · 8 trials · 1 seed\n"
        "best validated speedup per workload (annotation: speedup × successful/total)",
        fontsize=11,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
