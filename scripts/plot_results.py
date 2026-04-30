"""Plot per-candidate speedups, optionally comparing multiple model runs.

Single-run mode (default):
    python -m scripts.plot_results
    → reads `scripts/results/suite_results.json`
    → writes `scripts/results/suite_speedups.png`

Multi-run / aggregation mode:
    python -m scripts.plot_results path1.json path2.json ...
    → reads each JSON, draws a grouped bar chart with one bar per
      (workload, model_id), 1× baseline reference line.
    → writes `scripts/results/suite_speedups_compare.png` by default
      (override with `SUITE_PLOT_PATH`).

Workloads are ordered by ascending baseline cost (the runner's probe
ordering, taken from the first JSON that exposes it; falls back to a
stable name sort).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "scripts" / "results"
DEFAULT_JSON = RESULTS_DIR / "suite_results.json"
DEFAULT_PNG_SINGLE = RESULTS_DIR / "suite_speedups.png"
DEFAULT_PNG_MULTI = RESULTS_DIR / "suite_speedups_compare.png"

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing results file: {path}")
    return json.loads(path.read_text())


def _row_speedup(row: dict) -> float | None:
    sp = row.get("best_speedup")
    return sp if isinstance(sp, (int, float)) else None


def _plot_single(payload: dict, out_path: Path) -> None:
    rows = list(payload.get("rows", []))
    rows.sort(key=lambda r: (r.get("kind", ""), r.get("name", "")))

    names = [r["name"] for r in rows]
    kinds = [r.get("kind", "?") for r in rows]
    speedups = [_row_speedup(r) if _row_speedup(r) is not None else 1.0 for r in rows]
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
            ax.text(sp + 0.005, yi, f"{sp:.3f}×", va="center", fontsize=8)

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


def _ordered_workloads(payloads: list[dict]) -> list[tuple[str, str]]:
    """Pick a stable workload ordering: probe order from the first payload
    that exposes one; else union of all rows sorted (kind, name)."""

    for p in payloads:
        order = p.get("ordering")
        if order:
            return [(o["name"], o.get("kind", "?")) for o in order]

    seen: dict[str, str] = {}
    for p in payloads:
        for r in p.get("rows", []):
            seen.setdefault(r["name"], r.get("kind", "?"))
    items = sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))
    return [(n, k) for n, k in items]


def _plot_multi(payloads: list[dict], paths: list[Path], out_path: Path) -> None:
    workloads = _ordered_workloads(payloads)
    labels = [
        p.get("model_id") or path.stem for p, path in zip(payloads, paths)
    ]

    # name → kind, name → {label: speedup, label: errored?}
    rows_by_label: list[dict[str, dict]] = []
    for p in payloads:
        idx = {r["name"]: r for r in p.get("rows", [])}
        rows_by_label.append(idx)

    n_models = len(payloads)
    bar_h = 0.8 / n_models
    fig_height = max(5.0, 0.55 * len(workloads) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    yticks = []
    yticklabels = []
    for wi, (name, kind) in enumerate(workloads):
        center = wi
        yticks.append(center)
        yticklabels.append(f"{name}  ({kind})")
        for mi, (label, idx) in enumerate(zip(labels, rows_by_label)):
            row = idx.get(name)
            sp = _row_speedup(row) if row else None
            err = bool(row and "error" in row)
            color = _PALETTE[mi % len(_PALETTE)]
            y = center + (mi - (n_models - 1) / 2.0) * bar_h
            value = sp if sp is not None else 1.0
            edge_alpha = 0.35 if (sp is None) else 1.0
            ax.barh(
                y,
                value,
                height=bar_h * 0.95,
                color=color,
                alpha=(0.45 if sp is None else 0.95),
                edgecolor="#222",
                linewidth=0.4,
            )
            if err:
                ax.text(0.02, y, "err", va="center", fontsize=7, color="#444")
            elif sp is not None:
                ax.text(value + 0.004, y, f"{sp:.3f}×", va="center", fontsize=7)
            else:
                ax.text(0.02, y, "—", va="center", fontsize=7, color="#666")
            _ = edge_alpha  # silence linters

    ax.axvline(1.0, color="#444", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("speedup vs baseline (×)")

    title_lines = ["compilagent suite — pydantic_ai · multi-model comparison"]
    elapsed_strs = []
    for p in payloads:
        e = p.get("elapsed_seconds")
        if isinstance(e, (int, float)):
            elapsed_strs.append(f"{p.get('model_id','?')}: {e/60:.1f} min")
    if elapsed_strs:
        title_lines.append(" · ".join(elapsed_strs))
    gpu = next((p.get("gpu") for p in payloads if p.get("gpu")), "?")
    title_lines.append(gpu)
    ax.set_title("\n".join(title_lines), fontsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_PALETTE[i % len(_PALETTE)], label=lbl)
        for i, lbl in enumerate(labels)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="model")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    out_override = os.environ.get("SUITE_PLOT_PATH")

    if not argv:
        payload = _load(DEFAULT_JSON)
        out_path = Path(out_override) if out_override else DEFAULT_PNG_SINGLE
        _plot_single(payload, out_path)
        print(f"Saved plot → {out_path}")
        return 0

    paths = [Path(a) for a in argv]
    payloads = [_load(p) for p in paths]
    if len(payloads) == 1:
        out_path = Path(out_override) if out_override else DEFAULT_PNG_SINGLE
        _plot_single(payloads[0], out_path)
    else:
        out_path = Path(out_override) if out_override else DEFAULT_PNG_MULTI
        _plot_multi(payloads, paths, out_path)
    print(f"Saved plot → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
