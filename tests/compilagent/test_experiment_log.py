"""Unit tests for the `ExperimentLog` reader API (ticket E9).

The log itself has always been append-only/write-mostly; these tests pin
down the reader half (`read_all` / `recall` / `recall_failures`) that the
`ExperimentLogPolicy` consumes.
"""

from __future__ import annotations

import json

from compilagent.storage.experiment_log import ExperimentLog


def _row(**overrides) -> dict:
    base = {
        "run_id": "run-1",
        "workload_id": "softmax_4096",
        "backend_id": "triton_source",
        "family": "reduction",
        "arch": "cuda:sm_120",
        "successful": True,
        "speedup": 1.5,
        "interventions": [],
    }
    base.update(overrides)
    return base


def test_read_all_tolerates_missing_file_and_corrupt_lines(tmp_path):
    log = ExperimentLog(tmp_path)
    assert log.read_all() == []

    log.append(_row(speedup=1.1))
    with log.path.open("a", encoding="utf-8") as f:
        f.write("{not json\n\n")
    log.append(_row(speedup=1.2))

    rows = log.read_all()
    assert [r["speedup"] for r in rows] == [1.1, 1.2]
    assert all("timestamp" in r for r in rows)


def test_recall_filters_by_workload_backend_family_arch(tmp_path):
    log = ExperimentLog(tmp_path)
    log.append(_row(speedup=1.5))
    log.append(_row(workload_id="other", speedup=9.0))
    log.append(_row(backend_id="triton", speedup=9.0))
    log.append(_row(family="matmul", speedup=9.0))
    log.append(_row(arch="cuda:sm_90", speedup=9.0))
    log.append(_row(successful=False, speedup=9.0))

    rows = log.recall(
        workload_id="softmax_4096",
        backend_id="triton_source",
        family="reduction",
        arch="cuda:sm_120",
    )
    assert len(rows) == 1
    assert rows[0]["speedup"] == 1.5


def test_recall_sorts_by_speedup_and_honours_top_n(tmp_path):
    log = ExperimentLog(tmp_path)
    for sp in (1.1, 1.9, 1.4, 1.7):
        log.append(_row(speedup=sp))

    rows = log.recall(workload_id="softmax_4096", top_n=2)
    assert [r["speedup"] for r in rows] == [1.9, 1.7]


def test_recall_family_only_spans_workloads(tmp_path):
    """The C6 use case: recall by (family, backend, arch) — not workload id."""

    log = ExperimentLog(tmp_path)
    log.append(_row(workload_id="softmax_4096", speedup=1.8))
    log.append(_row(workload_id="sum_reduce_1m", speedup=1.3))
    log.append(_row(workload_id="matmul_relu", family="matmul", speedup=2.0))

    rows = log.recall(
        family="reduction", backend_id="triton_source", arch="cuda:sm_120"
    )
    assert {r["workload_id"] for r in rows} == {"softmax_4096", "sum_reduce_1m"}


def test_recall_failures_most_recent_first(tmp_path):
    log = ExperimentLog(tmp_path)
    log.append(_row(successful=False, diagnostics="older", timestamp=100.0))
    log.append(_row(successful=False, diagnostics="newer", timestamp=200.0))
    log.append(_row(successful=True))

    rows = log.recall_failures(workload_id="softmax_4096")
    assert [r["diagnostics"] for r in rows] == ["newer", "older"]


def test_append_is_one_json_line_per_row(tmp_path):
    log = ExperimentLog(tmp_path)
    log.append(_row())
    log.append(_row())
    lines = [
        line for line in log.path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 2
    assert all(json.loads(line)["workload_id"] == "softmax_4096" for line in lines)
