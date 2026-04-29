"""`OptimizationWorkspace` — sandboxed FS rooted under a session cwd.

Every artifact a session writes (per-run IR dumps, episode JSON, traces,
experiment log) lives under one workspace root, resolvable through
`workspace.resolve(...)` which guarantees the path stays inside the root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OptimizationWorkspace:
    """Path-safe workspace rooted inside a session cwd."""

    session_cwd: Path
    root_name: str = ".compilagent"

    @property
    def root(self) -> Path:
        return (self.session_cwd / self.root_name).resolve()

    @property
    def workloads_dir(self) -> Path:
        return self.root / "workloads"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    def ensure(self) -> OptimizationWorkspace:
        for path in (
            self.root,
            self.workloads_dir,
            self.reports_dir,
            self.traces_dir,
            self.memory_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def resolve(self, relative_path: str | Path) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                "Path must stay inside the optimization workspace."
            ) from exc
        return candidate

    def run_dir(self, workload_id: str, run_id: str, *parts: str) -> Path:
        return self.resolve(
            Path("workloads")
            / _safe_name(workload_id)
            / "runs"
            / _safe_name(run_id)
            / Path(*parts)
        )

    def baseline_dir(self, workload_id: str, run_id: str) -> Path:
        return self.run_dir(workload_id, run_id, "baseline")

    def candidate_dir(self, workload_id: str, run_id: str, candidate_id: str) -> Path:
        return self.run_dir(workload_id, run_id, "candidates", _safe_name(candidate_id))

    def episode_path(self, workload_id: str, run_id: str) -> Path:
        return self.run_dir(workload_id, run_id, "episode.json")


def _safe_name(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "-" for c in value)
    safe = safe.strip(".-")
    if not safe:
        raise ValueError("identifier must contain at least one safe character")
    return safe
