from __future__ import annotations

from compilagent.core.analysis import Analysis
from compilagent.core.candidate_policy import NullPolicy
from compilagent.core.workload import WorkloadKind, WorkloadSpec


def test_null_policy_returns_no_hints():
    spec = WorkloadSpec(
        id="demo",
        title="Demo",
        description="Test fixture",
        kind=WorkloadKind.KERNEL,
        backend_id="fake",
    )
    policy = NullPolicy()
    hints = policy.consult(workload=spec, analysis=Analysis(), family=None, arch="cpu")
    assert hints == ()
    assert policy.name == "null"
