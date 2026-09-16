"""The sandbox must never book its own failures as candidate failures.

Regression tests for the 2026-06-11 fault: a data file read by the package
``__init__`` was missing for seven minutes, every sandbox subprocess died at
import, and 77 in-flight episodes recorded those deaths as candidate compile
failures. Two guards close it: the runner's environment carries
``COMPILAGENT_SANDBOX_CHILD=1``, under which the package skips its
side-effect imports, and a no-JSON exit whose traceback never enters the
candidate raises ``SandboxInfrastructureError`` instead of returning a
failure dict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Everything is reached through the module object: the integrations conftest
# reloads every loaded integration module before each test, so names bound
# at import time (a class in particular) would no longer be the ones the
# reloaded code raises.
from compilagent.integrations.triton_source._internal import sandbox
from compilagent.integrations.triton_source._internal.sandbox_runner import MARKER

SANDBOX_CHILD_ENV = sandbox.SANDBOX_CHILD_ENV

RUNNER_IMPORT_TRACEBACK = textwrap.dedent(
    """\
    Traceback (most recent call last):
      File "<frozen runpy>", line 189, in _run_module_as_main
      File "/x/src/compilagent/integrations/triton_source/__init__.py", line 26, in <module>
        from . import kernelbench_workloads, workloads  # noqa: E402, F401
      File "/x/src/compilagent/integrations/triton_source/kernelbench_workloads.py", line 95, in <module>
        for _entry in _load_manifest()["selected"]:
    FileNotFoundError: [Errno 2] No such file or directory: '/x/kernelbench_manifest.json'
    """
)


def test_runner_import_failure_is_a_runner_fault(tmp_path: Path) -> None:
    assert sandbox._is_runner_fault(RUNNER_IMPORT_TRACEBACK, tmp_path)


def test_traceback_inside_the_candidate_is_not_a_runner_fault(tmp_path: Path) -> None:
    stderr = textwrap.dedent(
        f"""\
        Traceback (most recent call last):
          File "/x/src/compilagent/integrations/triton_source/_internal/sandbox_runner.py", line 80, in _load_module
            spec.loader.exec_module(module)
          File "{tmp_path}/candidate_module.py", line 3, in <module>
            raise SystemExit(1)
        SystemExit: 1
        """
    )
    assert not sandbox._is_runner_fault(stderr, tmp_path)


def test_no_traceback_is_not_a_runner_fault(tmp_path: Path) -> None:
    assert not sandbox._is_runner_fault("some warning\nanother line\n", tmp_path)


def _write_runner(tmp_path: Path, name: str, body: str) -> dict[str, str]:
    """Install a fake runner module OUTSIDE the artifact dir (as the real
    runner lives under the package, not next to the candidate)."""

    runner_dir = tmp_path / "runner"
    runner_dir.mkdir(exist_ok=True)
    (runner_dir / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return {**os.environ, "PYTHONPATH": str(runner_dir)}


def _artifacts(tmp_path: Path) -> Path:
    art = tmp_path / "artifacts"
    art.mkdir(exist_ok=True)
    return art


def _payload(tmp_path: Path) -> Path:
    art = _artifacts(tmp_path)
    payload = art / "sandbox_payload.json"
    payload.write_text(json.dumps({"workdir": str(art)}), encoding="utf-8")
    return payload


def test_runner_that_dies_at_import_raises(tmp_path: Path, monkeypatch) -> None:
    env = _write_runner(
        tmp_path,
        "fake_runner_boom",
        """
        raise FileNotFoundError("kernelbench_manifest.json")
        """,
    )
    monkeypatch.setattr(sandbox, "_RUNNER_MODULE", "fake_runner_boom")
    with pytest.raises(sandbox.SandboxInfrastructureError, match="before evaluating"):
        sandbox._invoke_runner(_payload(tmp_path), _artifacts(tmp_path), 30.0, env=env)


def test_runner_exit_without_traceback_is_a_candidate_failure(
    tmp_path: Path, monkeypatch
) -> None:
    env = _write_runner(
        tmp_path,
        "fake_runner_exit",
        """
        import sys
        sys.exit(3)
        """,
    )
    monkeypatch.setattr(sandbox, "_RUNNER_MODULE", "fake_runner_exit")
    result = sandbox._invoke_runner(_payload(tmp_path), _artifacts(tmp_path), 30.0, env=env)
    assert result["compiled"] is False
    assert "no result JSON" in result["error"]


def test_child_environment_is_marked(tmp_path: Path, monkeypatch) -> None:
    env = _write_runner(
        tmp_path,
        "fake_runner_env",
        f"""
        import json, os
        print({MARKER!r} + json.dumps({{
            "compiled": True, "gates": {{}},
            "child": os.environ.get({SANDBOX_CHILD_ENV!r}),
        }}))
        """,
    )
    monkeypatch.setattr(sandbox, "_RUNNER_MODULE", "fake_runner_env")
    result = sandbox._invoke_runner(_payload(tmp_path), _artifacts(tmp_path), 30.0, env=env)
    assert result["child"] == "1"
    # The inherited-environment path (env=None) marks the child as well.
    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])
    result = sandbox._invoke_runner(_payload(tmp_path), _artifacts(tmp_path), 30.0, env=None)
    assert result["child"] == "1"


def _registered_ids(child: bool) -> set[str]:
    env = {**os.environ}
    env.pop(SANDBOX_CHILD_ENV, None)
    if child:
        env[SANDBOX_CHILD_ENV] = "1"
    code = (
        "import json, compilagent.integrations.triton_source;"
        "from compilagent.core.workload_registry import workload_registry;"
        "print(json.dumps(sorted(workload_registry.ids())))"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def test_package_skips_workload_registration_inside_the_sandbox() -> None:
    assert "softmax_4096" in _registered_ids(child=False)
    assert not _registered_ids(child=True)
