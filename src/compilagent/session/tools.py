"""Canonical agent toolset for an `OptimizationSession`.

The 8 tools an agent uses to drive a session are declared here as
`ToolDecl`s once. Both pydantic-ai and Claude Agent SDK adapters consume
this single toolset and bind it into their native tool surfaces.

Each handler delegates to a bound `OptimizationSession` method. Methods
return JSON strings; on bad input they raise `ValueError`, which adapters
translate into the harness's native retry/error response.

Backend-provided introspection tools are appended via
`Toolset.with_extra(backend.list_introspection_tools())` outside this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from compilagent.core.tool_decl import ToolDecl
from compilagent.toolset import Toolset

if TYPE_CHECKING:
    from .session import OptimizationSession


_INSPECT_WORKLOAD_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_INSPECT_SEARCH_SPACE_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_PROPOSE_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "interventions_json": {
            "type": "string",
            "description": (
                "JSON list. Each entry: {\"target_kind\": str, "
                "\"target_selector\": str, \"payload\": <value>, "
                "\"rationale\"?: str}. All interventions in the list are "
                "applied together to a single compile + benchmark."
            ),
        },
        "description": {
            "type": "string",
            "description": "One-line human description of the candidate.",
        },
        "expected_effect": {
            "type": "string",
            "description": "Optional expected speedup mechanism / hypothesis.",
        },
    },
    "required": ["interventions_json", "description"],
    "additionalProperties": False,
}

_PROPOSE_CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "plans_json": {
            "type": "string",
            "description": (
                "JSON list. Each entry: {\"description\": str, "
                "\"expected_effect\"?: str, \"interventions\": [...]} where "
                "each intervention has the same shape as in propose_candidate."
            ),
        }
    },
    "required": ["plans_json"],
    "additionalProperties": False,
}

_RUN_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {
            "type": "string",
            "description": "Id returned from propose_candidate / propose_candidates.",
        }
    },
    "required": ["candidate_id"],
    "additionalProperties": False,
}

_RUN_CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_ids_json": {
            "type": "string",
            "description": "JSON list of candidate ids to run in sequence.",
        }
    },
    "required": ["candidate_ids_json"],
    "additionalProperties": False,
}

_NO_ARGS_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def build_session_toolset(session: OptimizationSession) -> Toolset:
    """Build the canonical 8-tool toolset bound to this session instance."""

    def _h_inspect_workload(_args: dict) -> str:
        return session.inspect_workload()

    def _h_inspect_search_space(_args: dict) -> str:
        return session.inspect_search_space()

    def _h_propose_candidate(args: dict) -> str:
        return session.propose_candidate(
            interventions_json=str(args.get("interventions_json", "")),
            description=str(args.get("description", "")),
            expected_effect=str(args.get("expected_effect", "")),
        )

    def _h_propose_candidates(args: dict) -> str:
        return session.propose_candidates(
            plans_json=str(args.get("plans_json", ""))
        )

    def _h_run_candidate(args: dict) -> str:
        return session.run_candidate(candidate_id=str(args.get("candidate_id", "")))

    def _h_run_candidates(args: dict) -> str:
        return session.run_candidates(
            candidate_ids_json=str(args.get("candidate_ids_json", ""))
        )

    def _h_synthesize_findings(_args: dict) -> str:
        return session.synthesize_findings()

    def _h_compare_runs(_args: dict) -> str:
        return session.compare_runs()

    tools: tuple[ToolDecl, ...] = (
        ToolDecl(
            name="inspect_workload",
            description=(
                "Return the workload spec, baseline timing, analysis summary, "
                "device capability, and any cross-run hints."
            ),
            args_schema=_INSPECT_WORKLOAD_SCHEMA,
            handler=_h_inspect_workload,
            read_only=True,
        ),
        ToolDecl(
            name="inspect_search_space",
            description=(
                "Return the derived lever catalog with per-lever evidence so "
                "the agent can reason about which axes matter for this workload."
            ),
            args_schema=_INSPECT_SEARCH_SPACE_SCHEMA,
            handler=_h_inspect_search_space,
            read_only=True,
        ),
        ToolDecl(
            name="propose_candidate",
            description=(
                "Register a multi-intervention candidate. Combine 2-4 levers per "
                "candidate when forming non-trivial hypotheses. Returns the new "
                "candidate id which run_candidate can then run."
            ),
            args_schema=_PROPOSE_CANDIDATE_SCHEMA,
            handler=_h_propose_candidate,
            read_only=False,
        ),
        ToolDecl(
            name="propose_candidates",
            description=(
                "Register several candidates at once. Useful for setting up a "
                "small batch of hypotheses (e.g. 3) before running them."
            ),
            args_schema=_PROPOSE_CANDIDATES_SCHEMA,
            handler=_h_propose_candidates,
            read_only=False,
        ),
        ToolDecl(
            name="run_candidate",
            description=(
                "Compile + time + correctness-check a previously proposed "
                "candidate. Returns median_ms, speedup_vs_baseline, "
                "correctness_ok, plus a hint when the run failed."
            ),
            args_schema=_RUN_CANDIDATE_SCHEMA,
            handler=_h_run_candidate,
            read_only=False,
        ),
        ToolDecl(
            name="run_candidates",
            description=(
                "Run a batch of previously proposed candidates in sequence and "
                "return per-candidate results plus an aggregate summary."
            ),
            args_schema=_RUN_CANDIDATES_SCHEMA,
            handler=_h_run_candidates,
            read_only=False,
        ),
        ToolDecl(
            name="synthesize_findings",
            description=(
                "Aggregate the current run's results: per-target_kind speedup "
                "distributions, co-occurring lever pairs in winners, and "
                "interventions that appear in failures."
            ),
            args_schema=_NO_ARGS_SCHEMA,
            handler=_h_synthesize_findings,
            read_only=True,
        ),
        ToolDecl(
            name="compare_runs",
            description=(
                "Return a leaderboard of (baseline + judged candidates) sorted "
                "by median_ms ascending."
            ),
            args_schema=_NO_ARGS_SCHEMA,
            handler=_h_compare_runs,
            read_only=True,
        ),
    )
    return Toolset(tools=tools)
