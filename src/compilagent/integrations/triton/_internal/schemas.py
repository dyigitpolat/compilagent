"""Internal schemas used by the lifted Triton harness + analysis.

These types are private to the triton integration. The boundary types the
core sees (`compilagent.core.CompileResult`, `Analysis`, …) live in
`compilagent.core.analysis`; the harness adapter maps the internal
`CompileResult` defined here into that.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DecisionKind(StrEnum):
    COALESCING = "coalescing"
    MATMUL = "matmul"
    UNKNOWN = "unknown"


class KernelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    path: Path | None = None
    entrypoint: str
    shapes: list[tuple[int, ...]] = Field(default_factory=list)
    dtypes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "name", "entrypoint")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class CompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kernel_id: str
    meta: dict[str, Any] = Field(default_factory=dict)
    force_recompile: bool = True
    dump_ir: bool = True
    stage_hook_key: str | None = None
    candidate_id: str | None = None
    diagnostic_env: dict[str, str] = Field(default_factory=dict)


class CompileArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    path: Path | None = None
    inline_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompileResult(BaseModel):
    """Internal compile-result type the lifted harness produces.

    The triton-backend adapter maps this into `compilagent.core.CompileResult`
    before returning to the session.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"compile-{uuid4().hex[:12]}")
    kernel_id: str
    candidate_id: str | None = None
    ok: bool
    cache_hit: bool | None = None
    target: str | None = None
    source_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[CompileArtifact] = Field(default_factory=list)
    diagnostics: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DecisionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"decision-{uuid4().hex[:12]}")
    run_id: str | None = None
    kind: DecisionKind = DecisionKind.UNKNOWN
    op_name: str | None = None
    op_location: str | None = None
    tensor_shape: tuple[int, ...] | None = None
    element_bitwidth: int | None = None
    chosen_order: tuple[int, ...] | None = None
    size_per_thread: tuple[int, ...] | None = None
    num_warps: int | None = None
    threads_per_warp: int | None = None
    shape_per_cta: tuple[int, ...] | None = None
    mma_version: int | None = None
    warps_per_tile: tuple[int, ...] | None = None
    evidence: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
