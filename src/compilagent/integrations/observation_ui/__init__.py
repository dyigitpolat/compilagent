"""Observation UI integration: FastAPI app + the existing SPA.

Read-only viewer for the on-disk `TraceStore`; backend-specific artifact
suffixes are routed through the core's `artifact_renderer_registry`, never
hard-coded here. Trigger optimization runs via `POST /api/runs/workload`
and watch them stream over `/ws` or `/api/stream`.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
