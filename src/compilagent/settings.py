"""Env-driven runtime settings for the compilagent core.

Harness and backend identifiers are plain strings — the registries validate
membership at runtime, so adding a new harness or backend never touches this
module. Harness-specific knobs (Claude SDK budget, permission mode, ...)
flow through `harness_extra` (populated from `COMPILAGENT_HARNESS_EXTRA_JSON`)
and the harness adapter consumes whatever it recognises.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

DEFAULT_MODEL = "anthropic:claude-opus-4-7"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_HARNESS = "pydantic_ai"
DEFAULT_WORKSPACE_DIR_NAME = ".compilagent"


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _env_value(name: str, dotenv: dict[str, str], default: str | None = None) -> str | None:
    return os.environ.get(name) or dotenv.get(name) or default


def _env_int(name: str, dotenv: dict[str, str], default: int) -> int:
    value = _env_value(name, dotenv)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, dotenv: dict[str, str], default: float) -> float:
    value = _env_value(name, dotenv)
    if value is None or value == "":
        return default
    return float(value)


class CompilagentSettings(BaseModel):
    """Runtime settings.

    Secret values are excluded from repr and serialized artifacts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anthropic_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    mistral_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    openai_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)

    model_name: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    max_tokens: int = 8192
    temperature: float = 0.2
    max_candidates: int = 4
    max_benchmark_seconds: int = 120
    noise_threshold_pct: float = 2.0
    workspace_dir_name: str = DEFAULT_WORKSPACE_DIR_NAME

    harness: str = DEFAULT_HARNESS
    harness_extra: dict[str, Any] = Field(default_factory=dict)

    integrations: tuple[str, ...] = ()

    @field_validator("max_tokens", "max_candidates", "max_benchmark_seconds")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("temperature")
    @classmethod
    def _valid_temperature(cls, value: float) -> float:
        if value < 0:
            raise ValueError("temperature must be non-negative")
        return value

    @field_validator("noise_threshold_pct")
    @classmethod
    def _valid_noise_threshold(cls, value: float) -> float:
        if value < 0:
            raise ValueError("noise threshold must be non-negative")
        return value

    @field_validator("harness")
    @classmethod
    def _valid_harness(cls, value: str) -> str:
        normalized = value.strip().replace("-", "_")
        if not normalized:
            raise ValueError("harness id must be a non-empty string")
        return normalized

    @classmethod
    def from_env(
        cls,
        *,
        dotenv_path: Path | str | None = None,
        project_root: Path | str | None = None,
    ) -> CompilagentSettings:
        root = Path.cwd() if project_root is None else Path(project_root)
        env_path = root / ".env" if dotenv_path is None else Path(dotenv_path)
        dotenv = _parse_env_file(env_path)

        anthropic_key = _env_value("ANTHROPIC_API_KEY", dotenv)
        mistral_key = _env_value("MISTRAL_API_KEY", dotenv)
        openai_key = _env_value("OPENAI_API_KEY", dotenv)

        harness_extra_raw = _env_value("COMPILAGENT_HARNESS_EXTRA_JSON", dotenv, "") or ""
        harness_extra: dict[str, Any] = {}
        if harness_extra_raw.strip():
            try:
                parsed = json.loads(harness_extra_raw)
                if isinstance(parsed, dict):
                    harness_extra = parsed
            except json.JSONDecodeError:
                harness_extra = {}

        integrations_raw = _env_value("COMPILAGENT_INTEGRATIONS", dotenv, "") or ""
        integrations = tuple(
            entry.strip() for entry in integrations_raw.split(",") if entry.strip()
        )

        data: dict[str, Any] = {
            "anthropic_api_key": SecretStr(anthropic_key) if anthropic_key else None,
            "mistral_api_key": SecretStr(mistral_key) if mistral_key else None,
            "openai_api_key": SecretStr(openai_key) if openai_key else None,
            "model_name": _env_value("COMPILAGENT_MODEL", dotenv, DEFAULT_MODEL),
            "reasoning_effort": _env_value(
                "COMPILAGENT_REASONING_EFFORT", dotenv, DEFAULT_REASONING_EFFORT
            ),
            "max_tokens": _env_int("COMPILAGENT_MAX_TOKENS", dotenv, 8192),
            "temperature": _env_float("COMPILAGENT_TEMPERATURE", dotenv, 0.2),
            "max_candidates": _env_int("COMPILAGENT_MAX_CANDIDATES", dotenv, 4),
            "max_benchmark_seconds": _env_int(
                "COMPILAGENT_MAX_BENCHMARK_SECONDS", dotenv, 120
            ),
            "noise_threshold_pct": _env_float(
                "COMPILAGENT_NOISE_THRESHOLD_PCT", dotenv, 2.0
            ),
            "workspace_dir_name": _env_value(
                "COMPILAGENT_WORKSPACE_DIR", dotenv, DEFAULT_WORKSPACE_DIR_NAME
            ),
            "harness": _env_value("COMPILAGENT_HARNESS", dotenv, DEFAULT_HARNESS),
            "harness_extra": harness_extra,
            "integrations": integrations,
        }
        return cls(**data)

    def public_metadata(self) -> dict[str, Any]:
        """Return non-secret settings suitable for episode artifacts."""

        return {
            "model_name": self.model_name,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_candidates": self.max_candidates,
            "max_benchmark_seconds": self.max_benchmark_seconds,
            "noise_threshold_pct": self.noise_threshold_pct,
            "workspace_dir_name": self.workspace_dir_name,
            "harness": self.harness,
            "harness_extra": dict(self.harness_extra),
            "integrations": list(self.integrations),
        }
