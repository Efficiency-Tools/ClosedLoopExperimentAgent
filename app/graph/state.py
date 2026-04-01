# File: app/graph/state.py
# Purpose: Define the central study state, run results, and JSON persistence helpers.
# License: GPL-3.0-or-later
"""Typed state and persistence helpers for the study loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


RunStatus = Literal["success", "failed", "crashed", "timeout"]
Decision = Literal["continue", "stop", "human_review"]
FailureCategoryName = Literal[
    "FAIL_OOM",
    "FAIL_NAN",
    "FAIL_TIMEOUT",
    "FAIL_CONFIG",
    "FAIL_RUNTIME",
    "FAIL_UNKNOWN",
]


class RunResult(BaseModel):
    """Standardized training outcome."""

    status: RunStatus
    primary_metric: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    stderr_summary: str | None = None
    duration_sec: float = 0.0
    run_dir: str


class LoopState(BaseModel):
    """Central state carried across the LangGraph workflow."""

    study_id: str
    objective_name: str
    direction: Literal["maximize", "minimize"]
    budget_total_trials: int
    budget_used_trials: int = 0
    budget_gpu_hours: float = 0.0
    used_gpu_hours: float = 0.0
    baseline_config: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    history_summary: dict[str, Any] = Field(default_factory=dict)
    best_trial: dict[str, Any] | None = None
    current_proposal: dict[str, Any] | None = None
    current_result: RunResult | None = None
    current_analysis: dict[str, Any] | None = None
    failure_count: int = 0
    plateau_count: int = 0
    decision: Decision | None = None
    recent_failed_signatures: list[str] = Field(default_factory=list)
    study_name: str = "closed-loop-study"
    study_dir: str = "runs/study"
    storage_url: str = "sqlite:///runs/study/optuna.sqlite3"
    tracking_uri: str = "file:./runs/mlruns"
    mlflow_experiment_name: str = "closed-loop-experiment-agent"
    training_command: list[str] = Field(default_factory=list)
    timeout_sec: float = 3600.0
    trial_number: int | None = None
    run_dir: str | None = None
    study_run_id: str | None = None
    resume: bool = False
    current_execution: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class TrialProposal(BaseModel):
    """A parameter proposal emitted by Optuna."""

    trial_number: int
    params: dict[str, Any]
    signature: str
    validated: bool = False
    validation_errors: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Guardrail validation result."""

    valid: bool
    reason: str | None = None
    blocked_keys: list[str] = Field(default_factory=list)


def load_state(path: Path) -> LoopState | None:
    """Load a persisted LoopState if it exists."""

    if not path.exists():
        return None
    return LoopState.model_validate_json(path.read_text())


def save_state(state: LoopState, path: Path) -> None:
    """Persist the loop state as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2, exclude_none=True))
