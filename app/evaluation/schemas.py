# File: app/evaluation/schemas.py
# Purpose: Define benchmark suite and evaluation report schemas.
# License: GPL-3.0-or-later
"""Schemas for system evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvaluationCaseSpec(BaseModel):
    name: str
    baseline_config: Path
    search_space: Path
    train_command: str
    timeout_sec: float = 3600.0
    study_name: str | None = None
    study_dir: Path | None = None
    storage_path: Path | None = None
    tracking_uri: str = "file:./runs/mlruns"
    mlflow_experiment_name: str = "closed-loop-experiment-agent"


class EvaluationSuiteSpec(BaseModel):
    suite_name: str
    version_tag: str | None = None
    cases: list[EvaluationCaseSpec] = Field(default_factory=list)


class EvaluationCaseResult(BaseModel):
    name: str
    study_name: str
    study_dir: str
    objective_name: str
    direction: str
    total_trials: int
    completed_trials: int
    failed_trials: int
    best_value: float | None = None
    best_trial: dict[str, Any] | None = None
    duration_sec: float = 0.0
    notes: str | None = None


class EvaluationDelta(BaseModel):
    name: str
    best_value_delta: float | None = None
    completed_trials_delta: int | None = None
    failed_trials_delta: int | None = None
    duration_sec_delta: float | None = None


class EvaluationReport(BaseModel):
    suite_name: str
    version_tag: str
    created_at: str
    cases: list[EvaluationCaseResult] = Field(default_factory=list)
    comparison_to: str | None = None
    deltas: list[EvaluationDelta] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

