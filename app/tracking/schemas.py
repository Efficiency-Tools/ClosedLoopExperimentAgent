# File: app/tracking/schemas.py
# Purpose: Define MLflow-facing trial and study summary schemas.
# License: GPL-3.0-or-later
"""Tracking-layer schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrialTrackingRecord(BaseModel):
    trial_number: int
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    status: str
    run_dir: str
    analysis: dict[str, Any] = Field(default_factory=dict)


class StudySummary(BaseModel):
    study_id: str
    objective_name: str
    direction: str
    total_trials: int
    completed_trials: int
    failed_trials: int
    best_trial: dict[str, Any] | None = None
    notes: str | None = None
