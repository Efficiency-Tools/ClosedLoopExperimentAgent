"""Tests for evaluation report aggregation and deltas."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.evaluator import _delta_for_case, _mean
from app.evaluation.schemas import EvaluationCaseResult


def test_mean_ignores_missing_values():
    assert _mean([1.0, None, 3.0]) == 2.0


def test_delta_for_case_computes_numeric_differences():
    current = EvaluationCaseResult(
        name="case",
        study_name="current",
        study_dir="runs/current",
        objective_name="val_auc",
        direction="maximize",
        total_trials=10,
        completed_trials=8,
        failed_trials=2,
        best_value=0.91,
        duration_sec=12.0,
    )
    reference = EvaluationCaseResult(
        name="case",
        study_name="reference",
        study_dir="runs/reference",
        objective_name="val_auc",
        direction="maximize",
        total_trials=10,
        completed_trials=6,
        failed_trials=4,
        best_value=0.88,
        duration_sec=15.0,
    )
    delta = _delta_for_case(current, reference)
    assert delta.best_value_delta == 0.03
    assert delta.completed_trials_delta == 2
    assert delta.failed_trials_delta == -2
    assert delta.duration_sec_delta == -3.0
