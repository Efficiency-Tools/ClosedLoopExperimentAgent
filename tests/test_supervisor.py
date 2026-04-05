"""Tests for training supervision and loss-health detection."""

from __future__ import annotations

from app.analysis.supervisor import assess_training_health
from app.graph.state import LoopState, RunResult


def _state() -> LoopState:
    return LoopState(
        study_id="study",
        objective_name="val_auc",
        direction="maximize",
        budget_total_trials=10,
        constraints={"max_consecutive_failures": 3},
    )


def test_assess_training_health_accepts_finite_loss():
    result = RunResult(
        status="success",
        primary_metric=0.9,
        metrics={"train_loss_final": 0.12},
        run_dir="runs/trial_0001",
    )
    health = assess_training_health(_state(), result)
    assert health.trained
    assert health.healthy
    assert not health.stop_now


def test_assess_training_health_stops_on_non_finite_loss():
    result = RunResult(
        status="success",
        primary_metric=0.9,
        metrics={"train_loss_final": float("nan")},
        run_dir="runs/trial_0001",
    )
    health = assess_training_health(_state(), result)
    assert health.trained
    assert not health.healthy
    assert health.stop_now


def test_assess_training_health_requests_review_when_loss_missing():
    result = RunResult(
        status="success",
        primary_metric=0.9,
        metrics={"val_auc": 0.9},
        run_dir="runs/trial_0001",
    )
    health = assess_training_health(_state(), result)
    assert not health.trained
    assert not health.healthy
    assert not health.stop_now
    assert health.recommendation == "human_review"
