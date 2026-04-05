# File: app/analysis/supervisor.py
# Purpose: Inspect training metrics for health signals such as missing or invalid loss values.
# License: GPL-3.0-or-later
"""Training supervision helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.graph.state import LoopState, RunResult


DEFAULT_LOSS_KEYS = (
    "loss",
    "train_loss",
    "train_loss_final",
    "val_loss",
    "validation_loss",
)


@dataclass
class TrainingHealth:
    """Summary of whether a run looks trainable and safe to continue."""

    trained: bool
    healthy: bool
    stop_now: bool
    loss_name: str | None
    loss_value: float | None
    reason: str
    recommendation: str
    payload: dict[str, Any]


def _loss_threshold(state: LoopState) -> float | None:
    value = state.constraints.get("max_loss_value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _loss_key_candidates(state: LoopState) -> list[str]:
    configured = state.constraints.get("loss_metric_keys")
    if isinstance(configured, list) and configured:
        return [str(key) for key in configured]
    return list(DEFAULT_LOSS_KEYS)


def _find_loss_metric(result: RunResult, state: LoopState) -> tuple[str | None, float | None]:
    metrics = result.metrics or {}
    for key in _loss_key_candidates(state):
        if key in metrics:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return key, float(value)
    for key, value in metrics.items():
        if "loss" in str(key).lower() and isinstance(value, (int, float)):
            return str(key), float(value)
    return None, None


def assess_training_health(state: LoopState, result: RunResult) -> TrainingHealth:
    """Inspect metrics and decide whether the run looks healthy."""

    loss_name, loss_value = _find_loss_metric(result, state)
    threshold = _loss_threshold(state)

    if loss_value is None:
        reason = "loss metric not reported"
        return TrainingHealth(
            trained=False,
            healthy=False,
            stop_now=False,
            loss_name=None,
            loss_value=None,
            reason=reason,
            recommendation="human_review",
            payload={
                "trained": False,
                "healthy": False,
                "stop_now": False,
                "loss_name": None,
                "loss_value": None,
                "reason": reason,
                "recommendation": "human_review",
            },
        )

    if not math.isfinite(loss_value):
        reason = f"non-finite loss detected in {loss_name}"
        return TrainingHealth(
            trained=True,
            healthy=False,
            stop_now=True,
            loss_name=loss_name,
            loss_value=loss_value,
            reason=reason,
            recommendation="stop",
            payload={
                "trained": True,
                "healthy": False,
                "stop_now": True,
                "loss_name": loss_name,
                "loss_value": loss_value,
                "reason": reason,
                "recommendation": "stop",
            },
        )

    if threshold is not None and loss_value > threshold:
        reason = f"loss exceeds threshold: {loss_value} > {threshold}"
        return TrainingHealth(
            trained=True,
            healthy=False,
            stop_now=True,
            loss_name=loss_name,
            loss_value=loss_value,
            reason=reason,
            recommendation="stop",
            payload={
                "trained": True,
                "healthy": False,
                "stop_now": True,
                "loss_name": loss_name,
                "loss_value": loss_value,
                "reason": reason,
                "recommendation": "stop",
                "threshold": threshold,
            },
        )

    reason = "loss is finite"
    return TrainingHealth(
        trained=True,
        healthy=True,
        stop_now=False,
        loss_name=loss_name,
        loss_value=loss_value,
        reason=reason,
        recommendation="continue",
        payload={
            "trained": True,
            "healthy": True,
            "stop_now": False,
            "loss_name": loss_name,
            "loss_value": loss_value,
            "reason": reason,
            "recommendation": "continue",
            "threshold": threshold,
        },
    )
