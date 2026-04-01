# File: app/analysis/heuristic_analyzer.py
# Purpose: Produce deterministic trial analysis and stop/review signals.
# License: GPL-3.0-or-later
"""Structured, programmatic trial analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.analysis.failure_parser import classify_failure
from app.graph.state import LoopState, RunResult


@dataclass
class AnalysisOutcome:
    """Compact analysis payload for a trial."""

    status: str
    failure_category: str | None
    primary_metric: float | None
    improved: bool
    plateau_signal: bool
    summary: str
    recommendation: str
    payload: dict[str, Any]


def analyze_result(state: LoopState, result: RunResult) -> AnalysisOutcome:
    """Produce a deterministic analysis without any LLM involvement."""

    best_value = state.best_trial.get("value") if state.best_trial else None
    improved = False
    if result.status == "success" and result.primary_metric is not None and best_value is not None:
        if state.direction == "maximize":
            improved = result.primary_metric > float(best_value)
        else:
            improved = result.primary_metric < float(best_value)
    elif result.status == "success" and best_value is None and result.primary_metric is not None:
        improved = True

    plateau_signal = state.plateau_count >= 3 and not improved
    failure_category = classify_failure(result)
    recommendation = "continue"
    if result.status != "success":
        recommendation = "investigate_failure"
    elif plateau_signal:
        recommendation = "consider_stop_or_review"

    summary = (
        f"status={result.status}, metric={result.primary_metric}, improved={improved}, "
        f"failure_category={failure_category}"
    )
    return AnalysisOutcome(
        status=result.status,
        failure_category=failure_category,
        primary_metric=result.primary_metric,
        improved=improved,
        plateau_signal=plateau_signal,
        summary=summary,
        recommendation=recommendation,
        payload={
            "status": result.status,
            "failure_category": failure_category,
            "primary_metric": result.primary_metric,
            "improved": improved,
            "plateau_signal": plateau_signal,
            "summary": summary,
            "recommendation": recommendation,
        },
    )
