# File: app/guards/constraints.py
# Purpose: Enforce budget limits and failure-based stop conditions for the study loop.
# License: GPL-3.0-or-later
"""Budget and stop-condition helpers."""

from __future__ import annotations

from app.graph.state import LoopState


def budget_exhausted(state: LoopState) -> bool:
    """Check if the study should stop due to budget limits."""

    if state.budget_used_trials >= state.budget_total_trials:
        return True
    if state.budget_gpu_hours > 0 and state.used_gpu_hours >= state.budget_gpu_hours:
        return True
    return False


def too_many_failures(state: LoopState) -> bool:
    """Check if the study should escalate after consecutive failures."""

    max_failures = int(state.constraints.get("max_consecutive_failures", 3))
    return state.failure_count >= max_failures
