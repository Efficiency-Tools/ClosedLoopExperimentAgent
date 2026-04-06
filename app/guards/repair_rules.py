# File: app/guards/repair_rules.py
# Purpose: Placeholder for safe proposal repair rules in future iterations.
# License: GPL-3.0-or-later
"""Minimal repair hooks for safe, automatic recovery."""

from __future__ import annotations

import copy
from typing import Any


_LR_KEY_HINTS = {
    "lr",
    "learning_rate",
    "step_size",
}
_BATCH_KEY_HINTS = {
    "batch_size",
    "micro_batch_size",
    "train_batch_size",
}


def _repair_numeric_fields(node: Any, *, key_hints: set[str], scale: float, floor: float) -> tuple[Any, bool]:
    """Recursively scale selected numeric leaf nodes."""

    repaired = False
    if isinstance(node, dict):
        updated: dict[str, Any] = {}
        for key, value in node.items():
            lowered = str(key).lower()
            if lowered in key_hints and isinstance(value, (int, float)):
                scaled = max(float(value) * scale, floor)
                updated[key] = type(value)(scaled) if isinstance(value, int) else scaled
                repaired = True
                continue
            new_value, child_repaired = _repair_numeric_fields(value, key_hints=key_hints, scale=scale, floor=floor)
            updated[key] = new_value
            repaired = repaired or child_repaired
        return updated, repaired
    if isinstance(node, list):
        updated_list = []
        for item in node:
            new_item, child_repaired = _repair_numeric_fields(item, key_hints=key_hints, scale=scale, floor=floor)
            updated_list.append(new_item)
            repaired = repaired or child_repaired
        return updated_list, repaired
    return node, False


def repair_proposal(proposal: dict[str, Any], analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply conservative automatic repairs based on watchdog analysis.

    The function intentionally stays conservative: it only nudges common
    training hyperparameters when the analysis points to numerical instability
    or memory pressure. If nothing safe can be inferred, the proposal is
    returned unchanged.
    """

    analysis = analysis or {}
    failure_category = str(analysis.get("failure_category") or "").upper()
    reason = str(analysis.get("reason") or "").lower()
    repaired = copy.deepcopy(proposal)

    if failure_category == "FAIL_OOM" or "out of memory" in reason or "oom" in reason:
        repaired, _ = _repair_numeric_fields(repaired, key_hints=_BATCH_KEY_HINTS, scale=0.5, floor=1.0)
        return repaired

    if (
        failure_category in {"FAIL_NAN", "FAIL_RUNTIME", "FAIL_UNKNOWN"}
        or "loss exceeds threshold" in reason
        or "loss diverged" in reason
        or "exploding loss" in reason
    ):
        repaired, _ = _repair_numeric_fields(repaired, key_hints=_LR_KEY_HINTS, scale=0.5, floor=1e-8)
        return repaired

    return repaired
