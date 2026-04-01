# File: app/analysis/failure_parser.py
# Purpose: Classify trial failures into structured categories for the study loop.
# License: GPL-3.0-or-later
"""Failure classification helpers."""

from __future__ import annotations

from app.graph.state import FailureCategoryName, RunResult


def classify_failure(result: RunResult) -> FailureCategoryName | None:
    """Map a failed run to a structured failure category."""

    if result.status == "success":
        return None
    text = (result.stderr_summary or "").lower()
    if "out of memory" in text or "oom" in text:
        return "FAIL_OOM"
    if "nan" in text or "inf" in text:
        return "FAIL_NAN"
    if "timeout" in text:
        return "FAIL_TIMEOUT"
    if "config" in text or "invalid" in text:
        return "FAIL_CONFIG"
    if "traceback" in text or "runtime" in text:
        return "FAIL_RUNTIME"
    return "FAIL_UNKNOWN"
