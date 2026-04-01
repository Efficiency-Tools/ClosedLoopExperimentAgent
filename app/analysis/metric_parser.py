# File: app/analysis/metric_parser.py
# Purpose: Parse the training output contract into the normalized RunResult schema.
# License: GPL-3.0-or-later
"""Parse results.json into a standardized RunResult."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.graph.state import RunResult


def parse_results_json(run_dir: Path, duration_sec: float, stderr_summary: str | None = None) -> RunResult:
    """Load the standardized results.json file or synthesize a failure result."""

    results_path = run_dir / "results.json"
    if not results_path.exists():
        return RunResult(
            status="crashed",
            primary_metric=None,
            metrics={},
            artifacts={"run_dir": str(run_dir)},
            stderr_summary=stderr_summary or "missing results.json",
            duration_sec=duration_sec,
            run_dir=str(run_dir),
        )

    data: dict[str, Any] = json.loads(results_path.read_text())
    status = data.get("status", "failed")
    metrics = data.get("metrics", {})
    artifacts = {"results_json": str(results_path)}
    return RunResult(
        status=status,
        primary_metric=data.get("primary_metric"),
        metrics=metrics,
        artifacts=artifacts,
        stderr_summary=stderr_summary or data.get("notes"),
        duration_sec=duration_sec,
        run_dir=str(run_dir),
    )
