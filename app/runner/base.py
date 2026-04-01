# File: app/runner/base.py
# Purpose: Define the shared experiment runner protocol and process outcome schema.
# License: GPL-3.0-or-later
"""Experiment runner abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ExecutionOutcome:
    """Raw process-level outcome before result parsing."""

    run_dir: str
    config_path: str
    command: list[str]
    returncode: int | None
    duration_sec: float
    stdout_path: str
    stderr_path: str
    timed_out: bool = False
    extra: dict[str, Any] | None = None


class ExperimentRunner(Protocol):
    """Protocol for launching an experiment."""

    def run(self, config_path: Path, run_dir: Path, timeout_sec: float) -> ExecutionOutcome:
        """Run the experiment and return the process outcome."""
