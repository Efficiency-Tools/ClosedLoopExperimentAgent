"""Tests for the local runner command rendering."""

from __future__ import annotations

import sys
from pathlib import Path

from app.runner.local_runner import LocalExperimentRunner


def test_python_command_uses_current_interpreter():
    runner = LocalExperimentRunner(["python", "-V"])
    rendered = runner._render_command(Path("config.yaml"), Path("runs/trial_0001"))
    assert rendered[0] == sys.executable
