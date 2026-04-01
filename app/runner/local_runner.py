# File: app/runner/local_runner.py
# Purpose: Launch training commands locally and capture stdout, stderr, and timing.
# License: GPL-3.0-or-later
"""Local subprocess-based experiment runner."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from app.runner.base import ExecutionOutcome


class LocalExperimentRunner:
    """Launch a training command on the local machine."""

    def __init__(
        self,
        command_template: list[str],
        env: dict[str, str] | None = None,
        working_dir: Path | None = None,
    ) -> None:
        self.command_template = command_template
        self.env = env or {}
        self.working_dir = working_dir or Path.cwd()

    def _render_command(self, config_path: Path, run_dir: Path) -> list[str]:
        rendered = [
            part.format(config_path=str(config_path), run_dir=str(run_dir))
            for part in self.command_template
        ]
        return rendered

    def run(self, config_path: Path, run_dir: Path, timeout_sec: float) -> ExecutionOutcome:
        """Execute the training command and capture stdout/stderr."""

        run_dir.mkdir(parents=True, exist_ok=True)
        command = self._render_command(config_path, run_dir)
        command_file = run_dir / "command.json"
        command_file.write_text(json.dumps({"command": command}, indent=2))
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=self.working_dir,
                env={**os.environ, **self.env},
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            stdout_path.write_text(completed.stdout or "")
            stderr_path.write_text(completed.stderr or "")
            return ExecutionOutcome(
                run_dir=str(run_dir),
                config_path=str(config_path),
                command=command,
                returncode=completed.returncode,
                duration_sec=time.time() - started,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                timed_out=False,
                extra={"cwd": str(run_dir)},
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "")
            stderr_path.write_text((exc.stderr or "") if isinstance(exc.stderr, str) else "")
            return ExecutionOutcome(
                run_dir=str(run_dir),
                config_path=str(config_path),
                command=command,
                returncode=None,
                duration_sec=time.time() - started,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                timed_out=True,
                extra={"error": "timeout"},
            )
