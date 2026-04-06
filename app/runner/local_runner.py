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
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

from app.guards.repair_rules import repair_proposal
from app.runtime.watchdog import WatchdogSignal, inspect_log_chunk
from app.runner.base import ExecutionOutcome


class LocalExperimentRunner:
    """Launch a training command on the local machine."""

    def __init__(
        self,
        command_template: list[str],
        env: dict[str, str] | None = None,
        working_dir: Path | None = None,
        watchdog_poll_interval_sec: float = 0.5,
        watchdog_loss_threshold: float | None = None,
        watchdog_max_restarts: int = 2,
    ) -> None:
        self.command_template = command_template
        self.env = env or {}
        self.working_dir = working_dir or Path.cwd()
        self.watchdog_poll_interval_sec = watchdog_poll_interval_sec
        self.watchdog_loss_threshold = watchdog_loss_threshold
        self.watchdog_max_restarts = max(0, watchdog_max_restarts)

    def _render_command(self, config_path: Path, run_dir: Path) -> list[str]:
        rendered = [
            part.format(config_path=str(config_path), run_dir=str(run_dir))
            for part in self.command_template
        ]
        if rendered and rendered[0] in {"python", "python3"}:
            rendered[0] = sys.executable
        return rendered

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2))

    def _mirror_stream(self, stream: Any, path: Path, stop_event: threading.Event) -> None:
        """Copy a process stream into a log file as the child produces output."""

        with path.open("w", encoding="utf-8") as handle:
            for line in iter(stream.readline, ""):
                handle.write(line)
                handle.flush()
                if stop_event.is_set():
                    break

    def _run_attempt(
        self,
        *,
        command: list[str],
        run_dir: Path,
        timeout_sec: float,
        attempt_index: int,
        total_restarts: int,
    ) -> tuple[ExecutionOutcome, WatchdogSignal | None]:
        """Execute one attempt while a watchdog scans the live log files."""

        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        command_file = run_dir / "command.json"
        self._write_json(
            command_file,
            {
                "command": command,
                "attempt_index": attempt_index,
                "watchdog_max_restarts": total_restarts,
            },
        )
        started = time.time()
        process = subprocess.Popen(
            command,
            cwd=self.working_dir,
            env={**os.environ, **self.env},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        stop_event = threading.Event()
        stdout_thread = threading.Thread(
            target=self._mirror_stream,
            args=(process.stdout, stdout_path, stop_event),
            daemon=True,
            name=f"clae-stdout-{attempt_index}",
        )
        stderr_thread = threading.Thread(
            target=self._mirror_stream,
            args=(process.stderr, stderr_path, stop_event),
            daemon=True,
            name=f"clae-stderr-{attempt_index}",
        )
        stdout_thread.start()
        stderr_thread.start()

        watchdog_signal: WatchdogSignal | None = None
        timed_out = False
        try:
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break

                for source, path in (("stderr", stderr_path), ("stdout", stdout_path)):
                    if not path.exists():
                        continue
                    signal = inspect_log_chunk(
                        path.read_text(errors="ignore"),
                        source=source,
                        loss_threshold=self.watchdog_loss_threshold,
                    )
                    if signal is not None:
                        watchdog_signal = signal
                        break

                if watchdog_signal is not None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    break

                if time.time() - started > timeout_sec:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    break

                time.sleep(self.watchdog_poll_interval_sec)
        finally:
            stop_event.set()
            stdout_thread.join(timeout=10)
            stderr_thread.join(timeout=10)
            process.stdout.close()
            process.stderr.close()

        returncode = process.returncode
        extra: dict[str, Any] = {
            "attempt_index": attempt_index,
            "watchdog_max_restarts": total_restarts,
        }
        if watchdog_signal is not None:
            extra["watchdog_signal"] = {
                "source": watchdog_signal.source,
                "reason": watchdog_signal.reason,
                "line": watchdog_signal.line,
                "loss_name": watchdog_signal.loss_name,
                "loss_value": watchdog_signal.loss_value,
                "failure_category": watchdog_signal.failure_category,
            }
        return (
            ExecutionOutcome(
                run_dir=str(run_dir),
                config_path=str(run_dir / "config.yaml"),
                command=command,
                returncode=returncode,
                duration_sec=time.time() - started,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                timed_out=timed_out,
                extra=extra,
            ),
            watchdog_signal,
        )

    def run(self, config_path: Path, run_dir: Path, timeout_sec: float) -> ExecutionOutcome:
        """Execute the training command and capture stdout/stderr."""

        try:
            config_data = yaml.safe_load(config_path.read_text()) or {}
        except FileNotFoundError:
            config_data = {}

        attempt_root = run_dir
        attempt_count = 0
        watchdog_history: list[dict[str, Any]] = []
        current_config_data = config_data

        while True:
            if attempt_count == 0:
                attempt_dir = attempt_root
            else:
                attempt_dir = attempt_root / "watchdog" / f"attempt_{attempt_count:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_config_path = attempt_dir / "config.yaml"
            attempt_config_path.write_text(yaml.safe_dump(current_config_data, sort_keys=False))
            command = self._render_command(attempt_config_path, attempt_dir)

            outcome, signal = self._run_attempt(
                command=command,
                run_dir=attempt_dir,
                timeout_sec=timeout_sec,
                attempt_index=attempt_count,
                total_restarts=self.watchdog_max_restarts,
            )
            outcome.extra = {
                **(outcome.extra or {}),
                "watchdog_history": watchdog_history,
                "retry_count": attempt_count,
                "root_run_dir": str(attempt_root),
            }

            if signal is None or attempt_count >= self.watchdog_max_restarts:
                return outcome

            analysis = signal.to_analysis(attempt=attempt_count, run_dir=attempt_dir, config_path=attempt_config_path)
            repaired_config = repair_proposal(current_config_data, analysis=analysis)
            (attempt_dir / "watchdog.json").write_text(json.dumps(analysis, indent=2))
            watchdog_history.append(
                {
                    "attempt_index": attempt_count,
                    "analysis": analysis,
                    "repair_applied": repaired_config != current_config_data,
                    "source_run_dir": str(attempt_dir),
                }
            )
            current_config_data = repaired_config
            attempt_count += 1
