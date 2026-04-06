# File: app/runtime/watchdog.py
# Purpose: Detect unhealthy training logs and describe recovery actions.
# License: GPL-3.0-or-later
"""Loss and error watchdog helpers for training logs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_LOSS_LINE_RE = re.compile(
    r"""
    (?P<name>[A-Za-z0-9_.-]*loss[A-Za-z0-9_.-]*)
    \s*(?:=|:)\s*
    (?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_JSON_LOSS_LINE_RE = re.compile(
    r"""
    ["'](?P<name>[A-Za-z0-9_.-]*loss[A-Za-z0-9_.-]*)["']
    \s*:\s*
    (?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_ERROR_PATTERNS = (
    "traceback",
    "exception",
    "runtimeerror",
    "cuda out of memory",
    "out of memory",
    "oom",
    "fatal",
    "segmentation fault",
)
_LOSS_THRESHOLD_PATTERNS = (
    "loss exceeds threshold",
    "loss too high",
    "loss diverged",
    "diverged loss",
    "exploding loss",
)


@dataclass
class WatchdogSignal:
    """Structured watchdog trigger emitted from log inspection."""

    source: str
    reason: str
    line: str
    loss_name: str | None = None
    loss_value: float | None = None

    @property
    def failure_category(self) -> str:
        """Map the signal into the shared failure taxonomy."""

        lowered = self.reason.lower()
        if self.loss_value is not None and not math.isfinite(self.loss_value):
            return "FAIL_NAN"
        if "oom" in lowered or "out of memory" in lowered:
            return "FAIL_OOM"
        if "traceback" in lowered or "exception" in lowered or "runtime error" in lowered:
            return "FAIL_RUNTIME"
        if "timeout" in lowered:
            return "FAIL_TIMEOUT"
        if "config" in lowered or "invalid" in lowered:
            return "FAIL_CONFIG"
        if "loss" in lowered:
            return "FAIL_RUNTIME"
        return "FAIL_UNKNOWN"

    def to_analysis(self, *, attempt: int, run_dir: Path, config_path: Path) -> dict[str, Any]:
        """Translate the signal into a compact analysis payload."""

        return {
            "analysis_type": "watchdog",
            "attempt": attempt,
            "source": self.source,
            "reason": self.reason,
            "line": self.line,
            "loss_name": self.loss_name,
            "loss_value": self.loss_value,
            "failure_category": self.failure_category,
            "stop_now": True,
            "run_dir": str(run_dir),
            "config_path": str(config_path),
        }


def _parse_loss_line(line: str) -> tuple[str | None, float | None]:
    for pattern in (_LOSS_LINE_RE, _JSON_LOSS_LINE_RE):
        match = pattern.search(line)
        if not match:
            continue
        loss_name = match.group("name")
        raw_value = match.group("value")
        try:
            return loss_name, float(raw_value)
        except ValueError:
            return loss_name, None
    return None, None


def inspect_log_chunk(chunk: str, *, source: str, loss_threshold: float | None) -> WatchdogSignal | None:
    """Inspect a newly appended log chunk for loss spikes or runtime errors."""

    if not chunk:
        return None

    for raw_line in chunk.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        loss_name, loss_value = _parse_loss_line(line)
        if loss_value is not None:
            if not math.isfinite(loss_value):
                return WatchdogSignal(
                    source=source,
                    reason=f"non-finite loss detected in {loss_name}",
                    line=line,
                    loss_name=loss_name,
                    loss_value=loss_value,
                )
            if loss_threshold is not None and loss_value > loss_threshold:
                return WatchdogSignal(
                    source=source,
                    reason=f"loss exceeds threshold: {loss_value} > {loss_threshold}",
                    line=line,
                    loss_name=loss_name,
                    loss_value=loss_value,
                )
            continue

        lowered = line.lower()
        if any(pattern in lowered for pattern in _LOSS_THRESHOLD_PATTERNS):
            return WatchdogSignal(
                source=source,
                reason="loss divergence reported in logs",
                line=line,
            )
        if any(pattern in lowered for pattern in _ERROR_PATTERNS):
            return WatchdogSignal(
                source=source,
                reason=f"runtime error detected: {line[:200]}",
                line=line,
            )

    return None


def scan_log_files(stdout_path: Path, stderr_path: Path, *, loss_threshold: float | None) -> WatchdogSignal | None:
    """Inspect the current stdout/stderr contents for watchdog triggers."""

    for source, path in (("stderr", stderr_path), ("stdout", stdout_path)):
        if not path.exists():
            continue
        signal = inspect_log_chunk(path.read_text(errors="ignore"), source=source, loss_threshold=loss_threshold)
        if signal is not None:
            return signal
    return None
