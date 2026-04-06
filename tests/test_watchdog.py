"""Tests for log watchdog detection."""

from __future__ import annotations

from app.runtime.watchdog import inspect_log_chunk


def test_watchdog_detects_loss_spike():
    signal = inspect_log_chunk("step=12 train_loss=42.0", source="stdout", loss_threshold=20.0)
    assert signal is not None
    assert signal.loss_name == "train_loss"
    assert signal.loss_value == 42.0
    assert "threshold" in signal.reason


def test_watchdog_detects_runtime_error():
    signal = inspect_log_chunk("Traceback (most recent call last):", source="stderr", loss_threshold=20.0)
    assert signal is not None
    assert signal.source == "stderr"
    assert signal.failure_category == "FAIL_RUNTIME"
