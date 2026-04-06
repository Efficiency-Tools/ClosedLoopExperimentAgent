"""Tests for conservative automatic repair rules."""

from __future__ import annotations

from app.guards.repair_rules import repair_proposal


def test_repair_proposal_reduces_learning_rate_after_nan_signal():
    proposal = {"train": {"lr": 1e-3}, "optimizer": {"learning_rate": 2e-4}}
    repaired = repair_proposal(proposal, analysis={"failure_category": "FAIL_NAN", "reason": "loss exceeds threshold"})
    assert repaired["train"]["lr"] == 5e-4
    assert repaired["optimizer"]["learning_rate"] == 1e-4


def test_repair_proposal_reduces_batch_size_after_oom_signal():
    proposal = {"train": {"batch_size": 16}, "model": {"micro_batch_size": 8}}
    repaired = repair_proposal(proposal, analysis={"failure_category": "FAIL_OOM"})
    assert repaired["train"]["batch_size"] == 8
    assert repaired["model"]["micro_batch_size"] == 4
