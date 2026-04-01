# File: app/graph/nodes.py
# Purpose: Implement the study-loop nodes that initialize, propose, run, analyze, and update trials.
# License: GPL-3.0-or-later
"""LangGraph node implementations for the MVP loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.analysis.heuristic_analyzer import analyze_result
from app.analysis.metric_parser import parse_results_json
from app.graph.state import LoopState, RunResult, TrialProposal, save_state
from app.guards.constraints import budget_exhausted, too_many_failures
from app.guards.validators import proposal_signature, validate_proposal
from app.optimizer.search_space import apply_overrides
from app.runner.base import ExecutionOutcome
from app.tracking.schemas import StudySummary, TrialTrackingRecord


def _state_path(state: LoopState) -> Path:
    return Path(state.study_dir) / "state.json"


def _trial_dir(state: LoopState, trial_number: int) -> Path:
    return Path(state.study_dir) / "trials" / f"trial_{trial_number:04d}"


def init_study(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Initialize the study state and filesystem scaffolding."""

    study_dir = Path(deps["study_dir"])
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "trials").mkdir(parents=True, exist_ok=True)

    loaded = deps["load_state"]()
    if loaded is not None:
        state_obj = loaded
    else:
        state_obj = LoopState(
            study_id=deps["study_id"],
            objective_name=deps["search_space"].objective.name,
            direction=deps["search_space"].objective.direction,
            budget_total_trials=deps["search_space"].constraints.max_trials,
            budget_gpu_hours=deps["search_space"].constraints.max_gpu_hours,
            baseline_config=deps["baseline_config"],
            search_space={name: spec.model_dump() for name, spec in deps["search_space"].space.items()},
            constraints=deps["search_space"].constraints.model_dump(),
            study_name=deps["study_name"],
            study_dir=str(study_dir),
            storage_url=deps["storage_url"],
            tracking_uri=deps["tracking_uri"],
            mlflow_experiment_name=deps["mlflow_experiment_name"],
            training_command=deps["training_command"],
            timeout_sec=deps["timeout_sec"],
        )
        save_state(state_obj, _state_path(state_obj))
    return state_obj.model_dump()


def load_history(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Load persisted history and synchronize summary fields."""

    state_obj = LoopState.model_validate(state)
    engine = deps["engine"]
    state_obj.history_summary = engine.history_summary()
    state_obj.best_trial = engine.best_trial()
    save_state(state_obj, _state_path(state_obj))
    return state_obj.model_dump()


def propose_trial(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Ask Optuna for the next proposal."""

    state_obj = LoopState.model_validate(state)
    asked = deps["engine"].ask()
    proposal = TrialProposal(
        trial_number=asked.trial_number,
        params=asked.params,
        signature=proposal_signature(asked.params),
    )
    state_obj.current_proposal = proposal.model_dump()
    state_obj.trial_number = asked.trial_number
    state_obj.decision = "continue"
    save_state(state_obj, _state_path(state_obj))
    return state_obj.model_dump()


def validate_proposal_node(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Validate a proposed trial against the declared whitelist."""

    state_obj = LoopState.model_validate(state)
    proposal = state_obj.current_proposal or {}
    params = proposal.get("params", {})
    validation = validate_proposal(
        params,
        deps["search_space"],
        recent_failed_signatures=state_obj.recent_failed_signatures,
    )
    if proposal:
        proposal["validated"] = validation.valid
        proposal["validation_errors"] = [] if validation.valid else [validation.reason or "invalid"]
        state_obj.current_proposal = proposal
    if not validation.valid:
        state_obj.current_result = RunResult(
            status="failed",
            primary_metric=None,
            metrics={},
            artifacts={},
            stderr_summary=validation.reason,
            duration_sec=0.0,
            run_dir=str(Path(state_obj.study_dir) / "invalid_proposal"),
        )
    save_state(state_obj, _state_path(state_obj))
    return state_obj.model_dump()


def launch_trial(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Generate the overridden config and run the local training command."""

    state_obj = LoopState.model_validate(state)
    if state_obj.current_result is not None and state_obj.current_result.status == "failed" and not state_obj.current_result.artifacts:
        return state_obj.model_dump()

    assert state_obj.current_proposal is not None
    trial_number = state_obj.trial_number or 0
    trial_dir = _trial_dir(state_obj, trial_number)
    trial_dir.mkdir(parents=True, exist_ok=True)
    overrides = state_obj.current_proposal["params"]
    trial_config = apply_overrides(state_obj.baseline_config, overrides)
    config_path = trial_dir / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(trial_config, sort_keys=False))
    outcome: ExecutionOutcome = deps["runner"].run(config_path=config_path, run_dir=trial_dir, timeout_sec=state_obj.timeout_sec)
    state_obj.run_dir = str(trial_dir)
    state_obj.history_summary = {**state_obj.history_summary, "last_command": outcome.command}
    state_obj.current_execution = {
        "run_dir": outcome.run_dir,
        "config_path": outcome.config_path,
        "command": outcome.command,
        "returncode": outcome.returncode,
        "duration_sec": outcome.duration_sec,
        "stdout_path": outcome.stdout_path,
        "stderr_path": outcome.stderr_path,
        "timed_out": outcome.timed_out,
        "extra": outcome.extra or {},
    }
    state_obj.current_analysis = {
        "launch": {
            "command": outcome.command,
            "returncode": outcome.returncode,
            "timed_out": outcome.timed_out,
            "stdout_path": outcome.stdout_path,
            "stderr_path": outcome.stderr_path,
        }
    }
    save_state(state_obj, _state_path(state_obj))
    return state_obj.model_dump()


def collect_metrics(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Parse results.json into a standardized RunResult."""

    state_obj = LoopState.model_validate(state)
    execution = state_obj.current_execution or {}
    if state_obj.run_dir is None:
        raise RuntimeError("run_dir is not set")
    run_dir = Path(state_obj.run_dir)
    stderr_path = Path(execution.get("stderr_path", run_dir / "stderr.txt"))
    stderr_summary = None
    if stderr_path.exists():
        stderr_summary = stderr_path.read_text()[:2000] or None
    result = parse_results_json(run_dir, duration_sec=float(execution.get("duration_sec", 0.0)), stderr_summary=stderr_summary)
    if execution.get("timed_out"):
        result.status = "timeout"
        if result.stderr_summary is None:
            result.stderr_summary = "timeout"
    state_obj.current_result = result
    save_state(state_obj, _state_path(state_obj))
    return state_obj.model_dump()


def analyze_result_node(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Compute deterministic analysis and persist it."""

    state_obj = LoopState.model_validate(state)
    if state_obj.current_result is None:
        raise RuntimeError("Missing current_result before analysis")
    analysis = analyze_result(state_obj, state_obj.current_result)
    state_obj.current_analysis = analysis.payload
    if state_obj.current_result.status != "success":
        state_obj.failure_count += 1
        signature = state_obj.current_proposal.get("signature") if state_obj.current_proposal else None
        if signature:
            state_obj.recent_failed_signatures.append(signature)
            state_obj.recent_failed_signatures = state_obj.recent_failed_signatures[-8:]
    else:
        state_obj.failure_count = 0
        if analysis.improved:
            state_obj.plateau_count = 0
        else:
            state_obj.plateau_count += 1
    analysis_path = Path(state_obj.run_dir or state_obj.study_dir) / "analysis.json"
    analysis_path.write_text(json.dumps(state_obj.current_analysis, indent=2))
    save_state(state_obj, _state_path(state_obj))
    return {**state_obj.model_dump(), "_analysis_path": str(analysis_path)}


def update_study(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Tell Optuna the outcome and update summary metrics."""

    state_obj = LoopState.model_validate(state)
    if state_obj.trial_number is None or state_obj.current_proposal is None or state_obj.current_result is None:
        raise RuntimeError("Missing trial context before study update")

    result = state_obj.current_result
    if result.status == "success" and result.primary_metric is not None:
        deps["engine"].tell_success(state_obj.trial_number, float(result.primary_metric))
    else:
        deps["engine"].tell_failure(state_obj.trial_number)

    state_obj.budget_used_trials += 1
    state_obj.used_gpu_hours += max(result.duration_sec / 3600.0, 0.0)
    state_obj.best_trial = deps["engine"].best_trial()
    state_obj.history_summary = deps["engine"].history_summary()
    state_obj.history_summary["current_status"] = result.status
    state_obj.history_summary["failure_count"] = state_obj.failure_count
    state_obj.history_summary["plateau_count"] = state_obj.plateau_count
    trial_dir = Path(state_obj.run_dir or state_obj.study_dir)
    record = TrialTrackingRecord(
        trial_number=state_obj.trial_number,
        params=state_obj.current_proposal.get("params", {}),
        metrics=result.metrics,
        status=result.status,
        run_dir=str(trial_dir),
        analysis=state_obj.current_analysis or {},
    )
    deps["tracker"].log_trial(
        record=record,
        config_path=trial_dir / "config.yaml",
        results_path=trial_dir / "results.json",
        analysis_path=trial_dir / "analysis.json",
    )
    summary = StudySummary(
        study_id=state_obj.study_id,
        objective_name=state_obj.objective_name,
        direction=state_obj.direction,
        total_trials=state_obj.budget_used_trials,
        completed_trials=state_obj.history_summary.get("completed_trials", 0),
        failed_trials=state_obj.history_summary.get("failed_trials", 0),
        best_trial=state_obj.best_trial,
        notes="intermediate",
    )
    summary_path = Path(state_obj.study_dir) / "study_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2))
    state_obj.history_summary["summary_path"] = str(summary_path)
    save_state(state_obj, _state_path(state_obj))
    return state_obj.model_dump()


def decide_next_action(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Compute the next loop decision from the current state."""

    state_obj = LoopState.model_validate(state)
    if budget_exhausted(state_obj):
        state_obj.decision = "stop"
    elif too_many_failures(state_obj):
        state_obj.decision = "human_review"
    elif state_obj.current_analysis and state_obj.current_analysis.get("plateau_signal"):
        state_obj.decision = "human_review"
    else:
        state_obj.decision = "continue"
    save_state(state_obj, _state_path(state_obj))
    return state_obj.model_dump()


def human_review(state: dict[str, Any], *, deps: dict[str, Any]) -> dict[str, Any]:
    """Terminal node for manual inspection requests."""

    state_obj = LoopState.model_validate(state)
    state_obj.decision = "human_review"
    save_state(state_obj, _state_path(state_obj))
    return state_obj.model_dump()
