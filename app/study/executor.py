# File: app/study/executor.py
# Purpose: Run a closed-loop study from reusable inputs.
# License: GPL-3.0-or-later
"""Reusable closed-loop study execution."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from app.graph.build_graph import build_graph
from app.graph.state import LoopState, load_state, save_state
from app.optimizer.optuna_engine import OptunaEngine
from app.optimizer.search_space import load_search_space
from app.runtime.background import BackgroundTaskManager
from app.runner.local_runner import LocalExperimentRunner
from app.tracking.mlflow_client import MlflowTracker
from app.tracking.schemas import StudySummary


@dataclass
class StudyRunInputs:
    baseline_config_path: Path
    search_space_path: Path
    study_name: str
    study_dir: Path
    storage_path: Path
    tracking_uri: str
    mlflow_experiment_name: str
    train_command: str
    timeout_sec: float = 3600.0


@dataclass
class StudyRunOutputs:
    final_state: LoopState
    summary: StudySummary
    summary_path: Path


def run_closed_loop_study(inputs: StudyRunInputs, *, background_workers: int = 4) -> StudyRunOutputs:
    """Execute one study end to end."""

    inputs.study_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    baseline_data = yaml.safe_load(inputs.baseline_config_path.read_text())
    search_space_data = load_search_space(inputs.search_space_path)
    storage_url = f"sqlite:///{inputs.storage_path}"
    engine = OptunaEngine(
        study_name=inputs.study_name,
        storage_url=storage_url,
        search_space=search_space_data,
    )
    runner = LocalExperimentRunner(
        shlex.split(inputs.train_command),
        watchdog_loss_threshold=search_space_data.constraints.max_loss_value,
    )
    background = BackgroundTaskManager(max_workers=background_workers)
    tracker = MlflowTracker(
        tracking_uri=inputs.tracking_uri,
        experiment_name=inputs.mlflow_experiment_name,
        artifact_root=str(inputs.study_dir / "artifacts"),
    )

    state_path = inputs.study_dir / "state.json"

    def _load_state():
        existing = load_state(state_path)
        if existing is not None:
            return existing
        return None

    runtime = {
        "study_id": inputs.study_name,
        "study_name": inputs.study_name,
        "study_dir": str(inputs.study_dir),
        "baseline_config": baseline_data,
        "search_space": search_space_data,
        "storage_url": storage_url,
        "tracking_uri": inputs.tracking_uri,
        "mlflow_experiment_name": inputs.mlflow_experiment_name,
        "training_command": shlex.split(inputs.train_command),
        "timeout_sec": inputs.timeout_sec,
        "engine": engine,
        "runner": runner,
        "tracker": tracker,
        "background": background,
        "load_state": _load_state,
    }
    graph = build_graph(runtime)
    initial_state = LoopState(
        study_id=inputs.study_name,
        objective_name=search_space_data.objective.name,
        direction=search_space_data.objective.direction,
        budget_total_trials=search_space_data.constraints.max_trials,
        budget_gpu_hours=search_space_data.constraints.max_gpu_hours,
        baseline_config=baseline_data,
        search_space={name: spec.model_dump() for name, spec in search_space_data.space.items()},
        constraints=search_space_data.constraints.model_dump(),
        study_name=inputs.study_name,
        study_dir=str(inputs.study_dir),
        storage_url=storage_url,
        tracking_uri=inputs.tracking_uri,
        mlflow_experiment_name=inputs.mlflow_experiment_name,
        training_command=shlex.split(inputs.train_command),
        timeout_sec=inputs.timeout_sec,
    )
    save_state(initial_state, state_path)

    try:
        with tracker.study_run(
            tags={
                "study_name": inputs.study_name,
                "objective_name": search_space_data.objective.name,
                "direction": search_space_data.objective.direction,
            }
        ):
            final_state = graph.invoke(initial_state.model_dump())
            final_state_obj = LoopState.model_validate(final_state)
            summary = StudySummary(
                study_id=final_state_obj.study_id,
                objective_name=final_state_obj.objective_name,
                direction=final_state_obj.direction,
                total_trials=final_state_obj.budget_used_trials,
                completed_trials=final_state_obj.history_summary.get("completed_trials", 0),
                failed_trials=final_state_obj.history_summary.get("failed_trials", 0),
                best_trial=final_state_obj.best_trial,
                notes=f"decision={final_state_obj.decision}",
            )
            summary_path = inputs.study_dir / "final_summary.json"
            summary_path.write_text(summary.model_dump_json(indent=2))
            background.submit(tracker.log_study_summary, summary, summary_path)
            background.drain()
            return StudyRunOutputs(
                final_state=final_state_obj,
                summary=summary,
                summary_path=summary_path,
            )
    finally:
        background.close()
