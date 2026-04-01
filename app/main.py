# File: app/main.py
# Purpose: Provide the Typer CLI entrypoint for launching and managing closed-loop studies.
# License: GPL-3.0-or-later
"""CLI entrypoint for running closed-loop studies."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Annotated

import typer
import yaml

from app.graph.build_graph import build_graph
from app.graph.state import LoopState, load_state, save_state
from app.optimizer.optuna_engine import OptunaEngine
from app.optimizer.search_space import load_search_space
from app.runner.local_runner import LocalExperimentRunner
from app.tracking.mlflow_client import MlflowTracker
from app.tracking.schemas import StudySummary


app = typer.Typer(add_completion=False, help="Closed-loop experiment controller.")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@app.callback()
def main() -> None:
    """Closed-loop experiment controller CLI."""


@app.command("launch-study")
def launch_study(
    baseline_config: Annotated[Path, typer.Option(exists=True, readable=True, help="Baseline config YAML.")],
    search_space: Annotated[Path, typer.Option(exists=True, readable=True, help="Search-space YAML.")],
    study_name: Annotated[str, typer.Option(help="Stable Optuna study name.")],
    study_dir: Annotated[Path, typer.Option(help="Directory for study artifacts.")],
    storage_path: Annotated[Path, typer.Option(help="SQLite path for Optuna storage.")],
    tracking_uri: Annotated[str, typer.Option(help="MLflow tracking URI.")],
    mlflow_experiment_name: Annotated[str, typer.Option(help="MLflow experiment name.")],
    train_command: Annotated[str, typer.Option(help="Training command with {config_path} and {run_dir} placeholders.")],
    timeout_sec: Annotated[float, typer.Option(help="Per-trial timeout in seconds.")] = 3600.0,
) -> None:
    """Run the study loop until the stopping condition is met."""

    study_dir.mkdir(parents=True, exist_ok=True)
    baseline_data = _load_yaml(baseline_config)
    search_space_data = load_search_space(search_space)
    storage_url = f"sqlite:///{storage_path}"
    engine = OptunaEngine(
        study_name=study_name,
        storage_url=storage_url,
        search_space=search_space_data,
    )
    runner = LocalExperimentRunner(shlex.split(train_command))
    tracker = MlflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=mlflow_experiment_name,
        artifact_root=str(study_dir / "artifacts"),
    )

    state_path = study_dir / "state.json"

    def _load_state():
        existing = load_state(state_path)
        if existing is not None:
            return existing
        return None

    runtime = {
        "study_id": study_name,
        "study_name": study_name,
        "study_dir": str(study_dir),
        "baseline_config": baseline_data,
        "search_space": search_space_data,
        "storage_url": storage_url,
        "tracking_uri": tracking_uri,
        "mlflow_experiment_name": mlflow_experiment_name,
        "training_command": shlex.split(train_command),
        "timeout_sec": timeout_sec,
        "engine": engine,
        "runner": runner,
        "tracker": tracker,
        "load_state": _load_state,
    }
    graph = build_graph(runtime)
    initial_state = LoopState(
        study_id=study_name,
        objective_name=search_space_data.objective.name,
        direction=search_space_data.objective.direction,
        budget_total_trials=search_space_data.constraints.max_trials,
        budget_gpu_hours=search_space_data.constraints.max_gpu_hours,
        baseline_config=baseline_data,
        search_space={name: spec.model_dump() for name, spec in search_space_data.space.items()},
        constraints=search_space_data.constraints.model_dump(),
        study_name=study_name,
        study_dir=str(study_dir),
        storage_url=storage_url,
        tracking_uri=tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        training_command=shlex.split(train_command),
        timeout_sec=timeout_sec,
    )
    save_state(initial_state, state_path)

    with tracker.study_run(
        tags={
            "study_name": study_name,
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
        summary_path = study_dir / "final_summary.json"
        summary_path.write_text(summary.model_dump_json(indent=2))
        tracker.log_study_summary(summary, summary_path)
        typer.echo(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
