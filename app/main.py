# File: app/main.py
# Purpose: Provide the Typer CLI entrypoint for launching and managing closed-loop studies.
# License: GPL-3.0-or-later
"""CLI entrypoint for running closed-loop studies."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from app.evaluation.evaluator import EvaluationRunOptions, evaluate_suite, load_suite
from app.study.executor import StudyRunInputs, run_closed_loop_study


app = typer.Typer(add_completion=False, help="Closed-loop experiment controller.")


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

    result = run_closed_loop_study(
        StudyRunInputs(
            baseline_config_path=baseline_config,
            search_space_path=search_space,
            study_name=study_name,
            study_dir=study_dir,
            storage_path=storage_path,
            tracking_uri=tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            train_command=train_command,
            timeout_sec=timeout_sec,
        )
    )
    typer.echo(result.summary.model_dump_json(indent=2))


@app.command("evaluate-system")
def evaluate_system(
    suite_path: Annotated[Path, typer.Option(exists=True, readable=True, help="Benchmark suite YAML.")],
    output_dir: Annotated[Path, typer.Option(help="Directory for evaluation artifacts.")],
    version_tag: Annotated[str | None, typer.Option(help="Version tag for this run.")] = None,
    reference_report: Annotated[
        Path | None,
        typer.Option(exists=True, readable=True, help="Previous evaluation report."),
    ] = None,
    max_workers: Annotated[int, typer.Option(help="Parallel benchmark workers.")] = 2,
) -> None:
    """Run a benchmark suite and optionally compare it to an earlier report."""

    suite = load_suite(suite_path)
    report = evaluate_suite(
        EvaluationRunOptions(
            suite=suite,
            output_dir=output_dir,
            version_tag=version_tag,
            reference_report=reference_report,
            max_workers=max_workers,
        )
    )
    report_path = output_dir / f"{suite.suite_name}_evaluation.json"
    report_path.write_text(report.model_dump_json(indent=2))
    typer.echo(report.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
