# File: app/evaluation/evaluator.py
# Purpose: Run benchmark suites and compare results across versions.
# License: GPL-3.0-or-later
"""System evaluation and version comparison."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.schemas import (
    EvaluationCaseResult,
    EvaluationDelta,
    EvaluationReport,
    EvaluationSuiteSpec,
)


@dataclass
class EvaluationRunOptions:
    suite: EvaluationSuiteSpec
    output_dir: Path
    version_tag: str | None = None
    reference_report: Path | None = None
    max_workers: int = 2


def load_suite(path: str | Path) -> EvaluationSuiteSpec:
    """Load a benchmark suite from YAML."""

    import yaml

    data = yaml.safe_load(Path(path).read_text())
    return EvaluationSuiteSpec.model_validate(data)


def _detect_version_tag() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        tag = result.stdout.strip()
        if tag:
            return tag
    except OSError:
        pass
    return "local"


def _case_paths(output_dir: Path, case_name: str) -> tuple[Path, Path]:
    case_root = output_dir / case_name
    study_dir = case_root / "study"
    storage_path = case_root / "optuna.sqlite3"
    return study_dir, storage_path


def _run_case(case, output_dir: Path, version_tag: str) -> EvaluationCaseResult:
    from app.study.executor import StudyRunInputs, run_closed_loop_study

    study_dir, storage_path = _case_paths(output_dir, case.name)
    inputs = StudyRunInputs(
        baseline_config_path=case.baseline_config,
        search_space_path=case.search_space,
        study_name=case.study_name or f"{case.name}-{version_tag}",
        study_dir=case.study_dir or study_dir,
        storage_path=case.storage_path or storage_path,
        tracking_uri=case.tracking_uri,
        mlflow_experiment_name=case.mlflow_experiment_name,
        train_command=case.train_command,
        timeout_sec=case.timeout_sec,
    )
    started = datetime.now(timezone.utc)
    outputs = run_closed_loop_study(inputs)
    duration_sec = (datetime.now(timezone.utc) - started).total_seconds()
    best_value = None
    if outputs.summary.best_trial and outputs.summary.best_trial.get("value") is not None:
        try:
            best_value = float(outputs.summary.best_trial["value"])
        except (TypeError, ValueError):
            best_value = None
    return EvaluationCaseResult(
        name=case.name,
        study_name=inputs.study_name,
        study_dir=str(inputs.study_dir),
        objective_name=outputs.final_state.objective_name,
        direction=outputs.final_state.direction,
        total_trials=outputs.final_state.budget_used_trials,
        completed_trials=outputs.final_state.history_summary.get("completed_trials", 0),
        failed_trials=outputs.final_state.history_summary.get("failed_trials", 0),
        best_value=best_value,
        best_trial=outputs.summary.best_trial,
        duration_sec=duration_sec,
        notes=outputs.summary.notes,
    )


def _load_reference(path: Path) -> EvaluationReport:
    return EvaluationReport.model_validate_json(path.read_text())


def _delta_for_case(current: EvaluationCaseResult, reference: EvaluationCaseResult) -> EvaluationDelta:
    def _maybe_delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return round(a - b, 6)

    return EvaluationDelta(
        name=current.name,
        best_value_delta=_maybe_delta(current.best_value, reference.best_value),
        completed_trials_delta=current.completed_trials - reference.completed_trials,
        failed_trials_delta=current.failed_trials - reference.failed_trials,
        duration_sec_delta=current.duration_sec - reference.duration_sec,
    )


def evaluate_suite(options: EvaluationRunOptions) -> EvaluationReport:
    """Run a suite of benchmark cases, optionally comparing to a reference report."""

    options.output_dir.mkdir(parents=True, exist_ok=True)
    version_tag = options.version_tag or _detect_version_tag()
    started_at = datetime.now(timezone.utc)

    cases = list(options.suite.cases)
    if not cases:
        raise ValueError("evaluation suite has no cases")

    with ThreadPoolExecutor(max_workers=max(1, options.max_workers)) as pool:
        futures = [pool.submit(_run_case, case, options.output_dir, version_tag) for case in cases]
        case_results = [future.result() for future in futures]

    report = EvaluationReport(
        suite_name=options.suite.suite_name,
        version_tag=version_tag,
        created_at=started_at.isoformat(),
        cases=case_results,
    )

    if options.reference_report is not None:
        reference = _load_reference(options.reference_report)
        report.comparison_to = str(options.reference_report)
        ref_by_name = {case.name: case for case in reference.cases}
        report.deltas = [
            _delta_for_case(case, ref_by_name[case.name])
            for case in case_results
            if case.name in ref_by_name
        ]

    report.summary = {
        "case_count": len(report.cases),
        "mean_best_value": _mean([case.best_value for case in report.cases]),
        "mean_completed_trials": _mean([float(case.completed_trials) for case in report.cases]),
        "mean_failed_trials": _mean([float(case.failed_trials) for case in report.cases]),
        "mean_duration_sec": _mean([case.duration_sec for case in report.cases]),
    }
    if report.deltas:
        report.summary["mean_best_value_delta"] = _mean([delta.best_value_delta for delta in report.deltas])
        report.summary["mean_failed_trials_delta"] = _mean([float(delta.failed_trials_delta) for delta in report.deltas])
        report.summary["mean_duration_sec_delta"] = _mean([delta.duration_sec_delta for delta in report.deltas])
    return report


def _mean(values: list[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return round(float(sum(cleaned) / len(cleaned)), 6)
