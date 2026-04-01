# File: app/tracking/mlflow_client.py
# Purpose: Log the parent study run and per-trial child runs to MLflow.
# License: GPL-3.0-or-later
"""MLflow logging wrapper for parent/child run tracking."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

from app.tracking.schemas import StudySummary, TrialTrackingRecord


class MlflowTracker:
    """Minimal MLflow helper that keeps the study run as the parent run."""

    def __init__(self, tracking_uri: str, experiment_name: str, artifact_root: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_root = Path(artifact_root)
        self.parent_run_id: str | None = None

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def study_run(self, tags: dict[str, str]) -> Iterator[str]:
        """Open the parent run for the whole study session."""

        with mlflow.start_run(run_name=tags.get("study_name", "study"), nested=False) as run:
            self.parent_run_id = run.info.run_id
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            yield self.parent_run_id

    def _require_parent(self) -> None:
        if self.parent_run_id is None:
            raise RuntimeError("Study run is not active.")

    def log_trial(
        self,
        record: TrialTrackingRecord,
        config_path: Path,
        results_path: Path | None,
        analysis_path: Path | None,
        extra_artifacts: dict[str, Path] | None = None,
    ) -> None:
        """Log a child run for one trial."""

        self._require_parent()
        with mlflow.start_run(
            run_name=f"trial-{record.trial_number}",
            nested=True,
            tags={"trial_number": str(record.trial_number)},
        ):
            mlflow.log_params({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in record.params.items()})
            mlflow.log_metrics(
                {k: float(v) for k, v in record.metrics.items() if isinstance(v, (int, float))}
            )
            mlflow.set_tag("status", record.status)
            mlflow.log_artifact(str(config_path), artifact_path="trial")
            if results_path and results_path.exists():
                mlflow.log_artifact(str(results_path), artifact_path="trial")
            if analysis_path and analysis_path.exists():
                mlflow.log_artifact(str(analysis_path), artifact_path="trial")
            if extra_artifacts:
                for name, path in extra_artifacts.items():
                    if path.exists():
                        mlflow.log_artifact(str(path), artifact_path=name)

    def log_study_summary(self, summary: StudySummary, summary_path: Path) -> None:
        """Log the final study summary to the parent run."""

        self._require_parent()
        mlflow.log_artifact(str(summary_path), artifact_path="study")
        mlflow.set_tag("study_completed", "true")
        mlflow.log_dict(summary.model_dump(), "study_summary.json")
