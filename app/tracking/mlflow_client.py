# File: app/tracking/mlflow_client.py
# Purpose: Log the parent study run and per-trial child runs to MLflow.
# License: GPL-3.0-or-later
"""MLflow logging wrapper for parent/child run tracking."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment]

from app.tracking.schemas import StudySummary, TrialTrackingRecord


class MlflowTracker:
    """Minimal MLflow helper that keeps the study run as the parent run."""

    def __init__(self, tracking_uri: str, experiment_name: str, artifact_root: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_root = Path(artifact_root)
        self.parent_run_id: str | None = None
        self.enabled = mlflow is not None and MlflowClient is not None
        self.client = MlflowClient(tracking_uri=tracking_uri) if self.enabled else None
        self.experiment_id: str | None = None

        if self.enabled:
            assert mlflow is not None
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise RuntimeError(f"MLflow experiment not found: {experiment_name}")
            self.experiment_id = experiment.experiment_id

    @contextmanager
    def study_run(self, tags: dict[str, str]) -> Iterator[str]:
        """Open the parent run for the whole study session."""

        if not self.enabled or self.client is None or self.experiment_id is None:
            self.parent_run_id = f"local-{tags.get('study_name', 'study')}"
            yield self.parent_run_id
            return

        run = self.client.create_run(
            experiment_id=self.experiment_id,
            tags={**tags, "mlflow.runName": tags.get("study_name", "study")},
        )
        self.parent_run_id = run.info.run_id
        try:
            yield self.parent_run_id
        finally:
            self.client.set_terminated(self.parent_run_id, status="FINISHED")

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

        if not self.enabled or self.client is None or self.experiment_id is None:
            return
        self._require_parent()
        child = self.client.create_run(
            experiment_id=self.experiment_id,
            tags={
                "mlflow.parentRunId": self.parent_run_id,
                "mlflow.runName": f"trial-{record.trial_number}",
                "trial_number": str(record.trial_number),
                "status": record.status,
            },
        )
        run_id = child.info.run_id
        for key, value in record.params.items():
            self.client.log_param(run_id, key, json.dumps(value) if isinstance(value, (dict, list)) else value)
        for key, value in record.metrics.items():
            if isinstance(value, (int, float)):
                self.client.log_metric(run_id, key, float(value))
        self.client.log_artifact(run_id, str(config_path), artifact_path="trial")
        if results_path and results_path.exists():
            self.client.log_artifact(run_id, str(results_path), artifact_path="trial")
        if analysis_path and analysis_path.exists():
            self.client.log_artifact(run_id, str(analysis_path), artifact_path="trial")
        if extra_artifacts:
            for name, path in extra_artifacts.items():
                if path.exists():
                    self.client.log_artifact(run_id, str(path), artifact_path=name)
        self.client.set_terminated(run_id, status="FINISHED")

    def log_study_summary(self, summary: StudySummary, summary_path: Path) -> None:
        """Log the final study summary to the parent run."""

        if not self.enabled or self.client is None or self.experiment_id is None:
            return
        self._require_parent()
        self.client.log_artifact(self.parent_run_id, str(summary_path), artifact_path="study")
        self.client.set_tag(self.parent_run_id, "study_completed", "true")
