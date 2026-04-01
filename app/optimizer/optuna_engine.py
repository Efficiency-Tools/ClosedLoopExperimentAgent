# File: app/optimizer/optuna_engine.py
# Purpose: Wrap Optuna ask-and-tell so the controller can suggest and record trials.
# License: GPL-3.0-or-later
"""Optuna ask-and-tell wrapper for the study loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
from optuna.trial import TrialState

from app.optimizer.search_space import SearchSpaceConfig


@dataclass
class OptunaAskTellResult:
    trial_number: int
    params: dict[str, Any]


class OptunaEngine:
    """Small ask-and-tell façade around Optuna."""

    def __init__(
        self,
        study_name: str,
        storage_url: str,
        search_space: SearchSpaceConfig,
        load_if_exists: bool = True,
    ) -> None:
        self.study_name = study_name
        self.storage_url = storage_url
        self.search_space = search_space
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction=search_space.objective.direction,
            load_if_exists=load_if_exists,
        )

    def ask(self) -> OptunaAskTellResult:
        """Request the next trial from Optuna and sample the declared space."""

        trial = self.study.ask()
        params: dict[str, Any] = {}
        for name, spec in self.search_space.space.items():
            if spec.type == "float":
                assert spec.low is not None and spec.high is not None
                params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
            elif spec.type == "int":
                assert spec.low is not None and spec.high is not None
                params[name] = trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
            elif spec.type == "categorical":
                assert spec.choices is not None
                params[name] = trial.suggest_categorical(name, spec.choices)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported parameter type: {spec.type}")
        return OptunaAskTellResult(trial_number=trial.number, params=params)

    def tell_success(self, trial_number: int, metric: float) -> None:
        """Record a successful trial outcome."""

        self.study.tell(trial_number, metric)

    def tell_failure(self, trial_number: int) -> None:
        """Record a failed or invalid trial outcome."""

        self.study.tell(trial_number, state=TrialState.FAIL)

    def best_trial(self) -> dict[str, Any] | None:
        """Return a JSON-serializable snapshot of the best trial if available."""

        if len(self.study.trials) == 0:
            return None
        try:
            best = self.study.best_trial
        except ValueError:
            return None
        return {
            "number": best.number,
            "value": best.value,
            "params": dict(best.params),
            "state": best.state.name,
        }

    def history_summary(self) -> dict[str, Any]:
        """Summarize the current Optuna study."""

        trials = self.study.trials
        completed = [t for t in trials if t.state == TrialState.COMPLETE]
        failed = [t for t in trials if t.state == TrialState.FAIL]
        running = [t for t in trials if t.state == TrialState.RUNNING]
        return {
            "total_trials": len(trials),
            "completed_trials": len(completed),
            "failed_trials": len(failed),
            "running_trials": len(running),
            "best_value": self.study.best_value if len(completed) else None,
        }
