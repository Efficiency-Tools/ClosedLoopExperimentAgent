# File: app/optimizer/search_space.py
# Purpose: Parse the search-space contract and apply whitelisted overrides to configs.
# License: GPL-3.0-or-later
"""Search-space parsing and proposal application helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field


class ObjectiveSpec(BaseModel):
    name: str
    direction: Literal["maximize", "minimize"]


class ParameterSpec(BaseModel):
    type: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Any] | None = None


class ConstraintsSpec(BaseModel):
    max_trials: int
    max_gpu_hours: float = 0.0
    max_consecutive_failures: int = 3
    protected_keys: list[str] = Field(default_factory=list)


class SearchSpaceConfig(BaseModel):
    objective: ObjectiveSpec
    space: dict[str, ParameterSpec]
    constraints: ConstraintsSpec


def load_search_space(path: str | Path) -> SearchSpaceConfig:
    """Read and validate the YAML search-space contract."""

    data = yaml.safe_load(Path(path).read_text())
    return SearchSpaceConfig.model_validate(data)


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict using dotted keys."""

    flat: dict[str, Any] = {}
    for key, value in payload.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, dotted))
        else:
            flat[dotted] = value
    return flat


def unflatten_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Expand dotted keys into nested dictionaries."""

    root: dict[str, Any] = {}
    for key, value in payload.items():
        cursor = root
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def apply_overrides(baseline_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge dotted-key overrides into a baseline config."""

    merged = OmegaConf.merge(OmegaConf.create(copy.deepcopy(baseline_config)), OmegaConf.create(unflatten_dict(overrides)))
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]
