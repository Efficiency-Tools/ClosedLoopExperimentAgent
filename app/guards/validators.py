# File: app/guards/validators.py
# Purpose: Validate trial proposals against the declared whitelist and protected keys.
# License: GPL-3.0-or-later
"""Proposal validation and whitelist enforcement."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.graph.state import ValidationResult
from app.optimizer.search_space import SearchSpaceConfig


def proposal_signature(params: dict[str, Any]) -> str:
    """Stable signature for a parameter proposal."""

    normalized = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_proposal(
    proposal: dict[str, Any],
    search_space: SearchSpaceConfig,
    recent_failed_signatures: list[str] | None = None,
) -> ValidationResult:
    """Enforce whitelist-only edits and type/range constraints."""

    recent_failed_signatures = recent_failed_signatures or []
    allowed_keys = set(search_space.space.keys())
    proposal_keys = set(proposal.keys())
    blocked = [key for key in proposal_keys if key not in allowed_keys]
    protected_overlap = [key for key in proposal_keys if key in set(search_space.constraints.protected_keys)]
    if blocked:
        return ValidationResult(valid=False, reason=f"Unknown proposal keys: {sorted(blocked)}", blocked_keys=blocked)
    if protected_overlap:
        return ValidationResult(
            valid=False,
            reason=f"Protected keys may not be changed: {sorted(protected_overlap)}",
            blocked_keys=protected_overlap,
        )

    for name, value in proposal.items():
        spec = search_space.space[name]
        if spec.type == "float":
            if not isinstance(value, (int, float)):
                return ValidationResult(valid=False, reason=f"{name} must be numeric")
            if spec.low is not None and float(value) < float(spec.low):
                return ValidationResult(valid=False, reason=f"{name} below lower bound")
            if spec.high is not None and float(value) > float(spec.high):
                return ValidationResult(valid=False, reason=f"{name} above upper bound")
        elif spec.type == "int":
            if not isinstance(value, int):
                return ValidationResult(valid=False, reason=f"{name} must be int")
            if spec.low is not None and value < int(spec.low):
                return ValidationResult(valid=False, reason=f"{name} below lower bound")
            if spec.high is not None and value > int(spec.high):
                return ValidationResult(valid=False, reason=f"{name} above upper bound")
        elif spec.type == "categorical":
            if spec.choices is None or value not in spec.choices:
                return ValidationResult(valid=False, reason=f"{name} not in categorical choices")

    signature = proposal_signature(proposal)
    if signature in set(recent_failed_signatures):
        return ValidationResult(
            valid=False,
            reason="Repeated unsafe proposal after a recent failure",
            blocked_keys=list(proposal.keys()),
        )
    return ValidationResult(valid=True)
