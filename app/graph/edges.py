# File: app/graph/edges.py
# Purpose: Route the study graph between proposal, execution, review, and stop states.
# License: GPL-3.0-or-later
"""Routing helpers for the LangGraph controller."""

from __future__ import annotations

from typing import Any, Literal


def after_validate(state: dict[str, Any]) -> Literal["LaunchTrial", "AnalyzeResult"]:
    """Route invalid proposals directly to analysis."""

    proposal = state.get("current_proposal") or {}
    if proposal.get("validated") is False:
        return "AnalyzeResult"
    return "LaunchTrial"


def after_update(state: dict[str, Any]) -> Literal["ProposeTrial", "__end__", "HumanReview"]:
    """Route the study based on the current decision."""

    decision = state.get("decision")
    if decision == "continue":
        return "ProposeTrial"
    if decision == "human_review":
        return "HumanReview"
    return "__end__"
