# File: app/graph/build_graph.py
# Purpose: Assemble the LangGraph workflow used by the closed-loop experiment loop.
# License: GPL-3.0-or-later
"""Build the LangGraph workflow for the closed-loop study."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from app.graph import edges, nodes


def build_graph(runtime: dict[str, Any]):
    """Create the compiled LangGraph state machine."""

    graph = StateGraph(dict)
    graph.add_node("InitStudy", partial(nodes.init_study, deps=runtime))
    graph.add_node("LoadHistory", partial(nodes.load_history, deps=runtime))
    graph.add_node("ProposeTrial", partial(nodes.propose_trial, deps=runtime))
    graph.add_node("ValidateProposal", partial(nodes.validate_proposal_node, deps=runtime))
    graph.add_node("LaunchTrial", partial(nodes.launch_trial, deps=runtime))
    graph.add_node("CollectMetrics", partial(nodes.collect_metrics, deps=runtime))
    graph.add_node("AnalyzeResult", partial(nodes.analyze_result_node, deps=runtime))
    graph.add_node("UpdateStudy", partial(nodes.update_study, deps=runtime))
    graph.add_node("DecideNextAction", partial(nodes.decide_next_action, deps=runtime))
    graph.add_node("HumanReview", partial(nodes.human_review, deps=runtime))

    graph.set_entry_point("InitStudy")
    graph.add_edge("InitStudy", "LoadHistory")
    graph.add_edge("LoadHistory", "ProposeTrial")
    graph.add_edge("ProposeTrial", "ValidateProposal")
    graph.add_conditional_edges("ValidateProposal", edges.after_validate, {"LaunchTrial": "LaunchTrial", "AnalyzeResult": "AnalyzeResult"})
    graph.add_edge("LaunchTrial", "CollectMetrics")
    graph.add_edge("CollectMetrics", "AnalyzeResult")
    graph.add_edge("AnalyzeResult", "UpdateStudy")
    graph.add_edge("UpdateStudy", "DecideNextAction")
    graph.add_conditional_edges("DecideNextAction", edges.after_update, {"ProposeTrial": "ProposeTrial", "HumanReview": "HumanReview", "__end__": END})
    graph.add_edge("HumanReview", END)
    return graph.compile()
