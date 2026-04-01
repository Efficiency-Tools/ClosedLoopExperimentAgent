"""Tests for guardrail validation and proposal safety checks."""

from app.guards.validators import validate_proposal
from app.optimizer.search_space import load_search_space


def test_validate_proposal_rejects_protected_keys():
    spec = load_search_space("examples/toy_pytorch/search_space.yaml")
    result = validate_proposal(
        {
            "optimizer.lr": 1e-4,
            "model.backbone": "other",
        },
        spec,
    )
    assert not result.valid
