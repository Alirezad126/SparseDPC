"""Evaluation: metrics, closed-loop rollouts, and the comparison orchestrator."""
from .metrics import compute_metrics, control_smoothness, policy_forward_latency
from .collapse import CollapseCriteria, diagnose_tracking_collapse
from .rollout import rollout_closed_loop, make_test_data
from .evaluate import evaluate, default_methods, dpc_loss_reference, sample_scenario, aggregate
from .twotank import (
    make_twotank_scenarios,
    evaluate_twotank_nominal,
    evaluate_twotank_safety,
    violation_summary,
)

__all__ = [
    "compute_metrics",
    "control_smoothness",
    "policy_forward_latency",
    "CollapseCriteria",
    "diagnose_tracking_collapse",
    "rollout_closed_loop",
    "make_test_data",
    "evaluate",
    "default_methods",
    "dpc_loss_reference",
    "sample_scenario",
    "aggregate",
    "make_twotank_scenarios",
    "evaluate_twotank_nominal",
    "evaluate_twotank_safety",
    "violation_summary",
]
