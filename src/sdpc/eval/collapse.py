"""Diagnostics for oscillatory online-adaptation collapse."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

__all__ = ["CollapseCriteria", "diagnose_tracking_collapse"]


@dataclass(frozen=True)
class CollapseCriteria:
    reach_tolerance: float = 0.1
    reach_hold_steps: int = 10
    analysis_start_fraction: float = 0.25
    motion_deadband: float = 1.0e-3
    min_direction_reversals: int = 6
    min_backtrack_ratio: float = 0.15


def _single_trajectory(value: torch.Tensor, name: str) -> torch.Tensor:
    value = value.detach().cpu()
    if value.dim() == 3:
        if value.shape[0] != 1:
            raise ValueError(f"{name} must contain one trajectory")
        value = value[0]
    if value.dim() != 2:
        raise ValueError(f"{name} must have shape (T, n) or (1, T, n)")
    return value


def diagnose_tracking_collapse(
    x_traj: torch.Tensor,
    r_traj: torch.Tensor,
    criteria: CollapseCriteria = CollapseCriteria(),
) -> Dict[str, float]:
    """Classify sustained non-convergence with back-and-forth state motion.

    ``reached_reference`` requires the Euclidean tracking error to remain within
    ``reach_tolerance`` for ``reach_hold_steps`` consecutive samples. Oscillation
    is measured after an initial transient by sign reversals and backward travel
    along the initial-state-to-final-reference direction.
    """
    x = _single_trajectory(x_traj, "x_traj")
    r = _single_trajectory(r_traj, "r_traj")
    T = min(x.shape[0], r.shape[0])
    if T < 2:
        raise ValueError("collapse diagnostics require at least two samples")
    x, r = x[:T], r[:T]
    error = torch.linalg.vector_norm(x - r, dim=-1)

    hold = max(1, min(int(criteria.reach_hold_steps), T))
    inside = error <= float(criteria.reach_tolerance)
    reached = bool(inside[-hold:].all().item())
    reached_at = None
    if reached:
        reached_at = T - 1
        while reached_at > 0 and bool(inside[reached_at - 1].item()):
            reached_at -= 1

    start = min(max(1, int(criteria.analysis_start_fraction * T)), T - 1)
    direction = r[-1] - x[0]
    direction_norm = torch.linalg.vector_norm(direction)
    if float(direction_norm.item()) <= 1.0e-12:
        direction = torch.zeros_like(direction)
    else:
        direction = direction / direction_norm

    projected_step = (x[1:] - x[:-1]) @ direction
    projected_step = projected_step[start - 1:]
    active = projected_step[projected_step.abs() > float(criteria.motion_deadband)]
    if active.numel() >= 2:
        signs = torch.sign(active)
        reversals = int((signs[1:] != signs[:-1]).sum().item())
    else:
        reversals = 0
    forward_distance = float(torch.clamp(active, min=0.0).sum().item())
    backward_distance = float(torch.clamp(-active, min=0.0).sum().item())
    backtrack_ratio = backward_distance / (forward_distance + 1.0e-12)

    back_and_forth = (
        reversals >= int(criteria.min_direction_reversals)
        and backtrack_ratio >= float(criteria.min_backtrack_ratio)
    )
    tail_length = max(hold, max(1, int(0.1 * T)))
    tail = error[-tail_length:]
    return {
        "reached_reference": bool(reached),
        "time_to_reach": float(reached_at) if reached else float("nan"),
        "not_reached": bool(not reached),
        "back_and_forth": bool(back_and_forth),
        "collapsed": bool((not reached) and back_and_forth),
        "final_tracking_error": float(error[-1].item()),
        "minimum_tracking_error": float(error.min().item()),
        "tail_mean_tracking_error": float(tail.mean().item()),
        "direction_reversals": reversals,
        "forward_distance": forward_distance,
        "backward_distance": backward_distance,
        "backtrack_ratio": backtrack_ratio,
        "analysis_start_step": start,
    }
