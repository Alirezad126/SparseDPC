"""Closed-loop rollout runners for evaluation (no adaptation).

These roll a *fixed* policy through a plant over a reference sequence, for the nominal-model
and model-mismatch comparisons. Adaptation rollouts live in :mod:`sdpc.adaptation`.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from ..adaptation.rollout import predict_rollout

__all__ = ["rollout_closed_loop", "make_test_data"]


def rollout_closed_loop(
    policy,
    plant,
    data: Dict[str, torch.Tensor],
    *,
    umin=None,
    umax=None,
    action_scale: float = 1.0,
) -> Dict:
    """Roll ``policy`` through ``plant`` following ``data['r']`` from ``data['xn']``.

    Returns ``x_traj`` (B, T, nx), ``u_traj`` (B, T-1, nu) and per-step wall-clock timing.
    """
    x0 = data["xn"][:, 0, :]
    r = data["r"]
    T = r.shape[1]
    t0 = time.perf_counter()
    x_traj, u_traj = predict_rollout(
        policy, plant, x0, r[:, : T - 1, :], T - 1,
        umin=umin, umax=umax, action_scale=action_scale, grad=False,
    )
    elapsed = time.perf_counter() - t0
    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "r_traj": r,
        "rollout_time_s": elapsed,
        "per_step_s": elapsed / max(T - 1, 1),
    }


def make_test_data(
    nx: int,
    nsteps: int,
    x0,
    ref,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build a single-trajectory ``{'xn','r'}`` dict from an initial state and a reference.

    ``ref`` may be a constant (broadcast over the horizon) or a full (nsteps+1, nref) array.
    """
    device = device or torch.device("cpu")
    xn = torch.as_tensor(x0, dtype=torch.float32, device=device).reshape(1, 1, nx)
    ref_t = torch.as_tensor(ref, dtype=torch.float32, device=device)
    if ref_t.dim() == 1:
        ref_t = ref_t.reshape(1, 1, -1).expand(1, nsteps + 1, ref_t.numel()).contiguous()
    else:
        ref_t = ref_t.reshape(1, ref_t.shape[0], ref_t.shape[1])
    return {"xn": xn, "r": ref_t}
