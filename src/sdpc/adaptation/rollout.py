"""Differentiable finite-horizon closed-loop rollout used by online adaptation.

``x_{t+1} = f(x_t, Pi_U(pi_Xi(x_t, r_t)))`` unrolled for ``N`` steps. When ``grad=True``
the graph is retained so the predictive barrier loss (Eq. 18) can be back-propagated to the
policy coefficients, exactly as in differentiable predictive control. This is a light,
dependency-free replacement for wrapping the plant in a Neuromancer ``System`` just to roll
out a single trajectory.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import torch

from ..actions import bounded_action

__all__ = ["clamp_action", "current_reference", "hold_current_reference", "predict_rollout"]


def clamp_action(
    u: torch.Tensor,
    umin,
    umax,
    action_scale: float = 1.0,
    *,
    gradient_mode: str = "exact",
    gradient_band: float = 0.1,
    gradient_leak: float = 0.05,
) -> torch.Tensor:
    """Scale and clamp the action using the repository-wide bounded-action convention.

    ``gradient_mode="straight_through"`` leaves the forward value exactly clamped but
    uses ``du_clamped/du_raw = 1`` during autograd. It is a surrogate derivative for
    saturated adaptation updates; applied plant actions remain clamped.
    """
    return bounded_action(
        u, umin, umax, action_scale, gradient_mode=gradient_mode,
        gradient_band=gradient_band, gradient_leak=gradient_leak,
    )


def current_reference(r_data: torch.Tensor, t: int) -> torch.Tensor:
    """Return the reference active at control index ``t`` without future look-ahead."""
    if r_data.dim() != 3:
        raise ValueError("reference data must have shape (batch, time, reference_dim)")
    if not 0 <= int(t) < r_data.shape[1]:
        raise IndexError(f"reference index {t} is outside [0, {r_data.shape[1]})")
    return r_data[:, int(t), :].detach()


def hold_current_reference(r_data: torch.Tensor, t: int, horizon: int) -> torch.Tensor:
    """Hold ``r_data[:, t]`` fixed over a predictive adaptation horizon.

    A future reference switch is intentionally invisible until its actual control index.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("reference horizon must be positive")
    r = current_reference(r_data, t)
    return r.unsqueeze(1).expand(r.shape[0], horizon, r.shape[1])


def predict_rollout(
    policy,
    plant,
    x0: torch.Tensor,
    r_seq: torch.Tensor,
    n_steps: int,
    *,
    umin=None,
    umax=None,
    action_scale: float = 1.0,
    grad: bool = False,
    action_gradient_mode: str = "exact",
    action_gradient_band: float = 0.1,
    action_gradient_leak: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll the closed loop ``n_steps`` from ``x0``.

    Parameters
    ----------
    policy : callable ``(x, r) -> u_raw`` (a :class:`SINDyVectorized` or a Node-like fn).
    plant  : callable ``(x, u) -> x_next`` (discrete map).
    x0     : (B, nx) initial state.
    r_seq  : (B, n_steps, nref) reference sequence, or (B, nref) held constant.
    grad   : keep the autograd graph (for the barrier gradient) if True.

    Returns
    -------
    (x_traj, u_traj) with shapes (B, n_steps+1, nx) and (B, n_steps, nu).
    """
    ctx = nullcontext() if grad else torch.no_grad()
    xs = [x0.unsqueeze(1)]
    us = []
    x = x0
    with ctx:
        for t in range(n_steps):
            r_t = current_reference(r_seq, t) if r_seq.dim() == 3 else r_seq
            u = clamp_action(
                policy(x, r_t), umin, umax, action_scale,
                gradient_mode=action_gradient_mode,
                gradient_band=action_gradient_band,
                gradient_leak=action_gradient_leak,
            )
            us.append(u.unsqueeze(1))
            x = plant(x, u)
            xs.append(x.unsqueeze(1))
    x_traj = torch.cat(xs, dim=1)
    u_traj = torch.cat(us, dim=1)
    return x_traj, u_traj
