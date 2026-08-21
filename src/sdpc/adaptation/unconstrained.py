"""Unconstrained reference-tracking online adaptation (Sec. 3.4.1).

Runs the reference step (Eq. 12) at every executed time step and applies the resulting
action to the real plant. No safety mechanism — this is the baseline that corrects the
steady-state offset under parametric uncertainty but is oblivious to constraints, and the
one the safe method (:mod:`sdpc.adaptation.safe`) is compared against. Supports both the
``autograd`` and ``symbolic`` sensitivity backends (paper's Two-Tank speed comparison).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..actions import ACTION_GRADIENT_MODES
from .jacobian import SymbolicJacobian, compute_updates_symbolic
from .reference import (
    adaptive_gamma_from_error,
    apply_policy_updates,
    compute_updates_discrete_ref,
    zero_xi_grads,
)
from .rollout import clamp_action, current_reference

__all__ = [
    "UnconstrainedAdaptationConfig",
    "PreparedUnconstrainedAdaptation",
    "prepare_unconstrained_adaptation",
    "run_unconstrained_adaptation",
]


@dataclass
class UnconstrainedAdaptationConfig:
    gamma_ref: float = 0.1
    clip_update: float = 0.5
    action_scale: float = 1.0
    ref_backend: str = "autograd"        # "autograd" or "symbolic"
    action_gradient_mode: str = "straight_through"  # clamp derivative used by adaptation
    action_gradient_band: float = 0.1
    action_gradient_leak: float = 0.05
    integration_method: str = "rk4"      # continuous symbolic prediction: "rk4" or "euler"
    adaptive_gamma: bool = False
    gamma_ref_min: float = 0.01
    gamma_ref_max: float = 0.2
    gamma_err_scale: float = 0.2


@dataclass(frozen=True)
class PreparedUnconstrainedAdaptation:
    """Backend objects built before the online reference-adaptation loop."""

    reference_jacobian: Optional[SymbolicJacobian]
    setup_time_s: float


def prepare_unconstrained_adaptation(
    policy,
    cfg: UnconstrainedAdaptationConfig,
    *,
    umin=None,
    umax=None,
    derivative_model=None,
    system=None,
) -> PreparedUnconstrainedAdaptation:
    """Build the configured reference derivative backend outside online execution."""
    setup_t0 = time.perf_counter()
    if cfg.ref_backend not in {"autograd", "symbolic"}:
        raise ValueError("ref_backend must be 'autograd' or 'symbolic'")
    if cfg.action_gradient_mode not in ACTION_GRADIENT_MODES:
        raise ValueError(
            f"action_gradient_mode must be one of {sorted(ACTION_GRADIENT_MODES)}"
        )
    if cfg.action_gradient_band < 0.0:
        raise ValueError("action_gradient_band must be nonnegative")
    if not 0.0 <= cfg.action_gradient_leak <= 1.0:
        raise ValueError("action_gradient_leak must lie in [0, 1]")
    if cfg.ref_backend == "symbolic" and derivative_model is None:
        raise ValueError(
            "symbolic unconstrained adaptation requires "
            "derivative_model=<SINDyVectorized dynamics model>"
        )
    reference_jacobian = (
        SymbolicJacobian(
            derivative_model, policy, umin, umax,
            system=system, action_scale=cfg.action_scale,
            integration_method=cfg.integration_method,
        )
        if cfg.ref_backend == "symbolic" else None
    )
    return PreparedUnconstrainedAdaptation(
        reference_jacobian=reference_jacobian,
        setup_time_s=time.perf_counter() - setup_t0,
    )


def run_unconstrained_adaptation(
    policy,
    plant,
    data: Dict[str, torch.Tensor],
    cfg: UnconstrainedAdaptationConfig,
    *,
    umin=None,
    umax=None,
    derivative_model=None,
    system=None,
    prepared: Optional[PreparedUnconstrainedAdaptation] = None,
) -> Dict:
    """Run reference-tracking adaptation over ``data['r']`` from ``data['xn']``.

    Returns ``x_traj`` (B, T, nx), ``u_traj`` (B, T, nu), and per-step ``logs`` (tracking
    error, applied step time). Each update uses only the reference active at its current
    data index. Timing is reported per control step for the runtime metric.
    """
    internal_setup_t0 = time.perf_counter()
    if prepared is None:
        prepared = prepare_unconstrained_adaptation(
            policy, cfg, umin=umin, umax=umax,
            derivative_model=derivative_model, system=system,
        )
    jac = prepared.reference_jacobian
    if cfg.ref_backend == "symbolic":
        if jac is None:
            raise ValueError("prepared symbolic reference Jacobian is missing")
        if jac.policy is not policy:
            raise ValueError("prepared reference Jacobian belongs to a different policy")
    internal_setup_time = time.perf_counter() - internal_setup_t0

    x_data = data["xn"].detach()
    r_data = data["r"].detach()
    B, _, nx = x_data.shape
    T = r_data.shape[1]

    x0 = x_data[:, 0, :]
    with torch.no_grad():
        u0 = clamp_action(policy(x0, current_reference(r_data, 0)), umin, umax, cfg.action_scale)
    nu = u0.shape[1]

    x_traj = torch.empty(B, T, nx, device=x_data.device, dtype=x_data.dtype)
    u_traj = torch.empty(B, T, nu, device=x_data.device, dtype=x_data.dtype)
    x_traj[:, 0, :] = x0
    u_traj[:, 0, :] = u0

    logs: List[Dict] = []
    online_t0 = time.perf_counter()
    for t in range(1, T):
        step_t0 = time.perf_counter()
        x = x_traj[:, t - 1, :].detach()
        r = current_reference(r_data, t)

        zero_xi_grads(policy)
        if cfg.ref_backend == "symbolic":
            updates = compute_updates_symbolic(
                jac, x, r, action_gradient_mode=cfg.action_gradient_mode,
                action_gradient_band=cfg.action_gradient_band,
                action_gradient_leak=cfg.action_gradient_leak,
            )
        else:
            updates = compute_updates_discrete_ref(
                plant, policy, x, r,
                umin=umin, umax=umax, action_scale=cfg.action_scale,
                action_gradient_mode=cfg.action_gradient_mode,
                action_gradient_band=cfg.action_gradient_band,
                action_gradient_leak=cfg.action_gradient_leak,
            )

        if cfg.adaptive_gamma:
            gamma_t = adaptive_gamma_from_error(
                x, r, gamma_min=cfg.gamma_ref_min, gamma_max=cfg.gamma_ref_max,
                err_scale=cfg.gamma_err_scale,
            )
        else:
            gamma_t = cfg.gamma_ref
        zero_xi_grads(policy)
        apply_policy_updates(policy, updates, gamma=gamma_t, clip=cfg.clip_update)

        with torch.no_grad():
            u_applied = clamp_action(policy(x, r), umin, umax, cfg.action_scale)
            x_next = plant(x, u_applied)
        x_traj[:, t, :] = x_next
        u_traj[:, t, :] = u_applied

        logs.append({
            "t": t,
            "gamma_ref": gamma_t,
            "track_err": float(torch.linalg.norm(r - x_next, dim=-1).mean().item()),
            "step_time": time.perf_counter() - step_t0,
        })

    online_time = time.perf_counter() - online_t0
    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "logs": logs,
        "timing": {
            "internal_setup_s": internal_setup_time,
            "prepared_setup_s": prepared.setup_time_s,
            "online_total_s": online_time,
        },
    }
