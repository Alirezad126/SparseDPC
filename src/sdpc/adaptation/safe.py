"""Predictive barrier-augmented safe online adaptation (Algorithm 2).

Per executed time step ``k``:

1. **Reference step** — one Levenberg-Marquardt-conditioned update toward ``r_k`` (Eq. 12),
   using either the autograd sensitivity (:mod:`sdpc.adaptation.reference`) or the analytic
   Jacobian (:mod:`sdpc.adaptation.jacobian`).
2. **Predict** the ``N``-step closed-loop rollout from ``x_k`` under the candidate policy,
   holding the currently active ``r_k`` constant. Future scheduled references are not
   visible until their actual execution index.
3. **Safety correction** — while the predictive barrier loss ``J_h`` (Eq. 18) is positive,
   descend the clipped barrier gradient in coefficient space (Eq. 17) and re-roll, until
   ``J_h = 0`` (predicted-safe certificate, Eq. 20) or a guard cap is reached. The barrier
   gradient itself can come from plain autograd through the unrolled rollout, or from the
   symbolic-Jacobian forward-sensitivity backend
   (:class:`sdpc.adaptation.safety_jacobian.SymbolicSafetyJacobian`) — select via
   ``safety_backend``.
4. **Commit & apply** the adapted policy and step the real plant once.

This single runner is used by every example; the barrier terms come entirely from the
system's :class:`SafetySpec`, so TwoTank/VanDerPol (box constraints) and DoubleIntegrator
(obstacle) all use the *same* barrier-based method — no per-example ReLU code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from ..actions import ACTION_GRADIENT_MODES
from ..safety.specs import SafetySpec
from .barrier import rollout_barrier_loss, rollout_box_barrier_loss, rollout_box_relu_loss
from .jacobian import SymbolicJacobian, compute_updates_symbolic
from .safety_jacobian import SymbolicSafetyJacobian
from .analytic import sindy_step
from .reference import (
    adaptive_gamma_from_error,
    apply_policy_updates,
    backup_policy_Xi,
    clip_grad_list_by_global_norm,
    compute_updates_discrete_ref,
    restore_policy_Xi,
    zero_xi_grads,
)
from .rollout import clamp_action, current_reference, hold_current_reference, predict_rollout

__all__ = [
    "SafeAdaptationConfig",
    "PreparedSafeAdaptation",
    "prepare_safe_adaptation",
    "run_safe_adaptation",
]


BOX_RELU_CONSTRAINT_KINDS = {"box_relu_constraints", "box_relu_hinge"}
BOX_BARRIER_CONSTRAINT_KINDS = {
    "box_barrier_constraints": "squared_hinge",
    "box_barrier_hinge": "squared_hinge",
    "box_relaxed_log_constraints": "relaxed_log",
}


@dataclass
class SafeAdaptationConfig:
    horizon: int = 20                      # N-step prediction horizon
    gamma_ref: float = 0.05                # reference-step gain (if adaptive gammas off)
    gamma_safe: float = 0.05               # safety-step gain (gamma_s in Eq. 17)
    clip_update: float = 0.5               # per-coefficient update clip
    safety_grad_max_norm: float = 10.0     # global-norm clip on the barrier gradient (clip_c)
    safety_gain: float = 1.0               # extra scalar c on the safety direction
    max_safety_iters: int = 30             # M_s guard on the inner loop
    barrier_kind: str = "squared_hinge"    # "squared_hinge", "relu_hinge", "relaxed_log", "box_relu_constraints", or "box_barrier_constraints"
    barrier_eps: float = 1e-6
    safety_loss_tol: float = 1e-10         # J_h <= tol counts as safe
    action_scale: float = 1.0
    integration_method: str = "rk4"        # continuous prediction: "rk4" or "euler"
    ref_backend: str = "autograd"          # "autograd" or "symbolic" (reference step, Eq. 12)
    safety_backend: str = "autograd"       # "autograd" or "symbolic" (safety-gradient step)
    action_gradient_mode: str = "straight_through"  # shared by reference and safety derivatives
    action_gradient_band: float = 0.1
    action_gradient_leak: float = 0.05
    # optional adaptive reference gain (disabled unless adaptive_gamma=True)
    adaptive_gamma: bool = False
    gamma_ref_min: float = 0.01
    gamma_ref_max: float = 0.2
    gamma_err_scale: float = 0.2
    verbose: bool = False


@dataclass(frozen=True)
class PreparedSafeAdaptation:
    """Prediction and derivative backends built before the online safety loop."""

    pred_plant: Callable
    reference_jacobian: Optional[SymbolicJacobian]
    safety_jacobian: Optional[SymbolicSafetyJacobian]
    prediction_setup_s: float
    reference_jacobian_setup_s: float
    safety_jacobian_setup_s: float
    setup_total_s: float


def prepare_safe_adaptation(
    policy,
    plant,
    cfg: SafeAdaptationConfig,
    *,
    umin=None,
    umax=None,
    pred_plant=None,
    derivative_model=None,
    system=None,
) -> PreparedSafeAdaptation:
    """Build only the prediction/Jacobian objects selected by ``cfg``."""
    setup_t0 = time.perf_counter()
    if cfg.ref_backend not in {"autograd", "symbolic"}:
        raise ValueError("ref_backend must be 'autograd' or 'symbolic'")
    if cfg.safety_backend not in {"autograd", "symbolic"}:
        raise ValueError("safety_backend must be 'autograd' or 'symbolic'")
    needs_symbolic = cfg.ref_backend == "symbolic" or cfg.safety_backend == "symbolic"
    if needs_symbolic and derivative_model is None:
        raise ValueError(
            "symbolic safe adaptation requires "
            "derivative_model=<SINDyVectorized dynamics model>"
        )

    component_t0 = time.perf_counter()
    if pred_plant is None:
        if derivative_model is not None:
            pred_plant = lambda x, u: sindy_step(
                derivative_model, x, u, system=system, method=cfg.integration_method
            )
        else:
            pred_plant = plant
    prediction_setup_s = time.perf_counter() - component_t0

    component_t0 = time.perf_counter()
    reference_jacobian = (
        SymbolicJacobian(
            derivative_model, policy, umin, umax,
            system=system, action_scale=cfg.action_scale,
            integration_method=cfg.integration_method,
        )
        if cfg.ref_backend == "symbolic" else None
    )
    reference_jacobian_setup_s = time.perf_counter() - component_t0

    component_t0 = time.perf_counter()
    safety_jacobian = (
        SymbolicSafetyJacobian(
            derivative_model, policy, umin, umax,
            system=system, action_scale=cfg.action_scale,
            integration_method=cfg.integration_method,
        )
        if cfg.safety_backend == "symbolic" else None
    )
    safety_jacobian_setup_s = time.perf_counter() - component_t0

    return PreparedSafeAdaptation(
        pred_plant=pred_plant,
        reference_jacobian=reference_jacobian,
        safety_jacobian=safety_jacobian,
        prediction_setup_s=prediction_setup_s,
        reference_jacobian_setup_s=reference_jacobian_setup_s,
        safety_jacobian_setup_s=safety_jacobian_setup_s,
        setup_total_s=time.perf_counter() - setup_t0,
    )


def _reference_updates(cfg, plant, policy, x, r, jac, umin, umax):
    if cfg.ref_backend == "symbolic":
        return compute_updates_symbolic(
            jac, x, r, action_gradient_mode=cfg.action_gradient_mode,
            action_gradient_band=cfg.action_gradient_band,
            action_gradient_leak=cfg.action_gradient_leak,
        )
    return compute_updates_discrete_ref(
        plant, policy, x, r,
        umin=umin, umax=umax, action_scale=cfg.action_scale,
        action_gradient_mode=cfg.action_gradient_mode,
        action_gradient_band=cfg.action_gradient_band,
        action_gradient_leak=cfg.action_gradient_leak,
    )


def run_safe_adaptation(
    policy,
    plant,
    data: Dict[str, torch.Tensor],
    spec: SafetySpec,
    cfg: SafeAdaptationConfig,
    *,
    umin=None,
    umax=None,
    pred_plant=None,
    derivative_model=None,
    system=None,
    logger=None,
    prepared: Optional[PreparedSafeAdaptation] = None,
) -> Dict:
    """Run Algorithm 2 over the reference sequence in ``data['r']`` from ``data['xn']``.

    ``plant`` steps the real (perturbed) system; ``pred_plant`` (default = ``plant``) is
    used for the N-step predictive rollout. Returns a dict with ``x_traj`` (B, T, nx),
    ``u_traj`` (B, T, nu), and per-step ``logs``.
    """
    runner_t0 = time.perf_counter()
    if cfg.action_gradient_mode not in ACTION_GRADIENT_MODES:
        raise ValueError(
            f"action_gradient_mode must be one of {sorted(ACTION_GRADIENT_MODES)}"
        )
    if cfg.action_gradient_band < 0.0:
        raise ValueError("action_gradient_band must be nonnegative")
    if not 0.0 <= cfg.action_gradient_leak <= 1.0:
        raise ValueError("action_gradient_leak must lie in [0, 1]")
    if prepared is None:
        prepared = prepare_safe_adaptation(
            policy, plant, cfg, umin=umin, umax=umax, pred_plant=pred_plant,
            derivative_model=derivative_model, system=system,
        )
        prediction_setup_time = prepared.prediction_setup_s
        reference_jacobian_setup_time = prepared.reference_jacobian_setup_s
        safety_jacobian_setup_time = prepared.safety_jacobian_setup_s
    else:
        prediction_setup_time = 0.0
        reference_jacobian_setup_time = 0.0
        safety_jacobian_setup_time = 0.0
    pred_plant = prepared.pred_plant
    jac = prepared.reference_jacobian
    safety_jac = prepared.safety_jacobian
    if cfg.ref_backend == "symbolic":
        if jac is None:
            raise ValueError("prepared symbolic reference Jacobian is missing")
        if jac.policy is not policy:
            raise ValueError("prepared reference Jacobian belongs to a different policy")
    if cfg.safety_backend == "symbolic":
        if safety_jac is None:
            raise ValueError("prepared symbolic safety Jacobian is missing")
        if safety_jac.policy is not policy:
            raise ValueError("prepared safety Jacobian belongs to a different policy")

    x_data = data["xn"].detach()
    r_data = data["r"].detach()
    B, _, nx = x_data.shape
    T = r_data.shape[1]
    device, dtype = x_data.device, x_data.dtype

    x0 = x_data[:, 0, :]
    with torch.no_grad():
        u0 = clamp_action(policy(x0, current_reference(r_data, 0)), umin, umax, cfg.action_scale)
    nu = u0.shape[1]

    x_traj = torch.empty(B, T, nx, device=device, dtype=dtype)
    u_traj = torch.empty(B, T, nu, device=device, dtype=dtype)
    x_traj[:, 0, :] = x0
    u_traj[:, 0, :] = u0

    logs: List[Dict] = []

    def _rollout_safety(x_in, r_horizon, u_prev, grad):
        action_gradient_mode = (
            cfg.action_gradient_mode
            if grad and cfg.safety_backend == "autograd"
            else "exact"
        )
        prediction_t0 = time.perf_counter()
        x_roll, u_roll = predict_rollout(
            policy, pred_plant, x_in, r_horizon, cfg.horizon,
            umin=umin, umax=umax, action_scale=cfg.action_scale, grad=grad,
            action_gradient_mode=action_gradient_mode,
            action_gradient_band=cfg.action_gradient_band,
            action_gradient_leak=cfg.action_gradient_leak,
        )
        prediction_time = time.perf_counter() - prediction_t0
        loss_t0 = time.perf_counter()
        if cfg.barrier_kind in BOX_RELU_CONSTRAINT_KINDS:
            loss, violation = rollout_box_relu_loss(x_roll, spec, include_x0=False)
            terms = {"box_relu_constraints": violation}
            margins = spec.margins(x_roll)
        elif cfg.barrier_kind in BOX_BARRIER_CONSTRAINT_KINDS:
            loss, terms = rollout_box_barrier_loss(
                x_roll, spec,
                barrier_kind=BOX_BARRIER_CONSTRAINT_KINDS[cfg.barrier_kind],
                eps=cfg.barrier_eps,
                include_x0=False,
            )
            margins = spec.margins(x_roll)
        else:
            loss, terms, margins = rollout_barrier_loss(
                x_roll, spec, kind=cfg.barrier_kind, eps=cfg.barrier_eps,
                u_traj=u_roll, u_prev=u_prev,
            )
        loss_time = time.perf_counter() - loss_t0
        return x_roll, u_roll, loss, terms, margins, prediction_time, loss_time

    setup_time = time.perf_counter() - runner_t0
    online_t0 = time.perf_counter()
    for t in range(1, T):
        step_t0 = time.perf_counter()
        x = x_traj[:, t - 1, :].detach()
        r = current_reference(r_data, t)
        u_prev = u_traj[:, t - 1, :].detach()
        r_horizon = hold_current_reference(r_data, t, cfg.horizon)

        # ---- 1. reference-tracking update ----
        reference_t0 = time.perf_counter()
        reference_gradient_t0 = time.perf_counter()
        zero_xi_grads(policy)
        ref_updates = _reference_updates(cfg, plant, policy, x, r, jac, umin, umax)
        reference_gradient_time = time.perf_counter() - reference_gradient_t0
        reference_apply_t0 = time.perf_counter()
        if cfg.adaptive_gamma:
            gamma_ref_t = adaptive_gamma_from_error(
                x, r, gamma_min=cfg.gamma_ref_min, gamma_max=cfg.gamma_ref_max,
                err_scale=cfg.gamma_err_scale,
            )
        else:
            gamma_ref_t = cfg.gamma_ref
        zero_xi_grads(policy)
        apply_policy_updates(policy, ref_updates, gamma=gamma_ref_t, clip=cfg.clip_update)
        reference_apply_time = time.perf_counter() - reference_apply_t0
        reference_time = time.perf_counter() - reference_t0

        # ---- 2-3. predict + safety inner loop ----
        m = 0
        safe_loss = float("inf")
        min_margin = float("inf")
        clipped = None
        safety_rollout_time = 0.0
        safety_prediction_time = 0.0
        safety_loss_time = 0.0
        safety_gradient_time = 0.0
        safety_derivative_time = 0.0
        safety_update_time = 0.0
        while True:
            rollout_t0 = time.perf_counter()
            if cfg.safety_backend == "autograd":
                zero_xi_grads(policy)
                x_roll, u_roll, loss, terms, margins, prediction_time, loss_time = _rollout_safety(
                    x, r_horizon, u_prev, grad=True
                )
            else:
                with torch.no_grad():
                    x_roll, u_roll, loss, terms, margins, prediction_time, loss_time = _rollout_safety(
                        x, r_horizon, u_prev, grad=False
                    )
            safety_prediction_time += prediction_time
            safety_loss_time += loss_time
            safe_loss = float(loss.detach().cpu().item())
            min_margin = min(float(v.min().item()) for v in margins.values()) if margins else float("inf")
            is_safe = safe_loss <= cfg.safety_loss_tol
            safety_rollout_time += time.perf_counter() - rollout_t0

            if is_safe or m >= cfg.max_safety_iters:
                if cfg.safety_backend == "autograd":
                    zero_xi_grads(policy)
                break

            # one clipped barrier-gradient descent step (Eq. 17)
            gradient_t0 = time.perf_counter()
            derivative_t0 = time.perf_counter()
            if cfg.safety_backend == "symbolic":
                # Forward-sensitivity recursion on the already-computed (no-grad) rollout
                # -- no autograd graph over the N-step unroll needed (safety_jacobian.py).
                raw, _ = safety_jac.rollout_sensitivity_grad(
                    x_roll, u_roll, r_horizon, spec,
                    kind=cfg.barrier_kind, eps=cfg.barrier_eps, u_prev=u_prev,
                    action_gradient_mode=cfg.action_gradient_mode,
                    action_gradient_band=cfg.action_gradient_band,
                    action_gradient_leak=cfg.action_gradient_leak,
                )
            else:
                loss.backward()
                raw = [(-p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                       for p in policy.Xi]
                zero_xi_grads(policy)
            safety_derivative_time += time.perf_counter() - derivative_t0
            update_t0 = time.perf_counter()
            clipped, gnorm, gscale = clip_grad_list_by_global_norm(
                raw, max_norm=cfg.safety_grad_max_norm
            )
            direction = [cfg.safety_gain * g for g in clipped]
            apply_policy_updates(policy, direction, gamma=cfg.gamma_safe, clip=cfg.clip_update)
            safety_update_time += time.perf_counter() - update_t0
            safety_gradient_time += time.perf_counter() - gradient_t0
            m += 1

        # ---- 4. commit & apply one real step ----
        plant_t0 = time.perf_counter()
        with torch.no_grad():
            u_applied = clamp_action(policy(x, r), umin, umax, cfg.action_scale)
            x_next = plant(x, u_applied)
        x_traj[:, t, :] = x_next
        u_traj[:, t, :] = u_applied
        plant_time = time.perf_counter() - plant_t0

        # realized-state safety (post-hoc, on the actually executed next state)
        diagnostics_t0 = time.perf_counter()
        realized_margins = spec.min_margins(x_next.unsqueeze(1))
        realized_violation = any(v < c.delta for c, v in zip(spec.state_constraints, realized_margins.values()))
        diagnostics_time = time.perf_counter() - diagnostics_t0
        step_time = time.perf_counter() - step_t0

        rec = {
            "t": t,
            "gamma_ref": gamma_ref_t,
            "safety_iters": m,
            "predicted_safe": bool(safe_loss <= cfg.safety_loss_tol),
            "barrier_loss": safe_loss,
            "reference tracking grad:": ref_updates,
            "safety_grads": clipped,
            "min_pred_margin": min_margin,
            "realized_min_margin": min(realized_margins.values()) if realized_margins else float("inf"),
            "realized_violation": bool(realized_violation),
            "reference_time": reference_time,
            "reference_gradient_time": reference_gradient_time,
            "reference_apply_time": reference_apply_time,
            "safety_rollout_time": safety_rollout_time,
            "safety_prediction_time": safety_prediction_time,
            "safety_loss_time": safety_loss_time,
            "safety_gradient_time": safety_gradient_time,
            "safety_derivative_time": safety_derivative_time,
            "safety_update_time": safety_update_time,
            "plant_time": plant_time,
            "diagnostics_time": diagnostics_time,
            "step_time": step_time,
        }
        logs.append(rec)
        if cfg.verbose and logger is not None:
            logger(rec)

    online_time = time.perf_counter() - online_t0
    reference_total = sum(row["reference_time"] for row in logs)
    reference_gradient_total = sum(row["reference_gradient_time"] for row in logs)
    reference_apply_total = sum(row["reference_apply_time"] for row in logs)
    rollout_total = sum(row["safety_rollout_time"] for row in logs)
    prediction_total = sum(row["safety_prediction_time"] for row in logs)
    loss_total = sum(row["safety_loss_time"] for row in logs)
    gradient_total = sum(row["safety_gradient_time"] for row in logs)
    derivative_total = sum(row["safety_derivative_time"] for row in logs)
    update_total = sum(row["safety_update_time"] for row in logs)
    plant_total = sum(row["plant_time"] for row in logs)
    diagnostics_total = sum(row["diagnostics_time"] for row in logs)
    measured_components = reference_total + rollout_total + gradient_total + plant_total + diagnostics_total
    timing = {
        "prediction_setup_s": prediction_setup_time,
        "reference_jacobian_setup_s": reference_jacobian_setup_time,
        "safety_jacobian_setup_s": safety_jacobian_setup_time,
        "setup_total_s": setup_time,
        "setup_overhead_s": max(
            0.0,
            setup_time
            - prediction_setup_time
            - reference_jacobian_setup_time
            - safety_jacobian_setup_time,
        ),
        "reference_update_total_s": reference_total,
        "reference_gradient_total_s": reference_gradient_total,
        "reference_apply_total_s": reference_apply_total,
        "reference_overhead_s": max(
            0.0, reference_total - reference_gradient_total - reference_apply_total
        ),
        "safety_rollout_total_s": rollout_total,
        "safety_prediction_total_s": prediction_total,
        "safety_loss_total_s": loss_total,
        "safety_rollout_overhead_s": max(0.0, rollout_total - prediction_total - loss_total),
        "safety_gradient_total_s": gradient_total,
        "safety_derivative_total_s": derivative_total,
        "safety_update_total_s": update_total,
        "safety_gradient_overhead_s": max(0.0, gradient_total - derivative_total - update_total),
        "plant_step_total_s": plant_total,
        "diagnostics_total_s": diagnostics_total,
        "online_overhead_s": max(0.0, online_time - measured_components),
        "online_total_s": online_time,
        "total_s": setup_time + online_time,
        "prepared_setup_total_s": prepared.setup_total_s,
    }
    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "logs": logs,
        "setup_time_s": setup_time,
        "wall_time_s": online_time,
        "timing": timing,
    }
