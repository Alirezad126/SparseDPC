"""Predictive Safety Filter baseline for safe online adaptation.

The policy receives the same reference-tracking coefficient update as the
unconstrained method.  A receding-horizon nonlinear program then minimally
modifies the learned action in input space while enforcing the hard
``SafetySpec`` constraints under the exact configured deployment model.

The CasADi problem is built once before the online loop and warm-started at
each step.  Solver setup time and per-step filter time are reported separately.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

try:  # pragma: no cover - environment dependent
    import casadi as ca

    _HAS_CASADI = True
except Exception:  # pragma: no cover
    ca = None
    _HAS_CASADI = False

from ..actions import ACTION_GRADIENT_MODES
from ..safety.specs import SafetySpec, StateConstraint
from .reference import (
    adaptive_gamma_from_error,
    apply_policy_updates,
    compute_updates_discrete_ref,
    zero_xi_grads,
)
from .rollout import clamp_action, current_reference

__all__ = ["PSFConfig", "run_psf_adaptation", "casadi_available"]


def casadi_available() -> bool:
    return _HAS_CASADI


@dataclass
class PSFConfig:
    horizon: int = 20
    gamma_ref: float = 0.1
    clip_update: float = 0.5
    action_scale: float = 1.0
    action_gradient_mode: str = "straight_through"
    action_gradient_band: float = 0.1
    action_gradient_leak: float = 0.05
    integration_method: str = "rk4"
    adaptive_gamma: bool = False
    gamma_ref_min: float = 0.01
    gamma_ref_max: float = 0.2
    gamma_err_scale: float = 0.2
    default_buffer: float = 0.0
    buffers: Dict[str, float] = field(default_factory=dict)
    tail_weight: float = 1.0e-6
    smooth_weight: float = 1.0e-4
    max_iter: int = 500
    tol: float = 1.0e-6
    verbose: bool = False


def _as_bound_vector(value, nu: int, *, default: float) -> np.ndarray:
    if value is None:
        return np.full(nu, default, dtype=float)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size == 1:
        out = np.full(nu, float(out.item()), dtype=float)
    if out.size != nu:
        raise ValueError(f"expected scalar or {nu} input bounds, got {out.size}")
    return out


def _constraint_buffer(con: StateConstraint, cfg: PSFConfig) -> float:
    kind = con.meta.get("kind", "")
    group = "box" if kind.startswith("box_") else "obstacle" if kind == "rotated_ellipse" else kind
    return float(cfg.buffers.get(con.name, cfg.buffers.get(group, cfg.default_buffer)))


def _state_margin_casadi(con: StateConstraint, x):
    meta = con.meta
    kind = meta.get("kind")
    if kind == "box_min":
        return x[int(meta["idx"])] - float(meta["bound"])
    if kind == "box_max":
        return float(meta["bound"]) - x[int(meta["idx"])]
    if kind == "rotated_ellipse":
        ix, iy = meta["idx"]
        dx = x[int(ix)] - float(meta["c"])
        dy = x[int(iy)] - float(meta["d"])
        ct = float(meta["cos_theta"])
        st = float(meta["sin_theta"])
        xr = ct * dx + st * dy
        yr = -st * dx + ct * dy
        return float(meta["b"]) * xr ** 2 + yr ** 2 - float(meta["boundary"])
    casadi_fn = meta.get("casadi_fn")
    if casadi_fn is not None:
        return casadi_fn(x)
    raise NotImplementedError(
        f"PSF has no CasADi expression for state constraint {con.name!r}; "
        "use box/rotated_ellipse constraints or provide meta['casadi_fn']"
    )


class _PSFSolver:
    def __init__(
        self,
        *,
        nx: int,
        nu: int,
        f_casadi: Callable,
        spec: SafetySpec,
        cfg: PSFConfig,
        lower: np.ndarray,
        upper: np.ndarray,
        safe_u_guess: Optional[Callable] = None,
    ):
        self.nx = nx
        self.nu = nu
        self.N = int(cfg.horizon)
        self.spec = spec
        self.cfg = cfg
        self.lower = lower
        self.upper = upper
        self.safe_u_guess = safe_u_guess
        self._U_previous = None
        self._lam_g_previous = None

        opti = ca.Opti()
        self.opti = opti
        self.X = opti.variable(nx, self.N + 1)
        self.U = opti.variable(nu, self.N)
        self.x0 = opti.parameter(nx)
        self.u_learn = opti.parameter(nu)
        self.u_previous = opti.parameter(nu)
        opti.subject_to(self.X[:, 0] == self.x0)

        lo_dm = ca.DM(lower)
        hi_dm = ca.DM(upper)
        objective = ca.sumsqr(self.U[:, 0] - self.u_learn)
        for k in range(self.N):
            opti.subject_to(self.X[:, k + 1] == f_casadi(self.X[:, k], self.U[:, k]))
            opti.subject_to(opti.bounded(lo_dm, self.U[:, k], hi_dm))
            for con in spec.state_constraints:
                margin = _state_margin_casadi(con, self.X[:, k + 1])
                opti.subject_to(margin >= float(con.delta) + _constraint_buffer(con, cfg))

            du = self.U[:, k] - (self.u_previous if k == 0 else self.U[:, k - 1])
            if spec.control_rate is not None:
                rate = spec.control_rate
                buffer = float(cfg.buffers.get(rate.name, cfg.buffers.get("du", 0.0)))
                if float(rate.delta) + buffer >= float(rate.du_max) ** 2:
                    raise ValueError(
                        "PSF du buffer must satisfy delta + buffer < du_max**2"
                    )
                opti.subject_to(
                    float(rate.du_max) ** 2 - ca.sumsqr(du) >= float(rate.delta) + buffer
                )
            if cfg.smooth_weight > 0.0:
                objective += float(cfg.smooth_weight) * ca.sumsqr(du)
            if k > 0 and cfg.tail_weight > 0.0:
                objective += float(cfg.tail_weight) * ca.sumsqr(self.U[:, k] - self.u_learn)

        opti.minimize(objective)
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": int(cfg.max_iter),
                "tol": float(cfg.tol),
                "acceptable_tol": max(float(cfg.tol) * 100.0, 1.0e-4),
                "acceptable_iter": 15,
                "mu_strategy": "adaptive",
                "bound_push": 1.0e-6,
                "bound_frac": 1.0e-6,
                "warm_start_init_point": "yes",
            },
        )
        x_sym = ca.MX.sym("psf_x", nx)
        u_sym = ca.MX.sym("psf_u", nu)
        self._f_numeric = ca.Function("psf_f_numeric", [x_sym, u_sym], [f_casadi(x_sym, u_sym)])

    def _shifted_guess(self, u_learn: np.ndarray) -> np.ndarray:
        if self._U_previous is None:
            return np.repeat(u_learn.reshape(-1, 1), self.N, axis=1)
        return np.concatenate([self._U_previous[:, 1:], self._U_previous[:, -1:]], axis=1)

    def _state_guess(self, x0: np.ndarray, U0: np.ndarray) -> np.ndarray:
        X0 = np.empty((self.nx, self.N + 1), dtype=float)
        X0[:, 0] = x0
        for k in range(self.N):
            X0[:, k + 1] = np.asarray(self._f_numeric(X0[:, k], U0[:, k])).reshape(-1)
        return X0

    def _ramp_guess(self, u_previous: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Move toward a safe hold input without violating the configured rate limit."""
        out = np.empty((self.nu, self.N), dtype=float)
        current = np.asarray(u_previous, dtype=float).copy()
        target = np.clip(np.asarray(target, dtype=float), self.lower, self.upper)
        max_du = float("inf")
        if self.spec.control_rate is not None:
            rate = self.spec.control_rate
            buffer = float(self.cfg.buffers.get(rate.name, self.cfg.buffers.get("du", 0.0)))
            max_du = np.sqrt(max(float(rate.du_max) ** 2 - float(rate.delta) - buffer, 0.0))
        for k in range(self.N):
            delta = target - current
            norm = float(np.linalg.norm(delta))
            if norm > max_du and norm > 0.0:
                delta *= max_du / norm
            current = np.clip(current + delta, self.lower, self.upper)
            out[:, k] = current
        return out

    def solve(self, x0: np.ndarray, u_learn: np.ndarray, u_previous: np.ndarray):
        call_t0 = time.perf_counter()
        preparation_t0 = time.perf_counter()
        opti = self.opti
        opti.set_value(self.x0, x0)
        opti.set_value(self.u_learn, u_learn)
        opti.set_value(self.u_previous, u_previous)

        guesses = [self._shifted_guess(u_learn)]
        repeated = np.repeat(u_learn.reshape(-1, 1), self.N, axis=1)
        if self._U_previous is not None and not np.allclose(guesses[0], repeated):
            guesses.append(repeated)
        previous = np.repeat(u_previous.reshape(-1, 1), self.N, axis=1)
        if not any(np.allclose(previous, guess) for guess in guesses):
            guesses.append(previous)
        if self.safe_u_guess is not None:
            safe_target = np.asarray(self.safe_u_guess(x0), dtype=float).reshape(self.nu)
            safe_guess = self._ramp_guess(u_previous, safe_target)
            if not any(np.allclose(safe_guess, guess) for guess in guesses):
                guesses.append(safe_guess)
        preparation_time = time.perf_counter() - preparation_t0

        last_error = None
        total_solve_time = 0.0
        initialization_time = 0.0
        for guess_id, U0 in enumerate(guesses):
            initialization_t0 = time.perf_counter()
            U0 = np.clip(U0, self.lower[:, None], self.upper[:, None])
            opti.set_initial(self.U, U0)
            opti.set_initial(self.X, self._state_guess(x0, U0))
            if guess_id == 0 and self._lam_g_previous is not None:
                opti.set_initial(opti.lam_g, self._lam_g_previous)
            initialization_time += time.perf_counter() - initialization_t0
            solve_t0 = time.perf_counter()
            try:
                sol = opti.solve()
                total_solve_time += time.perf_counter() - solve_t0
                extraction_t0 = time.perf_counter()
                U_sol = np.asarray(sol.value(self.U), dtype=float).reshape(self.nu, self.N)
                X_sol = np.asarray(sol.value(self.X), dtype=float).reshape(self.nx, self.N + 1)
                self._U_previous = U_sol
                self._lam_g_previous = np.asarray(sol.value(opti.lam_g), dtype=float)
                stats = sol.stats()
                extraction_time = time.perf_counter() - extraction_t0
                call_time = time.perf_counter() - call_t0
                attributed_time = (
                    preparation_time + initialization_time + total_solve_time + extraction_time
                )
                return U_sol[:, 0], X_sol, U_sol, {
                    "success": True,
                    "status": str(stats.get("return_status", "Solve_Succeeded")),
                    "solver_iters": int(stats.get("iter_count", 0)),
                    "filter_time": total_solve_time,
                    "filter_prepare_time": preparation_time,
                    "filter_initialization_time": initialization_time,
                    "filter_extraction_time": extraction_time,
                    "filter_call_time": call_time,
                    "filter_unattributed_time": max(0.0, call_time - attributed_time),
                    "guess_id": guess_id,
                }
            except RuntimeError as exc:
                total_solve_time += time.perf_counter() - solve_t0
                last_error = exc

        raise RuntimeError(f"PSF could not find a hard-feasible plan: {last_error}")


def _min_state_margin(spec: SafetySpec, x: torch.Tensor) -> float:
    clearances = [float((con.margin(x) - con.delta).min().item()) for con in spec.state_constraints]
    return min(clearances) if clearances else float("inf")


def run_psf_adaptation(
    policy,
    plant,
    data: Dict[str, torch.Tensor],
    spec: SafetySpec,
    cfg: PSFConfig,
    *,
    system,
    system_cfg: Dict,
    umin=None,
    umax=None,
) -> Dict:
    """Run reference adaptation followed by a hard predictive input filter.

    The returned ``setup_time_s`` is solver construction only. ``wall_time_s``
    measures the online loop, while every log records ``filter_time`` and total
    ``step_time`` separately. Only the reference active at the current data index is
    used; the PSF optimization never receives future scheduled references.
    """
    runner_t0 = time.perf_counter()
    if not _HAS_CASADI:
        raise RuntimeError("run_psf_adaptation requires CasADi in the active environment")
    if system is None:
        raise ValueError("PSF requires system=<System> for exact CasADi deployment dynamics")
    if cfg.action_gradient_mode not in ACTION_GRADIENT_MODES:
        raise ValueError(
            f"action_gradient_mode must be one of {sorted(ACTION_GRADIENT_MODES)}"
        )
    if cfg.action_gradient_band < 0.0:
        raise ValueError("action_gradient_band must be nonnegative")
    if not 0.0 <= cfg.action_gradient_leak <= 1.0:
        raise ValueError("action_gradient_leak must lie in [0, 1]")
    hooks = system.casadi_hooks(system_cfg, integration_method=cfg.integration_method)
    if hooks is None or "f_casadi" not in hooks:
        raise NotImplementedError(f"{system.name} does not provide PSF CasADi dynamics")

    x_data = data["xn"].detach()
    r_data = data["r"].detach()
    if x_data.shape[0] != 1:
        raise ValueError("PSF runner currently supports one online trajectory (B=1)")
    T = r_data.shape[1]
    nx = x_data.shape[2]
    x0 = x_data[:, 0, :]
    with torch.no_grad():
        u0 = clamp_action(policy(x0, current_reference(r_data, 0)), umin, umax, cfg.action_scale)
    nu = u0.shape[-1]

    lower = _as_bound_vector(umin, nu, default=-np.inf)
    upper = _as_bound_vector(umax, nu, default=np.inf)
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)

    setup_t0 = time.perf_counter()
    solver = _PSFSolver(
        nx=nx,
        nu=nu,
        f_casadi=hooks["f_casadi"],
        spec=spec,
        cfg=cfg,
        lower=lower,
        upper=upper,
        safe_u_guess=hooks.get("psf_u_guess"),
    )
    setup_time = time.perf_counter() - setup_t0

    x_traj = torch.empty(1, T, nx, device=x_data.device, dtype=x_data.dtype)
    u_traj = torch.empty(1, T, nu, device=x_data.device, dtype=x_data.dtype)
    u_learning_traj = torch.empty_like(u_traj)
    x_traj[:, 0, :] = x0
    u_traj[:, 0, :] = u0
    u_learning_traj[:, 0, :] = u0

    if spec.any_violation(x0.unsqueeze(1)):
        raise RuntimeError("PSF requires an initially safe state")

    logs: List[Dict] = []
    runner_setup_time = time.perf_counter() - runner_t0
    online_t0 = time.perf_counter()
    for t in range(1, T):
        step_t0 = time.perf_counter()
        x = x_traj[:, t - 1, :].detach()
        r = current_reference(r_data, t)

        reference_t0 = time.perf_counter()
        reference_gradient_t0 = time.perf_counter()
        zero_xi_grads(policy)
        updates = compute_updates_discrete_ref(
            plant, policy, x, r,
            umin=umin, umax=umax, action_scale=cfg.action_scale,
            action_gradient_mode=cfg.action_gradient_mode,
            action_gradient_band=cfg.action_gradient_band,
            action_gradient_leak=cfg.action_gradient_leak,
        )
        reference_gradient_time = time.perf_counter() - reference_gradient_t0
        reference_apply_t0 = time.perf_counter()
        if cfg.adaptive_gamma:
            gamma_t = adaptive_gamma_from_error(
                x,
                r,
                gamma_min=cfg.gamma_ref_min,
                gamma_max=cfg.gamma_ref_max,
                err_scale=cfg.gamma_err_scale,
            )
        else:
            gamma_t = cfg.gamma_ref
        zero_xi_grads(policy)
        apply_policy_updates(policy, updates, gamma=gamma_t, clip=cfg.clip_update)

        with torch.no_grad():
            u_learning = clamp_action(policy(x, r), umin, umax, cfg.action_scale)
        reference_apply_time = time.perf_counter() - reference_apply_t0
        reference_time = time.perf_counter() - reference_t0

        filter_t0 = time.perf_counter()
        u_filtered, X_plan, U_plan, solve_info = solver.solve(
            x.squeeze(0).cpu().numpy(),
            u_learning.squeeze(0).cpu().numpy(),
            u_traj[:, t - 1, :].squeeze(0).cpu().numpy(),
        )
        filter_wall_time = time.perf_counter() - filter_t0
        u_apply = torch.as_tensor(u_filtered, dtype=x.dtype, device=x.device).unsqueeze(0)
        correction_norm = float(torch.linalg.norm(u_apply - u_learning).item())
        plant_t0 = time.perf_counter()
        with torch.no_grad():
            x_next = plant(x, u_apply)
        plant_time = time.perf_counter() - plant_t0

        diagnostics_t0 = time.perf_counter()
        x_traj[:, t, :] = x_next
        u_traj[:, t, :] = u_apply
        u_learning_traj[:, t, :] = u_learning
        realized_state_min = _min_state_margin(spec, x_next.unsqueeze(1))
        predicted = torch.as_tensor(X_plan.T, dtype=x.dtype, device=x.device).unsqueeze(0)
        predicted_state_min = _min_state_margin(spec, predicted[:, 1:, :])
        realized_rate_min = float("inf")
        predicted_rate_min = float("inf")
        if spec.control_rate is not None:
            rate = spec.control_rate
            previous = u_traj[:, t - 1, :]
            realized_du_sq = ((u_apply - previous) ** 2).sum(dim=-1)
            realized_rate_min = float((rate.margin(realized_du_sq) - rate.delta).min().item())
            planned_u = torch.as_tensor(U_plan.T, dtype=x.dtype, device=x.device).unsqueeze(0)
            first_du = planned_u[:, :1, :] - previous.unsqueeze(1)
            later_du = planned_u[:, 1:, :] - planned_u[:, :-1, :]
            planned_du = torch.cat([first_du, later_du], dim=1)
            planned_du_sq = (planned_du ** 2).sum(dim=-1)
            predicted_rate_min = float((rate.margin(planned_du_sq) - rate.delta).min().item())
        realized_min = min(realized_state_min, realized_rate_min)
        predicted_min = min(predicted_state_min, predicted_rate_min)
        diagnostics_time = time.perf_counter() - diagnostics_t0
        step_time = time.perf_counter() - step_t0
        rec = {
            "t": t,
            "gamma_ref": gamma_t,
            "track_err": float(torch.linalg.norm(r - x_next, dim=-1).mean().item()),
            "correction_norm": correction_norm,
            "min_pred_margin": predicted_min,
            "min_pred_state_margin": predicted_state_min,
            "min_pred_control_rate_margin": predicted_rate_min,
            "realized_min_margin": realized_min,
            "realized_state_margin": realized_state_min,
            "realized_control_rate_margin": realized_rate_min,
            "realized_violation": bool(realized_min < -max(float(cfg.tol), 1.0e-7)),
            "reference_time": reference_time,
            "reference_gradient_time": reference_gradient_time,
            "reference_apply_time": reference_apply_time,
            "filter_wall_time": filter_wall_time,
            "plant_time": plant_time,
            "diagnostics_time": diagnostics_time,
            "step_time": step_time,
            **solve_info,
        }
        logs.append(rec)
        if cfg.verbose:
            print(
                f"t={t:04d} status={rec['status']} iters={rec['solver_iters']:3d} "
                f"filter={1e3 * rec['filter_time']:.2f} ms correction={rec['correction_norm']:.3e}"
            )

    online_time = time.perf_counter() - online_t0
    reference_total = sum(row["reference_time"] for row in logs)
    reference_gradient_total = sum(row["reference_gradient_time"] for row in logs)
    reference_apply_total = sum(row["reference_apply_time"] for row in logs)
    filter_wall_total = sum(row["filter_wall_time"] for row in logs)
    filter_call_total = sum(row["filter_call_time"] for row in logs)
    filter_solver_total = sum(row["filter_time"] for row in logs)
    filter_prepare_total = sum(row["filter_prepare_time"] for row in logs)
    filter_initialization_total = sum(row["filter_initialization_time"] for row in logs)
    filter_extraction_total = sum(row["filter_extraction_time"] for row in logs)
    filter_unattributed_total = sum(row["filter_unattributed_time"] for row in logs)
    plant_total = sum(row["plant_time"] for row in logs)
    diagnostics_total = sum(row["diagnostics_time"] for row in logs)
    measured_components = reference_total + filter_wall_total + plant_total + diagnostics_total
    timing = {
        "solver_setup_s": setup_time,
        "setup_overhead_s": max(0.0, runner_setup_time - setup_time),
        "setup_total_s": runner_setup_time,
        "reference_update_total_s": reference_total,
        "reference_gradient_total_s": reference_gradient_total,
        "reference_apply_total_s": reference_apply_total,
        "reference_overhead_s": max(
            0.0, reference_total - reference_gradient_total - reference_apply_total
        ),
        "filter_prepare_total_s": filter_prepare_total,
        "filter_initialization_total_s": filter_initialization_total,
        "filter_solver_total_s": filter_solver_total,
        "filter_extraction_total_s": filter_extraction_total,
        "filter_unattributed_total_s": filter_unattributed_total,
        "filter_outer_overhead_s": max(0.0, filter_wall_total - filter_call_total),
        "filter_wrapper_total_s": filter_wall_total,
        "filter_wrapper_overhead_s": max(0.0, filter_wall_total - filter_solver_total),
        "plant_step_total_s": plant_total,
        "diagnostics_total_s": diagnostics_total,
        "online_overhead_s": max(0.0, online_time - measured_components),
        "online_total_s": online_time,
        "total_s": runner_setup_time + online_time,
    }
    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "u_learning_traj": u_learning_traj,
        "logs": logs,
        "setup_time_s": setup_time,
        "wall_time_s": online_time,
        "timing": timing,
    }
