"""Barrier functions and the predictive barrier loss (Sec. 3.4.2 of the paper).

A barrier ``B(h; delta)`` is continuously differentiable, non-increasing, and satisfies
``B(h; delta) = 0  <=>  h >= delta`` (Eq. 19-20). Two choices are provided:

* ``squared_hinge`` — the paper's canonical ``B = 1/2 max(0, delta - h)^2``.
* ``relu_hinge`` — nonsquared ReLU violation.
* ``relaxed_log``  — the relaxed log-barrier used in the DoubleIntegrator notebooks,
  active only inside a clearance ``band`` above the hard threshold and linearly
  continued near the boundary to keep gradients finite.

Both are zero iff ``h >= delta + band`` so the "is-safe" test is identical regardless of
which barrier is selected; with ``band = 0`` the squared hinge matches Eq. 20 exactly.

:func:`rollout_barrier_loss` accumulates the barrier over a predicted trajectory for an
arbitrary :class:`~sdpc.safety.SafetySpec`, giving the ``J_h`` of Eq. 18. It is the single
implementation used by both the DoubleIntegrator (obstacle) and TwoTank/VanDerPol (box)
examples, replacing per-example ReLU code.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..safety.specs import SafetySpec

__all__ = [
    "squared_hinge",
    "relaxed_log_barrier",
    "barrier_value",
    "barrier_derivative",
    "rollout_barrier_loss",
    "rollout_box_relu_loss",
    "rollout_box_barrier_loss",
    "control_rate_sq_norm",
]


def squared_hinge(h: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """``1/2 * max(0, threshold - h)^2`` — zero iff ``h >= threshold``."""
    return 0.5 * torch.clamp(threshold - h, min=0.0) ** 2


def relaxed_log_barrier(
    h: torch.Tensor,
    delta: float,
    band: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Relaxed log-barrier on the clearance ``h - delta`` over an activation ``band``.

    Zero when ``h >= delta + band``; ``-log(hc/band)`` for ``eps < hc < band`` where
    ``hc = h - delta``; linearly continued for ``hc <= eps`` to avoid NaNs and keep a
    non-zero gradient. With ``delta = 0`` this reproduces the DoubleIntegrator barrier.
    """
    if band <= 0.0:
        raise ValueError("relaxed_log requires band > 0")
    hc = h - delta
    band_t = hc.new_tensor(band)
    eps_t = hc.new_tensor(eps)

    hc_clamped = torch.clamp(hc, min=eps_t)
    log_part = -torch.log(hc_clamped / band_t)
    linear_part = -torch.log(eps_t / band_t) + (eps_t - hc) / eps_t

    inside = torch.where(hc > eps_t, log_part, linear_part)
    return torch.where(hc < band_t, inside, torch.zeros_like(hc))


def barrier_value(
    h: torch.Tensor,
    *,
    delta: float,
    band: float,
    kind: str = "squared_hinge",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Evaluate the selected barrier; zero iff ``h >= delta + band`` for both kinds."""
    if kind == "squared_hinge":
        return squared_hinge(h, h.new_tensor(delta + band))
    if kind == "relu_hinge":
        return torch.clamp(h.new_tensor(delta + band) - h, min=0.0)
    if kind == "relaxed_log":
        return relaxed_log_barrier(h, delta=delta, band=band, eps=eps)
    raise ValueError(f"unknown barrier kind: {kind!r}")


def barrier_derivative(
    h: torch.Tensor,
    *,
    delta: float,
    band: float,
    kind: str = "squared_hinge",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Derivative ``dB/dh`` for :func:`barrier_value`."""
    if kind == "squared_hinge":
        threshold = h.new_tensor(delta + band)
        return torch.where(h < threshold, h - threshold, torch.zeros_like(h))
    if kind == "relu_hinge":
        threshold = h.new_tensor(delta + band)
        return torch.where(h < threshold, -torch.ones_like(h), torch.zeros_like(h))
    if kind == "relaxed_log":
        if band <= 0.0:
            raise ValueError("relaxed_log requires band > 0")
        hc = h - delta
        band_t = h.new_tensor(band)
        eps_t = h.new_tensor(eps)
        active = hc < band_t
        deriv = torch.where(hc > eps_t, -1.0 / torch.clamp(hc, min=eps_t), -1.0 / eps_t)
        return torch.where(active, deriv, torch.zeros_like(h))
    raise ValueError(f"unknown barrier kind: {kind!r}")


def control_rate_sq_norm(
    u_traj: torch.Tensor,
    u_prev: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Squared per-step control increments ``||u_k - u_{k-1}||^2`` over a (B, N, nu) rollout.

    If ``u_prev`` (B, nu) is given, the first increment is anchored to the previously
    executed real action so the very first predicted step is rate-limited too.
    """
    if u_prev is not None:
        u_aug = torch.cat([u_prev.detach().unsqueeze(1), u_traj], dim=1)
    else:
        u_aug = torch.cat([u_traj[:, :1, :], u_traj], dim=1)
    du = u_aug[:, 1:, :] - u_aug[:, :-1, :]
    return (du ** 2).sum(dim=-1)


def _box_bounds_from_spec(spec: SafetySpec, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    nx = x.shape[-1]
    xmin = torch.full((nx,), -torch.inf, device=x.device, dtype=x.dtype)
    xmax = torch.full((nx,), torch.inf, device=x.device, dtype=x.dtype)
    for con in spec.state_constraints:
        meta = getattr(con, "meta", {})
        kind = meta.get("kind")
        if kind == "box_min":
            xmin[int(meta["idx"])] = float(meta["bound"])
        elif kind == "box_max":
            xmax[int(meta["idx"])] = float(meta["bound"])
        else:
            raise ValueError("box ReLU loss requires only box_constraints")
    if bool(torch.isinf(xmin).any().item() or torch.isinf(xmax).any().item()):
        raise ValueError("box ReLU loss needs both min and max for every state")
    return xmin, xmax


def rollout_box_relu_loss(
    x_traj: torch.Tensor,
    spec: SafetySpec,
    *,
    include_x0: bool = False,
    reduction: str = "batch_sum_mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Box ReLU violation on future states.

    This is not a smooth barrier. It is the direct constraint penalty:
    ``relu(x - xmax) + relu(xmin - x)`` over ``x_1, ..., x_N`` by default.
    """
    x = x_traj if include_x0 else x_traj[:, 1:, :]
    xmin, xmax = _box_bounds_from_spec(spec, x)
    violation = torch.relu(x - xmax.view(1, 1, -1)) + torch.relu(xmin.view(1, 1, -1) - x)
    if reduction == "mean":
        loss = violation.mean()
    elif reduction == "sum":
        loss = violation.sum()
    elif reduction == "batch_sum_mean":
        loss = violation.sum(dim=(1, 2)).mean()
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'batch_sum_mean'")
    return loss, violation


def rollout_box_barrier_loss(
    x_traj: torch.Tensor,
    spec: SafetySpec,
    *,
    barrier_kind: str = "squared_hinge",
    eps: float = 1e-6,
    include_x0: bool = False,
    reduction: str = "batch_sum_mean",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Future-state-only box barrier loss.

    This uses the normal analytic barrier value on each box margin, but reduces over
    the same future-state horizon as the box ReLU constraint mode.
    """
    x = x_traj if include_x0 else x_traj[:, 1:, :]
    terms: Dict[str, torch.Tensor] = {}
    per_batch = x.new_zeros(x.shape[0])
    for con in spec.state_constraints:
        meta = getattr(con, "meta", {})
        if meta.get("kind") not in ("box_min", "box_max"):
            raise ValueError("box barrier loss requires only box_constraints")
        h = con.margin(x)
        bar = con.weight * barrier_value(
            h, delta=con.delta, band=con.band, kind=barrier_kind, eps=eps
        )
        terms[con.name] = bar
        per_batch = per_batch + bar.reshape(x.shape[0], -1).sum(dim=1)

    if reduction == "mean":
        loss = torch.stack([v for v in terms.values()]).mean()
    elif reduction == "sum":
        loss = per_batch.sum()
    elif reduction == "batch_sum_mean":
        loss = per_batch.mean()
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'batch_sum_mean'")
    return loss, terms


def rollout_barrier_loss(
    x_traj: torch.Tensor,
    spec: SafetySpec,
    *,
    kind: str = "squared_hinge",
    eps: float = 1e-6,
    u_traj: Optional[torch.Tensor] = None,
    u_prev: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Predictive barrier loss ``J_h`` accumulated over a trajectory (Eq. 18).

    Parameters
    ----------
    x_traj : (B, T, nx) predicted closed-loop states.
    spec   : the system's :class:`SafetySpec`.
    u_traj : (B, T, nu) predicted actions, required only if ``spec.control_rate`` is set.
    u_prev : (B, nu) previously executed action, to anchor the control-rate term.

    Returns
    -------
    (loss, terms, margins) where ``loss`` is a scalar, ``terms[name]`` is the per-constraint
    barrier tensor, and ``margins[name]`` is the per-constraint safety-margin tensor ``h``.
    ``loss == 0`` certifies predicted safety for all constraints (Eq. 20).
    """
    terms: Dict[str, torch.Tensor] = {}
    margins: Dict[str, torch.Tensor] = {}
    loss = x_traj.new_zeros(())

    for con in spec.state_constraints:
        h = con.margin(x_traj)
        bar = con.weight * barrier_value(
            h, delta=con.delta, band=con.band, kind=kind, eps=eps
        )
        terms[con.name] = bar
        margins[con.name] = h
        loss = loss + bar.sum()

    cr = spec.control_rate
    if cr is not None and u_traj is not None:
        du_sq = control_rate_sq_norm(u_traj, u_prev)
        h_du = cr.margin(du_sq)
        bar = cr.weight * barrier_value(
            h_du, delta=cr.delta, band=cr.band, kind=kind, eps=eps
        )
        terms[cr.name] = bar
        margins[cr.name] = h_du
        loss = loss + bar.sum()

    return loss, terms, margins
