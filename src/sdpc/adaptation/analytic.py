"""Analytic derivatives for ``SINDyVectorized`` models and policies.

This module is the no-autograd derivative backend used by the symbolic Jacobian
adaptation paths. It differentiates the compiled SINDy library layout directly and,
for continuous systems, propagates those vector-field Jacobians through the same RK4
step used by :class:`sdpc.systems.System`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

from ..sindy.model import SINDyVectorized

BoundType = Union[float, Sequence[float], torch.Tensor, None]

__all__ = [
    "BoundType",
    "to_bound",
    "dtheta_dx",
    "dtheta_dz",
    "sindy_vector_jacobians",
    "sindy_step_jacobians",
    "sindy_step",
    "policy_jacobians",
    "policy_du_da_blocks",
    "saturation_mask",
]


def to_bound(v: BoundType, n_u: int) -> Optional[torch.Tensor]:
    if v is None:
        return None
    t = v.detach().clone().float() if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32)
    if t.ndim == 0:
        t = t.expand(n_u)
    if t.numel() != n_u:
        raise ValueError(f"bound must be scalar or length {n_u}")
    return t


def _sel_len(sel) -> int:
    if sel is None:
        return 0
    return int(sel.shape[1]) if hasattr(sel, "dim") and sel.dim() == 2 else int(sel.numel())


def _zeros(B: int, m: int, n: int, x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(B, m, n, device=x.device, dtype=x.dtype)


def _poly_dtheta_dx(lib, plan, x: torch.Tensor, d: int, rows: torch.Tensor) -> torch.Tensor:
    rows = rows.to(device=x.device)
    combos = lib._poly_combos[d].to(device=x.device)[rows]
    B = x.shape[0]
    m = combos.shape[0]
    Xsel = x.index_select(1, combos.reshape(-1)).reshape(B, m, d)
    prefix = torch.cumprod(Xsel, dim=2)
    suffix = torch.cumprod(torch.flip(Xsel, dims=[2]), dim=2)
    suffix = torch.flip(suffix, dims=[2])
    D = _zeros(B, m, lib.n_features, x)
    ones = torch.ones(B, m, device=x.device, dtype=x.dtype)
    for s in range(d):
        left = prefix[:, :, s - 1] if s > 0 else ones
        right = suffix[:, :, s + 1] if s < d - 1 else ones
        idx = combos[:, s]
        D.scatter_add_(2, idx.view(1, m, 1).expand(B, m, 1), (left * right).unsqueeze(2))
    return D


@torch.no_grad()
def dtheta_dx(lib, plan, x: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
    """Derivative of ``Theta(x, z)`` w.r.t. state ``x``.

    Returns ``(B, M_plan, n_x)`` in exactly the same column order as
    ``lib.evaluate_plan(x, z, plan)``.
    """
    B, nx = x.shape
    blocks = []

    if plan.take_bias:
        blocks.append(_zeros(B, 1, nx, x))
    if plan.x_idx is not None:
        idx = plan.x_idx.to(device=x.device)
        m = idx.numel()
        D = _zeros(B, m, nx, x)
        D[:, torch.arange(m, device=x.device), idx] = 1.0
        blocks.append(D)
    if plan.u_idx is not None:
        blocks.append(_zeros(B, plan.u_idx.numel(), nx, x))
    if plan.sqrt_x_idx is not None:
        idx = plan.sqrt_x_idx.to(device=x.device)
        m = idx.numel()
        deriv = 0.5 / torch.sqrt(torch.clamp(x[:, idx], min=1e-12))
        D = _zeros(B, m, nx, x)
        D[:, torch.arange(m, device=x.device), idx] = deriv
        blocks.append(D)
    for d, rows in plan.poly_sel.items():
        blocks.append(_poly_dtheta_dx(lib, plan, x, d, rows))
    if plan.uu_pairs_sel is not None:
        blocks.append(_zeros(B, plan.uu_pairs_sel.shape[1], nx, x))
    if plan.xu_sel is not None:
        if z is None:
            raise ValueError("z is required for x*z library terms")
        sel = plan.xu_sel.to(device=x.device)
        i_idx, j_idx = sel[0], sel[1]
        m = i_idx.numel()
        D = _zeros(B, m, nx, x)
        D.scatter_add_(2, i_idx.view(1, m, 1).expand(B, m, 1), z[:, j_idx].unsqueeze(2))
        blocks.append(D)
    if plan.fourier_sin_sel is not None:
        sel = plan.fourier_sin_sel.to(device=x.device)
        i_idx, k_idx0 = sel[0], sel[1]
        m = i_idx.numel()
        freqs = (k_idx0 + 1).to(x.dtype).view(1, m)
        val = freqs * torch.cos(x[:, i_idx] * freqs)
        D = _zeros(B, m, nx, x)
        D.scatter_add_(2, i_idx.view(1, m, 1).expand(B, m, 1), val.unsqueeze(2))
        blocks.append(D)
    if plan.fourier_cos_sel is not None:
        sel = plan.fourier_cos_sel.to(device=x.device)
        i_idx, k_idx0 = sel[0], sel[1]
        m = i_idx.numel()
        freqs = (k_idx0 + 1).to(x.dtype).view(1, m)
        val = -freqs * torch.sin(x[:, i_idx] * freqs)
        D = _zeros(B, m, nx, x)
        D.scatter_add_(2, i_idx.view(1, m, 1).expand(B, m, 1), val.unsqueeze(2))
        blocks.append(D)
    if getattr(plan, "fourier_sin_r_sel", None) is not None:
        blocks.append(_zeros(B, _sel_len(plan.fourier_sin_r_sel), nx, x))
    if getattr(plan, "fourier_cos_r_sel", None) is not None:
        blocks.append(_zeros(B, _sel_len(plan.fourier_cos_r_sel), nx, x))

    return torch.cat(blocks, dim=1) if blocks else torch.empty(B, 0, nx, device=x.device, dtype=x.dtype)


@torch.no_grad()
def dtheta_dz(lib, plan, x: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
    """Derivative of ``Theta(x, z)`` w.r.t. the second input ``z``.

    ``z`` is control for dynamics libraries and reference for policy libraries.
    Returns ``(B, M_plan, n_z)``.
    """
    if z is None:
        nz = int(getattr(lib, "n_control", 0))
        return torch.empty(x.shape[0], 0, nz, device=x.device, dtype=x.dtype)

    B, nz = z.shape
    blocks = []
    if plan.take_bias:
        blocks.append(_zeros(B, 1, nz, x))
    if plan.x_idx is not None:
        blocks.append(_zeros(B, plan.x_idx.numel(), nz, x))
    if plan.u_idx is not None:
        idx = plan.u_idx.to(device=x.device)
        m = idx.numel()
        D = _zeros(B, m, nz, x)
        D[:, torch.arange(m, device=x.device), idx] = 1.0
        blocks.append(D)
    if plan.sqrt_x_idx is not None:
        blocks.append(_zeros(B, plan.sqrt_x_idx.numel(), nz, x))
    for _, rows in plan.poly_sel.items():
        blocks.append(_zeros(B, rows.numel(), nz, x))
    if plan.uu_pairs_sel is not None:
        pairs = plan.uu_pairs_sel.to(device=x.device)
        p, q = pairs[0], pairs[1]
        m = p.numel()
        D = _zeros(B, m, nz, x)
        D[:, torch.arange(m, device=x.device), p] += z[:, q]
        D[:, torch.arange(m, device=x.device), q] += z[:, p]
        blocks.append(D)
    if plan.xu_sel is not None:
        sel = plan.xu_sel.to(device=x.device)
        i_idx, j_idx = sel[0], sel[1]
        m = i_idx.numel()
        D = _zeros(B, m, nz, x)
        D[:, torch.arange(m, device=x.device), j_idx] = x[:, i_idx]
        blocks.append(D)
    if plan.fourier_sin_sel is not None:
        blocks.append(_zeros(B, plan.fourier_sin_sel.shape[1], nz, x))
    if plan.fourier_cos_sel is not None:
        blocks.append(_zeros(B, plan.fourier_cos_sel.shape[1], nz, x))
    if getattr(plan, "fourier_sin_r_sel", None) is not None:
        sel = plan.fourier_sin_r_sel.to(device=x.device)
        j_idx, k_idx0 = sel[0], sel[1]
        m = j_idx.numel()
        freqs = (k_idx0 + 1).to(x.dtype).view(1, m)
        val = freqs * torch.cos(z[:, j_idx] * freqs)
        D = _zeros(B, m, nz, x)
        D.scatter_add_(2, j_idx.view(1, m, 1).expand(B, m, 1), val.unsqueeze(2))
        blocks.append(D)
    if getattr(plan, "fourier_cos_r_sel", None) is not None:
        sel = plan.fourier_cos_r_sel.to(device=x.device)
        j_idx, k_idx0 = sel[0], sel[1]
        m = j_idx.numel()
        freqs = (k_idx0 + 1).to(x.dtype).view(1, m)
        val = -freqs * torch.sin(z[:, j_idx] * freqs)
        D = _zeros(B, m, nz, x)
        D.scatter_add_(2, j_idx.view(1, m, 1).expand(B, m, 1), val.unsqueeze(2))
        blocks.append(D)

    return torch.cat(blocks, dim=1) if blocks else torch.empty(B, 0, nz, device=x.device, dtype=x.dtype)


def _require_sindy(model) -> SINDyVectorized:
    if not isinstance(model, SINDyVectorized):
        raise TypeError("symbolic Jacobian backend requires SINDyVectorized models")
    return model


@torch.no_grad()
def sindy_vector_jacobians(model: SINDyVectorized, x: torch.Tensor, u: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Jacobians of the SINDy vector field/map ``model(x, u)``."""
    model = _require_sindy(model)
    plan = model._plan
    theta_x = dtheta_dx(model.library, plan, x, u)
    theta_u = dtheta_dz(model.library, plan, x, u)
    B = x.shape[0]
    nx = model.n_out
    nu = 0 if u is None else u.shape[1]
    dfdx = x.new_zeros((B, nx, x.shape[1]))
    dfdu = x.new_zeros((B, nx, nu))
    for i in range(nx):
        loc = model._state_local_idx[i].to(device=x.device)
        Xi_i = model.Xi[i].detach().squeeze(1).to(device=x.device, dtype=x.dtype)
        dfdx[:, i, :] = (theta_x.index_select(1, loc) * Xi_i.view(1, -1, 1)).sum(dim=1)
        if nu:
            dfdu[:, i, :] = (theta_u.index_select(1, loc) * Xi_i.view(1, -1, 1)).sum(dim=1)
    return dfdx, dfdu


def sindy_step(
    model: SINDyVectorized,
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    system=None,
    method: str = "rk4",
) -> torch.Tensor:
    model = _require_sindy(model)
    if system is None or getattr(system, "is_discrete", False):
        return model(x, u)
    ts = float(getattr(system, "ts", 1.0))
    if method == "euler":
        return x + ts * model.ode_equations(x, u)
    if method != "rk4":
        raise ValueError(f"unknown integration method {method!r}; expected 'rk4' or 'euler'")
    k1 = model.ode_equations(x, u)
    k2 = model.ode_equations(x + 0.5 * ts * k1, u)
    k3 = model.ode_equations(x + 0.5 * ts * k2, u)
    k4 = model.ode_equations(x + ts * k3, u)
    return x + (ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def sindy_step_jacobians(
    model: SINDyVectorized,
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    system=None,
    method: str = "rk4",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Jacobians of the discrete one-step map used during adaptation."""
    model = _require_sindy(model)
    if system is None or getattr(system, "is_discrete", False):
        return sindy_vector_jacobians(model, x, u)

    ts = float(getattr(system, "ts", 1.0))
    B, nx = x.shape
    nu = u.shape[1]
    I = torch.eye(nx, device=x.device, dtype=x.dtype).expand(B, nx, nx)
    if method == "euler":
        fx, fu = sindy_vector_jacobians(model, x, u)
        return I + ts * fx, ts * fu
    if method != "rk4":
        raise ValueError(f"unknown integration method {method!r}; expected 'rk4' or 'euler'")
    Z = x.new_zeros(B, nx, nu)

    k1 = model.ode_equations(x, u)
    g1x, g1u = sindy_vector_jacobians(model, x, u)
    dk1x, dk1u = g1x, g1u

    x2 = x + 0.5 * ts * k1
    x2x = I + 0.5 * ts * dk1x
    x2u = Z + 0.5 * ts * dk1u
    k2 = model.ode_equations(x2, u)
    g2x, g2u = sindy_vector_jacobians(model, x2, u)
    dk2x = torch.bmm(g2x, x2x)
    dk2u = torch.bmm(g2x, x2u) + g2u

    x3 = x + 0.5 * ts * k2
    x3x = I + 0.5 * ts * dk2x
    x3u = Z + 0.5 * ts * dk2u
    k3 = model.ode_equations(x3, u)
    g3x, g3u = sindy_vector_jacobians(model, x3, u)
    dk3x = torch.bmm(g3x, x3x)
    dk3u = torch.bmm(g3x, x3u) + g3u

    x4 = x + ts * k3
    x4x = I + ts * dk3x
    x4u = Z + ts * dk3u
    g4x, g4u = sindy_vector_jacobians(model, x4, u)
    dk4x = torch.bmm(g4x, x4x)
    dk4u = torch.bmm(g4x, x4u) + g4u

    dFdx = I + (ts / 6.0) * (dk1x + 2 * dk2x + 2 * dk3x + dk4x)
    dFdu = (ts / 6.0) * (dk1u + 2 * dk2u + 2 * dk3u + dk4u)
    return dFdx, dFdu


@torch.no_grad()
def saturation_mask(u_raw: torch.Tensor, umin: BoundType, umax: BoundType) -> torch.Tensor:
    n_u = u_raw.shape[1]
    lo = to_bound(umin, n_u)
    hi = to_bound(umax, n_u)
    mask = torch.ones_like(u_raw)
    if lo is not None:
        mask = mask * (u_raw > lo.to(device=u_raw.device, dtype=u_raw.dtype)).to(u_raw.dtype)
    if hi is not None:
        mask = mask * (u_raw < hi.to(device=u_raw.device, dtype=u_raw.dtype)).to(u_raw.dtype)
    return mask


@torch.no_grad()
def policy_jacobians(
    policy: SINDyVectorized,
    x: torch.Tensor,
    r: Optional[torch.Tensor],
    *,
    u_raw: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    umin: BoundType = None,
    umax: BoundType = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return clamped-policy ``du/dx`` and ``du/da`` as numeric tensors."""
    policy = _require_sindy(policy)
    if r is None:
        r = x.new_empty(x.shape[0], 0)
    if mask is None and u_raw is None:
        u_raw = policy(x, r)
    if mask is None:
        mask = saturation_mask(u_raw, umin, umax)
    else:
        mask = mask.to(device=x.device, dtype=x.dtype)

    theta = policy.library.evaluate_plan(x, r, policy._plan)
    theta_x = dtheta_dx(policy.library, policy._plan, x, r)
    B = x.shape[0]
    n_u = policy.n_out
    n_a = sum(int(p.numel()) for p in policy.Xi)
    dudx = x.new_zeros((B, n_u, x.shape[1]))
    duda = x.new_zeros((B, n_u, n_a))
    col = 0
    for k in range(n_u):
        loc = policy._state_local_idx[k].to(device=x.device)
        Xi_k = policy.Xi[k].detach().squeeze(1).to(device=x.device, dtype=x.dtype)
        dtheta_k = theta_x.index_select(1, loc)
        theta_k = theta.index_select(1, loc)
        dudx[:, k, :] = mask[:, k:k + 1] * (dtheta_k * Xi_k.view(1, -1, 1)).sum(dim=1)
        n_k = int(policy.Xi[k].numel())
        duda[:, k, col:col + n_k] = mask[:, k:k + 1] * theta_k
        col += n_k
    return dudx, duda


@torch.no_grad()
def policy_du_da_blocks(
    policy: SINDyVectorized,
    x: torch.Tensor,
    r: Optional[torch.Tensor],
    *,
    u_raw: Optional[torch.Tensor] = None,
    umin: BoundType = None,
    umax: BoundType = None,
) -> List[torch.Tensor]:
    _, duda = policy_jacobians(policy, x, r, u_raw=u_raw, umin=umin, umax=umax)
    out: List[torch.Tensor] = []
    col = 0
    for k, Xi_k in enumerate(policy.Xi):
        n_k = int(Xi_k.numel())
        out.append(duda[:, k, col:col + n_k])
        col += n_k
    return out
