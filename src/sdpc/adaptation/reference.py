"""Unconstrained reference-tracking online adaptation (Sec. 3.4.1 of the paper).

The implemented law operates on the sampled closed-loop map and minimises the one-step
prediction residual ``e_k = r_k - F(x_k, u_k)``. The Levenberg-Marquardt-conditioned step

    Delta Xi = D_k^T e_k / (1 + ||D_k||_F^2),      Xi <- Xi + gamma * Delta Xi     (Eq. 12)

is a descent direction on ``J_ref = 1/2 ||e_k||^2`` whose length is bounded by the residual
(Proposition 1). ``D_k = dF/dXi`` is obtained here by autograd; the symbolic-Jacobian
backend in :mod:`sdpc.adaptation.jacobian` computes the same ``D_k`` analytically.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from ..actions import ACTION_GRADIENT_MODES, bounded_action

__all__ = [
    "compute_updates_discrete_ref",
    "apply_policy_updates",
    "adaptive_gamma_from_error",
    "backup_policy_Xi",
    "restore_policy_Xi",
    "zero_xi_grads",
    "grad_global_norm",
    "clip_grad_list_by_global_norm",
]


def compute_updates_discrete_ref(
    dyn_model,
    policy,
    x: torch.Tensor,
    ref_next: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    *,
    umin=None,
    umax=None,
    action_scale: float = 1.0,
    action_gradient_mode: str = "exact",
    action_gradient_band: float = 0.1,
    action_gradient_leak: float = 0.05,
) -> List[torch.Tensor]:
    """Reference-tracking coefficient update for a discrete map ``x_{k+1} = F(x_k, u_k)``.

    When ``u`` is omitted, this function constructs the hard-clamped policy action and
    uses ``action_gradient_mode`` for only its derivative. Thus every adaptation runner
    shares identical reference-gradient semantics while applied actions remain bounded.

    A supplied ``u`` must already carry the desired policy graph; it is retained for
    backward compatibility. Returns per-output updates aligned with ``policy.Xi``:
    ``Delta Xi_k = (sum_i e_i dF_i/dXi_k) / (1 + ||D||_F^2)``.

    """
    if action_gradient_mode not in ACTION_GRADIENT_MODES:
        raise ValueError(
            f"action_gradient_mode must be one of {sorted(ACTION_GRADIENT_MODES)}, "
            f"got {action_gradient_mode!r}"
        )
    if u is None:
        u = bounded_action(
            policy(x, ref_next), umin, umax, action_scale,
            gradient_mode=action_gradient_mode,
            gradient_band=action_gradient_band,
            gradient_leak=action_gradient_leak,
        )
    params = tuple(policy.Xi)
    for p in params:
        assert p.requires_grad, "policy.Xi must require grad (do not detach u)"

    x_next = dyn_model(x, u).squeeze(0)          # (nx,)
    nx = x_next.numel()
    ref_next = ref_next.squeeze(0)               # (nx,)
    e_next = ref_next - x_next                    # (nx,)

    g_rows = []
    for i in range(nx):
        gi = torch.autograd.grad(
            x_next[i], params, retain_graph=True, create_graph=False, allow_unused=False
        )
        g_rows.append(gi)

    fro2 = x_next.new_zeros(())
    for i in range(nx):
        for g in g_rows[i]:
            fro2 = fro2 + (g ** 2).sum()
    denom = 1.0 + fro2

    updates = []
    for k in range(len(params)):
        num_k = sum(e_next[i] * g_rows[i][k] for i in range(nx))
        updates.append(num_k / denom)
    return updates


def apply_policy_updates(
    policy,
    updates: Sequence[torch.Tensor],
    gamma: float = 1.0,
    clip: Optional[float] = None,
) -> None:
    """In-place ``Xi_k <- Xi_k + gamma * clip(updates_k)`` with NaN/Inf guarding."""
    assert len(updates) == len(policy.Xi), "updates and Xi must align"
    with torch.no_grad():
        for Xi_k, dk in zip(policy.Xi, updates):
            dk = dk.to(dtype=Xi_k.dtype, device=Xi_k.device)
            dk = torch.nan_to_num(dk, nan=0.0, posinf=0.0, neginf=0.0)
            if clip is not None:
                dk = dk.clamp(min=-clip, max=clip)
            Xi_k.add_(gamma * dk)


def adaptive_gamma_from_error(
    x: torch.Tensor,
    r: torch.Tensor,
    *,
    gamma_min: float,
    gamma_max: float,
    err_scale: float = 0.05,
) -> float:
    """Larger step when the tracking error is small: ``gamma`` interpolated by ``exp(-|e|/s)``."""
    with torch.no_grad():
        e = r - x
        e_norm = torch.linalg.norm(e) if e.dim() == 1 else torch.linalg.norm(e, dim=-1).mean()
        gain = torch.exp(-e_norm / err_scale)
        gamma = gamma_min + (gamma_max - gamma_min) * gain
        gamma = torch.clamp(gamma, min=gamma_min, max=gamma_max)
    return float(gamma.detach().cpu().item())


# --------------------------------------------------------------------------- #
# Coefficient bookkeeping helpers (used by the safe-adaptation inner loop)
# --------------------------------------------------------------------------- #
def backup_policy_Xi(policy) -> List[torch.Tensor]:
    return [p.detach().clone() for p in policy.Xi]


def restore_policy_Xi(policy, backup: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, p_old in zip(policy.Xi, backup):
            p.copy_(p_old)


def zero_xi_grads(policy) -> None:
    for p in policy.Xi:
        p.grad = None


def grad_global_norm(grad_list: Sequence[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    total = None
    for g in grad_list:
        val = (g.detach() ** 2).sum()
        total = val if total is None else total + val
    if total is None:
        return torch.zeros(())
    return torch.sqrt(total + eps)


def clip_grad_list_by_global_norm(grad_list, *, max_norm: float, eps: float = 1e-12):
    """Scale a gradient list down (never up) so its global norm is at most ``max_norm``."""
    norm = grad_global_norm(grad_list, eps=eps)
    scale = torch.clamp(max_norm / (norm + eps), max=1.0)
    clipped = [scale * g for g in grad_list]
    return clipped, float(norm.detach().cpu().item()), float(scale.detach().cpu().item())
