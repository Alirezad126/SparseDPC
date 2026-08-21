"""Analytic closed-loop sensitivities for sparse SINDy policies.

The symbolic backend differentiates the interpretable SINDy libraries directly. It does
not call PyTorch's automatic differentiation machinery and it does not keep a rollout
computation graph.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ..actions import ACTION_GRADIENT_MODES, action_gradient_mask
from .analytic import (
    BoundType,
    policy_du_da_blocks,
    policy_jacobians,
    saturation_mask,
    sindy_step,
    sindy_step_jacobians,
    to_bound,
)

__all__ = ["SymbolicJacobian", "build_symbolic_jacobian", "compute_updates_symbolic"]


class SymbolicJacobian:
    """Analytic ``dF/dXi`` for ``x_next = F(x, clamp(pi_Xi(x, r)))``.

    Parameters
    ----------
    dyn_model:
        A :class:`sdpc.sindy.SINDyVectorized` dynamics model.
    policy:
        A sparse :class:`sdpc.sindy.SINDyVectorized` policy.
    system:
        Optional :class:`sdpc.systems.System`; when provided for continuous systems, the
        Jacobian is propagated through the same RK4 step used by ``system.discrete_step``.
    """

    def __init__(
        self,
        dyn_model,
        policy,
        umin: BoundType = None,
        umax: BoundType = None,
        *,
        system=None,
        action_scale: float = 1.0,
        integration_method: str = "rk4",
        couple_no_u_rows: bool = True,
    ):
        self.dyn_model = dyn_model
        self.policy = policy
        self.system = system
        self.n_u = int(policy.n_out)
        self.umin = to_bound(umin, self.n_u)
        self.umax = to_bound(umax, self.n_u)
        self.action_scale = float(action_scale)
        self.integration_method = integration_method
        self.couple_no_u_rows = couple_no_u_rows

    @torch.no_grad()
    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return sindy_step(
            self.dyn_model, x, u, system=self.system, method=self.integration_method
        )

    @torch.no_grad()
    def du_da(self, x: torch.Tensor, r: Optional[torch.Tensor]) -> List[torch.Tensor]:
        u_raw = self.policy(x, r if r is not None else x.new_empty(x.shape[0], 0))
        u_scaled = self.action_scale * u_raw
        return [
            self.action_scale * g
            for g in policy_du_da_blocks(
                self.policy, x, r, u_raw=u_scaled, umin=self.umin, umax=self.umax
            )
        ]

    @torch.no_grad()
    def du_dx(self, x: torch.Tensor, r: Optional[torch.Tensor], u_raw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u_raw is None:
            r_in = r if r is not None else x.new_empty(x.shape[0], 0)
            u_raw = self.action_scale * self.policy(x, r_in)
        dudx, _ = policy_jacobians(
            self.policy, x, r, u_raw=u_raw, umin=self.umin, umax=self.umax
        )
        return self.action_scale * dudx

    @torch.no_grad()
    def _dfdx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dfdx, _ = sindy_step_jacobians(
            self.dyn_model, x, u, system=self.system, method=self.integration_method
        )
        return dfdx

    @torch.no_grad()
    def _dfdu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        _, dfdu = sindy_step_jacobians(
            self.dyn_model, x, u, system=self.system, method=self.integration_method
        )
        return dfdu

    @torch.no_grad()
    def _step_jacobians(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return sindy_step_jacobians(
            self.dyn_model, x, u, system=self.system, method=self.integration_method
        )

    @torch.no_grad()
    def _clamped_action(self, x: torch.Tensor, r: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r_in = r if r is not None else x.new_empty(x.shape[0], 0)
        u_scaled = self.action_scale * self.policy(x, r_in)
        lo = None if self.umin is None else self.umin.to(device=u_scaled.device, dtype=u_scaled.dtype)
        hi = None if self.umax is None else self.umax.to(device=u_scaled.device, dtype=u_scaled.dtype)
        u = u_scaled if (lo is None and hi is None) else torch.clamp(u_scaled, lo, hi)
        return u_scaled, u

    @torch.no_grad()
    def closed_loop_jacobians(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        r: Optional[torch.Tensor],
        *,
        u_raw: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``A=dF_cl/dx``, ``B=dF_cl/da``, ``du/dx``, and ``du/da``."""
        if u_raw is None and mask is None:
            r_in = r if r is not None else x.new_empty(x.shape[0], 0)
            u_raw = self.action_scale * self.policy(x, r_in)
        dfdx, dfdu = self._step_jacobians(x, u)
        dudx, duda = policy_jacobians(
            self.policy, x, r, u_raw=u_raw, mask=mask, umin=self.umin, umax=self.umax
        )
        dudx = self.action_scale * dudx
        duda = self.action_scale * duda
        A = dfdx + torch.bmm(dfdu, dudx)
        B = torch.bmm(dfdu, duda)
        return A, B, dudx, duda

    @torch.no_grad()
    def closed_loop_AB(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        r: Optional[torch.Tensor],
        *,
        u_raw: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``A=dF_cl/dx``, ``B=dF_cl/da``, and ``du/da``."""
        A, B, _, duda = self.closed_loop_jacobians(x, u, r, u_raw=u_raw, mask=mask)
        return A, B, duda

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        u_raw, u = self._clamped_action(x, r)
        _, B, _ = self.closed_loop_AB(x, u, r, u_raw=u_raw)
        return B


def build_symbolic_jacobian(dyn_model, policy, umin=None, umax=None, **kwargs) -> SymbolicJacobian:
    return SymbolicJacobian(dyn_model, policy, umin=umin, umax=umax, **kwargs)


@torch.no_grad()
def compute_updates_symbolic(
    jac: SymbolicJacobian,
    x: torch.Tensor,
    ref_next: torch.Tensor,
    *,
    action_gradient_mode: str = "exact",
    action_gradient_band: float = 0.1,
    action_gradient_leak: float = 0.05,
) -> List[torch.Tensor]:
    """Reference-tracking update using analytic ``dF/dXi``.

    ``straight_through`` keeps the forward action clamped but evaluates ``dF/dXi``
    with an identity clamp derivative. This is useful for experiments where an action
    sitting on a bound would otherwise remove a policy-output row from the update.
    """
    if action_gradient_mode not in ACTION_GRADIENT_MODES:
        raise ValueError(
            f"action_gradient_mode must be one of {sorted(ACTION_GRADIENT_MODES)}"
        )
    policy = jac.policy
    r = ref_next
    u_raw, u = jac._clamped_action(x, r)
    x_next = jac.step(x, u)
    e = (ref_next - x_next).mean(0)

    mask = action_gradient_mask(
        u_raw, jac.umin, jac.umax,
        gradient_mode=action_gradient_mode,
        gradient_band=action_gradient_band,
        gradient_leak=action_gradient_leak,
    )
    D = jac.closed_loop_AB(x, u, r, u_raw=u_raw, mask=mask)[1].mean(0)
    denom = 1.0 + (D ** 2).sum()

    updates: List[torch.Tensor] = []
    col = 0
    for Xi_k in policy.Xi:
        n_k = int(Xi_k.numel())
        Dk = D[:, col:col + n_k]
        num_k = (Dk * e.unsqueeze(1)).sum(0)
        updates.append((num_k / denom).view_as(Xi_k).to(Xi_k))
        col += n_k
    return updates
