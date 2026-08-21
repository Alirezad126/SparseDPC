"""Analytic safety-gradient backend for barrier-based safe adaptation.

The backend propagates the rollout sensitivity
``S_k = dx_k / dXi`` forward through the closed loop and accumulates the barrier
gradient by chain rule. All Jacobians are evaluated from the SINDy library structure;
no PyTorch derivative graph is built.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ..actions import ACTION_GRADIENT_MODES, action_gradient_mask
from ..safety.specs import SafetySpec
from .barrier import barrier_derivative, barrier_value
from .jacobian import SymbolicJacobian

__all__ = ["SymbolicSafetyJacobian"]


BOX_RELU_CONSTRAINT_KINDS = {"box_relu_constraints", "box_relu_hinge"}
BOX_BARRIER_CONSTRAINT_KINDS = {
    "box_barrier_constraints": "squared_hinge",
    "box_barrier_hinge": "squared_hinge",
    "box_relaxed_log_constraints": "relaxed_log",
}
def _validate_action_gradient_mode(mode: str) -> None:
    if mode not in ACTION_GRADIENT_MODES:
        raise ValueError(
            f"action_gradient_mode must be one of {sorted(ACTION_GRADIENT_MODES)}"
        )


def _mask_from_applied_action(
    jac: "SymbolicSafetyJacobian",
    u_applied: torch.Tensor,
    mode: str,
    *,
    u_raw: Optional[torch.Tensor] = None,
    gradient_band: float = 0.1,
    gradient_leak: float = 0.05,
) -> torch.Tensor:
    """Policy-action derivative mask for an already clamped rollout action."""
    _validate_action_gradient_mode(mode)
    source = u_raw if mode == "leaky_straight_through" else u_applied
    if source is None:
        raise ValueError("leaky straight-through mask requires the scaled raw action")
    return action_gradient_mask(
        source, jac.umin, jac.umax,
        gradient_mode=mode,
        gradient_band=gradient_band,
        gradient_leak=gradient_leak,
    )


@torch.no_grad()
def _barrier_state_grad(
    x_next: torch.Tensor,
    spec: SafetySpec,
    *,
    kind: str,
    eps: float,
) -> torch.Tensor:
    """Analytic ``d state_barrier / dx`` for all state constraints."""
    grad = torch.zeros_like(x_next)
    for con in spec.state_constraints:
        h = con.margin(x_next)
        dB_dh = con.weight * barrier_derivative(
            h, delta=con.delta, band=con.band, kind=kind, eps=eps
        )
        grad = grad + dB_dh.unsqueeze(-1) * con.grad(x_next)
    return grad


@torch.no_grad()
def _barrier_state_grad_loss(
    x_next: torch.Tensor,
    spec: SafetySpec,
    *,
    kind: str,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic ``d state_barrier / dx`` and scalar state-barrier loss."""
    grad = torch.zeros_like(x_next)
    loss = x_next.new_zeros(())
    for con in spec.state_constraints:
        h = con.margin(x_next)
        bar = con.weight * barrier_value(
            h, delta=con.delta, band=con.band, kind=kind, eps=eps
        )
        loss = loss + bar.sum()
        dB_dh = con.weight * barrier_derivative(
            h, delta=con.delta, band=con.band, kind=kind, eps=eps
        )
        grad = grad + dB_dh.unsqueeze(-1) * con.grad(x_next)
    return grad, loss


@torch.no_grad()
def _box_state_grad_loss_fast(
    x_next: torch.Tensor,
    spec: SafetySpec,
    *,
    kind: str,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fast path for ``box_constraints`` names: ``x{i}_min`` / ``x{i}_max``."""
    if not spec.state_constraints:
        return x_next.new_zeros(x_next.shape), x_next.new_zeros(())
    parsed = []
    for con in spec.state_constraints:
        meta = getattr(con, "meta", {})
        if meta.get("kind") not in ("box_min", "box_max"):
            return None
        idx = int(meta["idx"])
        if idx >= x_next.shape[-1]:
            return None
        sign = 1.0 if meta["kind"] == "box_min" else -1.0
        parsed.append((con, idx, sign, float(meta["bound"])))

    grad = torch.zeros_like(x_next)
    loss = x_next.new_zeros(())
    for con, idx, sign, bound in parsed:
        h = x_next[:, idx] - bound if sign > 0 else bound - x_next[:, idx]
        loss = loss + (
            con.weight * barrier_value(
                h, delta=con.delta, band=con.band, kind=kind, eps=eps
            )
        ).sum()
        dB_dh = con.weight * barrier_derivative(
            h, delta=con.delta, band=con.band, kind=kind, eps=eps
        )
        grad[:, idx] = grad[:, idx] + sign * dB_dh
    return grad, loss


def _box_bounds_from_spec(spec: SafetySpec, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    nx = x.shape[-1]
    xmin = torch.full((nx,), -torch.inf, device=x.device, dtype=x.dtype)
    xmax = torch.full((nx,), torch.inf, device=x.device, dtype=x.dtype)
    for con in spec.state_constraints:
        meta = getattr(con, "meta", {})
        if meta.get("kind") == "box_min":
            xmin[int(meta["idx"])] = float(meta["bound"])
        elif meta.get("kind") == "box_max":
            xmax[int(meta["idx"])] = float(meta["bound"])
        else:
            raise ValueError("box ReLU safety gradient requires box_constraints")
    if bool(torch.isinf(xmin).any().item() or torch.isinf(xmax).any().item()):
        raise ValueError("box ReLU safety gradient needs min and max bounds")
    return xmin, xmax


class SymbolicSafetyJacobian(SymbolicJacobian):
    """Forward-sensitivity safety gradient for sparse policies and SINDy dynamics.

    Straight-through mode keeps the analytic forward map hard-clamped but replaces its
    saturation mask by ones, matching the corresponding autograd surrogate derivative.
    """

    @torch.no_grad()
    def rollout_and_sensitivity_grad(
        self,
        x0: torch.Tensor,
        r_traj: torch.Tensor,
        spec: SafetySpec,
        *,
        kind: str = "squared_hinge",
        eps: float = 1e-6,
        u_prev: Optional[torch.Tensor] = None,
        return_traj: bool = False,
        action_gradient_mode: str = "exact",
        action_gradient_band: float = 0.1,
        action_gradient_leak: float = 0.05,
    ):
        """Fused optimized rollout + sensitivity-gradient pass.

        This is the fast path for timing and for Euler-style symbolic safety updates:
        the closed-loop rollout, local Jacobians, sensitivity recursion, barrier loss,
        and barrier gradient are all computed in one horizon loop. It avoids the older
        two-pass pattern of first storing a no-grad rollout and then revisiting every
        horizon point to compute Jacobians.
        """
        _validate_action_gradient_mode(action_gradient_mode)
        if r_traj.dim() != 3:
            raise ValueError(f"r_traj must be (B, N, nr), got {tuple(r_traj.shape)}")

        B, N, _ = r_traj.shape
        nx = x0.shape[1]
        n_a = sum(int(p.numel()) for p in self.policy.Xi)
        device, dtype = x0.device, x0.dtype

        x = x0
        Sx = torch.zeros(B, nx, n_a, device=device, dtype=dtype)
        grad_a = torch.zeros(B, n_a, device=device, dtype=dtype)
        loss = x0.new_zeros(())

        Su_prev = None
        u_prev_step = u_prev
        xs = [x0.unsqueeze(1)] if return_traj else None
        us = [] if return_traj else None

        for k in range(N):
            r_k = r_traj[:, k, :]
            u_raw, u_k = self._clamped_action(x, r_k)
            mask_k = _mask_from_applied_action(
                self, u_k, action_gradient_mode, u_raw=u_raw,
                gradient_band=action_gradient_band,
                gradient_leak=action_gradient_leak,
            )
            A_k, B_k, dudx_k, duda_k = self.closed_loop_jacobians(
                x, u_k, r_k, u_raw=u_raw, mask=mask_k
            )
            x_next = self.step(x, u_k)
            Sx_next = torch.bmm(A_k, Sx) + B_k

            for con in spec.state_constraints:
                h = con.margin(x_next)
                bar = con.weight * barrier_value(
                    h, delta=con.delta, band=con.band, kind=kind, eps=eps
                )
                loss = loss + bar.sum()
                dB_dh = con.weight * barrier_derivative(
                    h, delta=con.delta, band=con.band, kind=kind, eps=eps
                )
                dB_dx = dB_dh.unsqueeze(-1) * con.grad(x_next)
                grad_a = grad_a + torch.einsum("bi,bij->bj", dB_dx, Sx_next)

            cr = spec.control_rate
            if cr is not None:
                if k == 0 and u_prev_step is None:
                    du = torch.zeros_like(u_k)
                    active_rate = False
                    Sdu = None
                else:
                    du = u_k - (u_prev_step if k == 0 else u_prev_step)
                    h_du = cr.margin((du ** 2).sum(dim=-1))
                    bar_du = cr.weight * barrier_value(
                        h_du, delta=cr.delta, band=cr.band, kind=kind, eps=eps
                    )
                    loss = loss + bar_du.sum()
                    dB_dh = cr.weight * barrier_derivative(
                        h_du, delta=cr.delta, band=cr.band, kind=kind, eps=eps
                    )
                    active_rate = bool((dB_dh != 0).any().item())
                    if active_rate:
                        Su = torch.bmm(dudx_k, Sx) + duda_k
                        Sdu = Su if k == 0 else Su - Su_prev
                        dB_ddu = dB_dh.unsqueeze(-1) * (-2.0 * du)
                        grad_a = grad_a + torch.einsum("bi,bij->bj", dB_ddu, Sdu)
                        Su_prev = Su
                if not active_rate:
                    Su_prev = torch.bmm(dudx_k, Sx) + duda_k if cr is not None else None
                u_prev_step = u_k
            elif u_prev_step is not None:
                u_prev_step = u_k

            if return_traj:
                xs.append(x_next.unsqueeze(1))
                us.append(u_k.unsqueeze(1))
            x = x_next
            Sx = Sx_next

        grad_mean = grad_a.mean(0)
        grads: List[torch.Tensor] = []
        col = 0
        for Xi_k in self.policy.Xi:
            n_k = int(Xi_k.numel())
            grads.append((-grad_mean[col:col + n_k]).view_as(Xi_k).to(Xi_k))
            col += n_k

        loss_float = float(loss.detach().cpu().item())
        if not return_traj:
            return grads, loss_float
        return grads, loss_float, torch.cat(xs, dim=1), torch.cat(us, dim=1)

    @torch.no_grad()
    def rollout_sensitivity_grad(
        self,
        x_traj: torch.Tensor,
        u_traj: torch.Tensor,
        r_traj: torch.Tensor,
        spec: SafetySpec,
        *,
        kind: str = "squared_hinge",
        eps: float = 1e-6,
        u_prev: Optional[torch.Tensor] = None,
        action_gradient_mode: str = "exact",
        action_gradient_band: float = 0.1,
        action_gradient_leak: float = 0.05,
    ) -> Tuple[List[torch.Tensor], float]:
        """Return a descent direction ``-dJ_h/dXi`` and the rollout barrier loss."""
        if kind in BOX_RELU_CONSTRAINT_KINDS:
            return self.rollout_box_relu_sensitivity_grad(
                x_traj, u_traj, r_traj, spec, u_prev=u_prev,
                action_gradient_mode=action_gradient_mode,
                action_gradient_band=action_gradient_band,
                action_gradient_leak=action_gradient_leak,
            )
        if kind in BOX_BARRIER_CONSTRAINT_KINDS:
            return self.rollout_box_barrier_sensitivity_grad(
                x_traj, u_traj, r_traj, spec,
                barrier_kind=BOX_BARRIER_CONSTRAINT_KINDS[kind],
                eps=eps,
                u_prev=u_prev,
                normalize=False,
                action_gradient_mode=action_gradient_mode,
                action_gradient_band=action_gradient_band,
                action_gradient_leak=action_gradient_leak,
            )

        _validate_action_gradient_mode(action_gradient_mode)

        B, _, nx = x_traj.shape
        N = u_traj.shape[1]
        device, dtype = x_traj.device, x_traj.dtype
        n_a = sum(int(p.numel()) for p in self.policy.Xi)

        x_flat = x_traj[:, :N, :].reshape(B * N, nx)
        u_flat = u_traj[:, :N, :].reshape(B * N, u_traj.shape[-1])
        if r_traj.dim() == 3:
            r_flat = r_traj[:, :N, :].reshape(B * N, r_traj.shape[-1])
        else:
            r_flat = r_traj.unsqueeze(1).expand(B, N, r_traj.shape[-1]).reshape(B * N, r_traj.shape[-1])
        u_raw_flat = None
        if action_gradient_mode == "leaky_straight_through":
            u_raw_flat = self.action_scale * self.policy(x_flat, r_flat)
        mask_flat = _mask_from_applied_action(
            self, u_flat, action_gradient_mode, u_raw=u_raw_flat,
            gradient_band=action_gradient_band,
            gradient_leak=action_gradient_leak,
        )
        A_flat, B_flat, dudx_flat, duda_flat = self.closed_loop_jacobians(
            x_flat, u_flat, r_flat, mask=mask_flat
        )
        A_all = A_flat.reshape(B, N, nx, nx).transpose(0, 1).contiguous()
        B_all = B_flat.reshape(B, N, nx, n_a).transpose(0, 1).contiguous()
        dudx_all = dudx_flat.reshape(B, N, self.n_u, nx).transpose(0, 1).contiguous()
        duda_all = duda_flat.reshape(B, N, self.n_u, n_a).transpose(0, 1).contiguous()
        x_future_flat = x_traj[:, 1:N + 1, :].reshape(B * N, nx)
        fast_state = _box_state_grad_loss_fast(x_future_flat, spec, kind=kind, eps=eps)
        if fast_state is None:
            dBdx_flat, loss_val = _barrier_state_grad_loss(
                x_future_flat, spec, kind=kind, eps=eps
            )
        else:
            dBdx_flat, loss_val = fast_state
        dBdx_all = dBdx_flat.reshape(B, N, nx).transpose(0, 1).contiguous()

        cr = spec.control_rate
        rate_active = False
        if cr is not None:
            if u_prev is not None:
                u_aug = torch.cat([u_prev.unsqueeze(1), u_traj], dim=1)
            else:
                u_aug = torch.cat([u_traj[:, :1, :], u_traj], dim=1)
            du_all = u_aug[:, 1:, :] - u_aug[:, :-1, :]
            h_du_all = cr.margin((du_all ** 2).sum(dim=-1))
            loss_val = loss_val + (
                cr.weight * barrier_value(
                    h_du_all, delta=cr.delta, band=cr.band, kind=kind, eps=eps
                )
            ).sum()
            rate_active = bool(
                (barrier_derivative(
                    h_du_all, delta=cr.delta, band=cr.band, kind=kind, eps=eps
                ) != 0).any().item()
            )

        Sx = torch.zeros(B, nx, n_a, device=device, dtype=dtype)
        grad_a = torch.zeros(B, n_a, device=device, dtype=dtype)
        Su_prev = None

        for k in range(N):
            u_k = u_traj[:, k, :]

            Su = torch.bmm(dudx_all[k], Sx) + duda_all[k] if rate_active else None
            Sx = torch.bmm(A_all[k], Sx) + B_all[k]

            grad_a = grad_a + torch.einsum("bi,bij->bj", dBdx_all[k], Sx)

            if rate_active:
                if k == 0:
                    if u_prev is None:
                        du = torch.zeros_like(u_k)
                        Sdu = torch.zeros_like(duda_all[k])
                    else:
                        du = u_k - u_prev
                        Sdu = Su
                else:
                    du = u_k - u_traj[:, k - 1, :]
                    Sdu = Su - Su_prev
                du_sq = (du ** 2).sum(dim=-1)
                h_du = cr.margin(du_sq)
                dB_dh = cr.weight * barrier_derivative(
                    h_du, delta=cr.delta, band=cr.band, kind=kind, eps=eps
                )
                dB_ddu = dB_dh.unsqueeze(-1) * (-2.0 * du)
                grad_a = grad_a + torch.einsum("bi,bij->bj", dB_ddu, Sdu)
            Su_prev = Su

        grad_mean = grad_a.mean(0)
        grads: List[torch.Tensor] = []
        col = 0
        for Xi_k in self.policy.Xi:
            n_k = int(Xi_k.numel())
            grads.append((-grad_mean[col:col + n_k]).view_as(Xi_k).to(Xi_k))
            col += n_k
        return grads, float(loss_val.detach().cpu().item())

    @torch.no_grad()
    def rollout_box_relu_sensitivity_grad(
        self,
        x_traj: torch.Tensor,
        u_traj: torch.Tensor,
        r_traj: torch.Tensor,
        spec: SafetySpec,
        *,
        u_prev: Optional[torch.Tensor] = None,
        include_x0: bool = False,
        normalize: bool = True,
        action_gradient_mode: str = "exact",
        action_gradient_band: float = 0.1,
        action_gradient_leak: float = 0.05,
    ) -> Tuple[List[torch.Tensor], float]:
        """Box ReLU constraint gradient from an existing rollout.

        This matches ``constraint_grads_box_via_S`` from the TwoTank reference notebook:
        future-state box signs, forward sensitivity recursion, and per-batch
        ``1 + ||S||^2`` normalization. It works for both Euler and RK4 because
        ``closed_loop_jacobians`` returns the selected discrete step tangent map.
        """
        del u_prev  # box-constraint mode does not include control-rate constraints
        B, _, nx = x_traj.shape
        N = u_traj.shape[1]
        device, dtype = x_traj.device, x_traj.dtype
        n_a = sum(int(p.numel()) for p in self.policy.Xi)

        x_flat = x_traj[:, :N, :].reshape(B * N, nx)
        u_flat = u_traj[:, :N, :].reshape(B * N, u_traj.shape[-1])
        if r_traj.dim() == 3:
            r_flat = r_traj[:, :N, :].reshape(B * N, r_traj.shape[-1])
        else:
            r_flat = r_traj.unsqueeze(1).expand(B, N, r_traj.shape[-1]).reshape(B * N, r_traj.shape[-1])

        u_raw_flat = None
        if action_gradient_mode == "leaky_straight_through":
            u_raw_flat = self.action_scale * self.policy(x_flat, r_flat)
        mask_flat = _mask_from_applied_action(
            self, u_flat, action_gradient_mode, u_raw=u_raw_flat,
            gradient_band=action_gradient_band,
            gradient_leak=action_gradient_leak,
        )

        A_flat, B_flat, _, _ = self.closed_loop_jacobians(
            x_flat, u_flat, r_flat, mask=mask_flat
        )
        A_all = A_flat.reshape(B, N, nx, nx).transpose(0, 1).contiguous()
        B_all = B_flat.reshape(B, N, nx, n_a).transpose(0, 1).contiguous()

        x_future = x_traj[:, 1:N + 1, :]
        xmin, xmax = _box_bounds_from_spec(spec, x_future)
        xmin_view = xmin.view(1, 1, nx)
        xmax_view = xmax.view(1, 1, nx)
        sign_all = (
            (x_future > xmax_view).to(dtype) - (x_future < xmin_view).to(dtype)
        ).transpose(0, 1).contiguous()
        violation = torch.relu(x_future - xmax_view) + torch.relu(xmin_view - x_future)
        loss_val = violation.sum(dim=(1, 2)).mean()

        Sx = torch.zeros(B, nx, n_a, device=device, dtype=dtype)
        grad_a = torch.zeros(B, n_a, device=device, dtype=dtype)
        denom_acc = torch.zeros(B, device=device, dtype=dtype)

        k_start = 0 if include_x0 else 1
        for k in range(N):
            Sx = torch.bmm(A_all[k], Sx) + B_all[k]
            if k + 1 >= k_start:
                grad_a = grad_a + torch.einsum("bi,bij->bj", sign_all[k], Sx)
                denom_acc = denom_acc + (Sx * Sx).sum(dim=(1, 2))

        if normalize:
            grad_a = grad_a / (1.0 + denom_acc).unsqueeze(1)
        grad_mean = grad_a.mean(0)
        grads: List[torch.Tensor] = []
        col = 0
        for Xi_k in self.policy.Xi:
            n_k = int(Xi_k.numel())
            grads.append((-grad_mean[col:col + n_k]).view_as(Xi_k).to(Xi_k))
            col += n_k
        return grads, float(loss_val.detach().cpu().item())

    @torch.no_grad()
    def rollout_box_barrier_sensitivity_grad(
        self,
        x_traj: torch.Tensor,
        u_traj: torch.Tensor,
        r_traj: torch.Tensor,
        spec: SafetySpec,
        *,
        barrier_kind: str = "squared_hinge",
        eps: float = 1e-6,
        u_prev: Optional[torch.Tensor] = None,
        include_x0: bool = False,
        normalize: bool = False,
        action_gradient_mode: str = "exact",
        action_gradient_band: float = 0.1,
        action_gradient_leak: float = 0.05,
    ) -> Tuple[List[torch.Tensor], float]:
        """Box barrier constraint gradient with the reference-notebook recursion."""
        del u_prev
        B, _, nx = x_traj.shape
        N = u_traj.shape[1]
        device, dtype = x_traj.device, x_traj.dtype
        n_a = sum(int(p.numel()) for p in self.policy.Xi)

        x_flat = x_traj[:, :N, :].reshape(B * N, nx)
        u_flat = u_traj[:, :N, :].reshape(B * N, u_traj.shape[-1])
        if r_traj.dim() == 3:
            r_flat = r_traj[:, :N, :].reshape(B * N, r_traj.shape[-1])
        else:
            r_flat = r_traj.unsqueeze(1).expand(B, N, r_traj.shape[-1]).reshape(B * N, r_traj.shape[-1])

        u_raw_flat = None
        if action_gradient_mode == "leaky_straight_through":
            u_raw_flat = self.action_scale * self.policy(x_flat, r_flat)
        mask_flat = _mask_from_applied_action(
            self, u_flat, action_gradient_mode, u_raw=u_raw_flat,
            gradient_band=action_gradient_band,
            gradient_leak=action_gradient_leak,
        )

        A_flat, B_flat, _, _ = self.closed_loop_jacobians(
            x_flat, u_flat, r_flat, mask=mask_flat
        )
        A_all = A_flat.reshape(B, N, nx, nx).transpose(0, 1).contiguous()
        B_all = B_flat.reshape(B, N, nx, n_a).transpose(0, 1).contiguous()

        x_future_flat = x_traj[:, 1:N + 1, :].reshape(B * N, nx)
        fast_state = _box_state_grad_loss_fast(
            x_future_flat, spec, kind=barrier_kind, eps=eps
        )
        if fast_state is None:
            raise ValueError("box barrier gradient requires box_constraints")
        dBdx_flat, loss_sum = fast_state
        dBdx_all = dBdx_flat.reshape(B, N, nx).transpose(0, 1).contiguous()
        loss_val = loss_sum / B

        Sx = torch.zeros(B, nx, n_a, device=device, dtype=dtype)
        grad_a = torch.zeros(B, n_a, device=device, dtype=dtype)
        denom_acc = torch.zeros(B, device=device, dtype=dtype)

        k_start = 0 if include_x0 else 1
        for k in range(N):
            Sx = torch.bmm(A_all[k], Sx) + B_all[k]
            if k + 1 >= k_start:
                grad_a = grad_a + torch.einsum("bi,bij->bj", dBdx_all[k], Sx)
                denom_acc = denom_acc + (Sx * Sx).sum(dim=(1, 2))

        if normalize:
            grad_a = grad_a / (1.0 + denom_acc).unsqueeze(1)
        grad_mean = grad_a.mean(0)
        grads: List[torch.Tensor] = []
        col = 0
        for Xi_k in self.policy.Xi:
            n_k = int(Xi_k.numel())
            grads.append((-grad_mean[col:col + n_k]).view_as(Xi_k).to(Xi_k))
            col += n_k
        return grads, float(loss_val.detach().cpu().item())
