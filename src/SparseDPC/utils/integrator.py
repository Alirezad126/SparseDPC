import torch
import torch.nn as nn


class OneElementEulerIntegrator(nn.Module):
    """
    One-step forward Euler integrator for a single scalar output model fx.

    This integrates a single element `xi` of the state using its associated
    dynamics `dx_i = fx(X, u)`.

    Parameters
    ----------
    fx : nn.Module
        A model that computes dx_i = fx(X, u) with shape (B, 1), where:
            - X : (B, nx) is the full state batch
            - u : (B, nu) is the control input batch
    h : float or torch.Tensor
        Integration time step (scalar)
    """
    def __init__(self, fx: nn.Module, h: float | torch.Tensor):
        super().__init__()
        self.fx = fx
        self.h = h

    def forward(
        self,
        X: torch.Tensor,   # (B, nx)
        xi: torch.Tensor,  # (B, 1)
        u: torch.Tensor    # (B, nu)
    ) -> torch.Tensor:
        """
        Perform one Euler step for a scalar state xi.

        Returns
        -------
        xi_next : torch.Tensor
            Next state value after one integration step (B, 1)
        """
        dx_i = self.fx(X, u)  # (B, 1)
        return xi + self.h * dx_i


class FullStateEulerIntegrator(nn.Module):
    """
    One-step forward Euler integrator for full state x using per-dimension models.

    Parameters
    ----------
    fx_list : list[nn.Module]
        List of fx_i(x, u) → dx_i, one per state component (len = nx).
    h : float or torch.Tensor
        Integration time step (scalar)
    """
    def __init__(self, fx_list: list[nn.Module], h: float | torch.Tensor):
        super().__init__()
        self.fx_list = fx_list
        self.h = h

    def forward(
        self,
        x: torch.Tensor,  # (B, nx)
        u: torch.Tensor   # (B, nu)
    ) -> torch.Tensor:
        """
        Perform one Euler integration step over all state dimensions.

        Returns
        -------
        x_next : torch.Tensor
            Next state vector (B, nx)
        """
        dx = torch.zeros_like(x)
        for idx, fx in enumerate(self.fx_list):
            dx[:, idx:idx + 1] = fx(x, u)
        return x + self.h * dx


class FullStateRK4Integrator(nn.Module):
    """
    One-step 4th-order Runge–Kutta integrator for full state vector.

    Assumes one model per state dimension.

    Parameters
    ----------
    fx_list : list[nn.Module]
        List of fx_i(x, u) → dx_i (output shape: B×1), one per state dimension (len = nx).
    h : float or torch.Tensor
        Integration time step (scalar)
    """
    def __init__(self, fx_list: list[nn.Module], h: float | torch.Tensor):
        super().__init__()
        self.fx_list = nn.ModuleList(fx_list)
        self.h = torch.as_tensor(h)

    def _f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all f_i(x, u) for the full state vector.

        Returns
        -------
        dx : torch.Tensor
            Time derivative vector dx/dt, shape (B, nx)
        """
        dx = torch.empty_like(x)
        for i, fx in enumerate(self.fx_list):
            dx[:, i:i+1] = fx(x, u)
        return dx

    def forward(
        self,
        x: torch.Tensor,  # (B, nx)
        u: torch.Tensor   # (B, nu)
    ) -> torch.Tensor:
        """
        Perform one Runge–Kutta step over the entire state vector.

        Returns
        -------
        x_next : torch.Tensor
            Next state vector (B, nx)
        """
        h = self.h
        k1 = self._f(x, u)
        k2 = self._f(x + h * k1 / 2.0, u)
        k3 = self._f(x + h * k2 / 2.0, u)
        k4 = self._f(x + h * k3, u)
        return x + h * (k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0)
