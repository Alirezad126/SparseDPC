import torch

class OneElementEulerIntegrator(torch.nn.Module):
    def __init__(self, fx, h):
        super().__init__()
        self.fx = fx  # fx must be nn.Module with registered parameters
        self.h = h

    def forward(self, X, xi, u):
        dx_i = self.fx(X, u)  # shape: (batch, 1)
        return xi + self.h * dx_i


class FullStateEulerIntegrator(torch.nn.Module):
    def __init__(self, fx_list, h):
        super().__init__()
        self.fx_list = fx_list  # fx must be nn.Module with registered parameters
        self.h = h
    def forward(self, x, u):
        dx = torch.zeros_like(x)
        for idx, fx in enumerate(self.fx_list):
            dx[:, idx:idx + 1] = fx(x, u)
        return x + self.h * dx



import torch.nn as nn


class FullStateRK4Integrator(nn.Module):
    """
    One-step 4th-order Runge–Kutta that evolves the *entire* state vector x
    when you have **one scalar-output model per state component** in
    `fx_list`.

    Each fx_i must implement  f_i(x, u) → ẋ_i (shape: B×1).

    Parameters
    ----------
    fx_list : list[nn.Module]   – length = nx, one SINDy model per state
    h       : float or 0-D tensor  – time step
    """
    def __init__(self, fx_list, h):
        super().__init__()
        self.fx_list = nn.ModuleList(fx_list)
        self.h       = torch.as_tensor(h)          # allows scalar or tensor

    # ---- helper to evaluate all f_i ----------------------------------------
    def _f(self, x, u):
        """Return full derivative vector ẋ matching x’s shape."""
        dx = torch.empty_like(x)
        for i, fx in enumerate(self.fx_list):
            dx[:, i:i+1] = fx(x, u)                # (B,1)
        return dx

    # ---- RK4 step -----------------------------------------------------------
    @torch.no_grad()
    def forward(self, x, u):
        """
        x : (B, nx)   – current state
        u : (B, nu)   – control input
        returns next state  x_{k+1}
        """
        h = self.h
        k1 = self._f(x, u)  # k1 = f(x_i, t_i)
        k2 = self.block(x + h * k1 / 2.0, u)  # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        k3 = self.block(x + h * k2 / 2.0, u)  # k3 = f(x_i + 0.5*h*k2, t_i + 0.5*h)
        k4 = self.block(x + h * k3, u)  # k4 = f(y_i + h*k3, t_i + h)
        return x + h * (k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0)
