import torch
import itertools
from typing import Callable, List

from SparseDPC.sindy.library import FunctionLibrary
from SparseDPC.sindy.sindy import SINDy


def fx_library(
    main_idx: int,
    nx: int,
    nu: int,
    seed: int,
    device: torch.device,
    *,
    max_degree: int = 2,
    add_sqrt: bool = True,
    add_u_prod: bool = True,
    add_fourier: bool = False,
    max_freq: int = 1
) -> SINDy:
    """
    Builds a SINDy model with a symbolic function library for dynamics learning.

    Includes polynomial, control, and optional sinusoidal features.

    Returns:
        SINDy: Configured symbolic dynamics model
    """
    theta_funs: List[Callable] = []
    theta_names: List[str] = []

    def x(i: int) -> str: return f"x{i}"
    def u(k: int) -> str: return f"u_{k}"

    # State monomials (linear first, main state first)
    theta_funs.append(lambda X, U, i=main_idx: X[:, i])
    theta_names.append(x(main_idx))

    for j in range(nx):
        if j != main_idx:
            theta_funs.append(lambda X, U, j=j: X[:, j])
            theta_names.append(x(j))

    # Higher-degree monomials
    for deg in range(2, max_degree + 1):
        for combo in itertools.combinations_with_replacement(range(nx), deg):
            def _poly_fun(combo=combo):
                def _f(X, U):
                    out = torch.ones(X.size(0), device=X.device)
                    for idx in combo:
                        out *= X[:, idx]
                    return out
                return _f

            counts = {i: combo.count(i) for i in set(combo)}
            name = "*".join(f"{x(k)}" if p == 1 else f"{x(k)}^{p}"
                            for k, p in sorted(counts.items()))

            theta_funs.append(_poly_fun())
            theta_names.append(name)

    # Optional Fourier terms
    if add_fourier and max_freq >= 1:
        for j in range(nx):
            for k in range(1, max_freq + 1):
                theta_funs.append(lambda X, U, j=j, k=k: torch.sin(k * X[:, j]))
                theta_names.append(f"sin({k}·{x(j)})")

                theta_funs.append(lambda X, U, j=j, k=k: torch.cos(k * X[:, j]))
                theta_names.append(f"cos({k}·{x(j)})")

    # Control inputs
    for k in range(nu):
        theta_funs.append(lambda X, U, k=k: U[:, k])
        theta_names.append(u(k))

    # Optional control cross-terms
    if add_u_prod and nu >= 2:
        for p in range(nu):
            for q in range(p + 1, nu):
                theta_funs.append(lambda X, U, p=p, q=q: U[:, p] * U[:, q])
                theta_names.append(f"{u(p)}*{u(q)}")

    # Cross terms between states and controls
    for k in range(nu):
        for j in range(nx):
            theta_funs.append(lambda X, U, j=j, k=k: X[:, j] * U[:, k])
            theta_names.append(f"{x(j)}*{u(k)}")

    # Optional sqrt terms
    if add_sqrt:
        for j in range(nx):
            theta_funs.append(lambda X, U, j=j: torch.sqrt(torch.clamp(X[:, j], min=1e-6)))
            theta_names.append(f"sqrt({x(j)})")

    # Final library and model
    lib = FunctionLibrary(theta_funs, n_features=1, n_control=nu, function_names=theta_names)
    return SINDy(library=lib, main_idx=main_idx, seed=seed, device=device)


def fx_policy_library(
    nx: int,
    nref: int,
    policy_name: str,
    seed: int,
    device: torch.device
) -> SINDy:
    """
    Returns a symbolic policy model using mixed state-reference basis.

    Includes linear, trigonometric, and polynomial interactions between x and r.
    """
    theta_funs: List[Callable] = []
    theta_names: List[str] = []

    # x features
    for i in range(nx):
        theta_funs.append(lambda X, r, i=i: X[:, i])
        theta_names.append(f"x_{i}")

        theta_funs.append(lambda X, r, i=i: torch.sin(X[:, i]))
        theta_names.append(f"sin(x_{i})")

        theta_funs.append(lambda X, r, i=i: torch.cos(X[:, i]))
        theta_names.append(f"cos(x_{i})")

        theta_funs.append(lambda X, r, i=i: torch.sin(2 * X[:, i]))
        theta_names.append(f"sin(2*x_{i})")

        theta_funs.append(lambda X, r, i=i: torch.cos(2 * X[:, i]))
        theta_names.append(f"cos(2*x_{i})")

        theta_funs.append(lambda X, r, i=i: X[:, i] * torch.sin(X[:, i]))
        theta_names.append(f"x_{i}*sin(x_{i})")

        theta_funs.append(lambda X, r, i=i: X[:, i] * torch.cos(X[:, i]))
        theta_names.append(f"x_{i}*cos(x_{i})")

    # Pairwise x_i * x_j
    for i, j in itertools.combinations(range(nx), 2):
        theta_funs.append(lambda X, r, i=i, j=j: X[:, i] * X[:, j])
        theta_names.append(f"x_{i} * x_{j}")

    # r features
    for i in range(nref):
        theta_funs.append(lambda X, r, i=i: r[:, i])
        theta_names.append(f"r_{i}")

        theta_funs.append(lambda X, r, i=i: torch.cos(r[:, i]))
        theta_names.append(f"cos(r_{i})")

        theta_funs.append(lambda X, r, i=i: torch.sin(r[:, i]))
        theta_names.append(f"sin(r_{i})")

        theta_funs.append(lambda X, r, i=i: torch.cos(2 * r[:, i]))
        theta_names.append(f"cos(2*r_{i})")

        theta_funs.append(lambda X, r, i=i: torch.sin(2 * r[:, i]))
        theta_names.append(f"sin(2*r_{i})")

    # x_i * r_j terms
    for i in range(nx):
        for j in range(nref):
            theta_funs.append(lambda X, r, i=i, j=j: X[:, i] * r[:, j])
            theta_names.append(f"x_{i} * r_{j}")

            theta_funs.append(lambda X, r, i=i, j=j: X[:, i]**2 * r[:, j])
            theta_names.append(f"x_{i}^2 * r_{j}")

            theta_funs.append(lambda X, r, i=i, j=j: X[:, i] * r[:, j]**2)
            theta_names.append(f"x_{i} * r_{j}^2")

    lib = FunctionLibrary(theta_funs, n_features=1, n_control=nref, function_names=theta_names)
    return SINDy(library=lib, policy_name=policy_name, seed=seed, device=device)
