# fx_library.py
import torch
from SparseDPC.src.sindy.library import FunctionLibrary
from SparseDPC.src.sindy.sindy import SINDy

def fx_library(main_idx: int, nx: int, nu: int, device):
    """
    Build a SINDy library whose *first* column is x_{main_idx}
    and whose remaining columns are all other x_j plus controls.

    Parameters
    ----------
    main_idx : which state is the “main” one (0‑based)
    nx       : total number of states
    nu       : # of control inputs
    device   : torch.device
    """
    theta_funs, theta_names = [], []

    # alias helpers
    def x(i):  return f"x{i}"
    def u(j):  return f"u_{j}"

    # ➊ linear terms ----------------------------------------------------------
    theta_funs.append(lambda X, U, i=main_idx: X[:, i])
    theta_names.append(x(main_idx))

    for j in range(nx):
        if j == main_idx: continue
        theta_funs.append(lambda X, U, j=j: X[:, j])
        theta_names.append(x(j))

    # ➋ quadratic and cross‑products -----------------------------------------
    theta_funs.append(lambda X, U, i=main_idx: X[:, i] ** 2)
    theta_names.append(f"{x(main_idx)}^2")

    for j in range(nx):
        if j == main_idx: continue
        theta_funs.append(lambda X, U, j=j: X[:, j] ** 2)
        theta_names.append(f"{x(j)}^2")

        theta_funs.append(lambda X, U, i=main_idx, j=j: X[:, i] * X[:, j])
        theta_names.append(f"{x(main_idx)}*{x(j)}")


    # ➌ control inputs --------------------------------------------------------
    for k in range(nu):
        theta_funs.append(lambda X, U, k=k: U[:, k])
        theta_names.append(u(k))

    # NEW: pairwise control–control products  u_p * u_q  ----------------------
    if nu >= 2:
        for p in range(nu):
            for q in range(p + 1, nu):               # avoid duplicates & squares
                theta_funs.append(
                    lambda X, U, p=p, q=q: U[:, p] * U[:, q]
                )
                theta_names.append(f"{u(p)}*{u(q)}")

    # ➍ interactions with controls -------------------------------------------
    for k in range(nu):
        theta_funs.append(lambda X, U, i=main_idx, k=k: X[:, i] * U[:, k])
        theta_names.append(f"{x(main_idx)}*{u(k)}")

        for j in range(nx):
            if j == main_idx: continue
            theta_funs.append(lambda X, U, j=j, k=k: X[:, j] * U[:, k])
            theta_names.append(f"{x(j)}*{u(k)}")

    # ➎ sqrt terms ------------------------------------------------------------
    theta_funs.append(
        lambda X, U, i=main_idx: torch.sqrt(torch.clamp(X[:, i], 1e-6))
    )
    theta_names.append(f"sqrt({x(main_idx)})")

    for j in range(nx):
        if j == main_idx: continue
        theta_funs.append(
            lambda X, U, j=j: torch.sqrt(torch.clamp(X[:, j], 1e-6))
        )
        theta_names.append(f"sqrt({x(j)})")

    # create library + SINDy model -------------------------------------------
    lib = FunctionLibrary(theta_funs, n_features=1,
                          n_control=nu, function_names=theta_names)
    return SINDy(library=lib, main_idx= main_idx)
