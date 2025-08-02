# fx_library.py
import torch
from SparseDPC.src.sindy.library import FunctionLibrary
from SparseDPC.src.sindy.sindy import SINDy
def fx_library(main_idx: int, nx: int, nu: int, seed: int):
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
    return SINDy(library=lib, main_idx= main_idx, seed=seed)


import itertools

def fx_policy_library(nx: int, nref: int, policy_name: str, seed: int):
    n_features = nx
    nref = nref

    theta_funs = []
    theta_names = []

    theta_funs += [(lambda X, r, i=i: X[:, i]) for i in range(n_features)]
    theta_names += [f"x_{i}" for i in range(n_features)]

    theta_funs += [(lambda X, r, i=i: torch.sin(X[:, i])) for i in range(n_features)]
    theta_names += [f"sin(x_{i})" for i in range(n_features)]

    theta_funs += [(lambda X, r, i=i: torch.cos(X[:, i])) for i in range(n_features)]
    theta_names += [f"cos(x_{i})" for i in range(n_features)]

    theta_funs += [(lambda X, r, i=i: torch.sin(2*X[:, i])) for i in range(n_features)]
    theta_names += [f"sin(2*x_{i})" for i in range(n_features)]

    theta_funs += [(lambda X, r, i=i: torch.cos(2*X[:, i])) for i in range(n_features)]
    theta_names += [f"cos(2*x_{i})" for i in range(n_features)]
    #
    theta_funs += [(lambda X, r, i=i: X[:, i] * torch.sin(X[:, i])) for i in range(n_features)]
    theta_names += [f"x_{i}*sin(x_{i})" for i in range(n_features)]

    theta_funs += [(lambda X, r, i=i: X[:, i] * torch.cos(X[:, i])) for i in range(n_features)]
    theta_names += [f"x_{i}*cos(x_{i})" for i in range(n_features)]


    for i, j in itertools.combinations(range(n_features), 2):
        theta_funs.append(lambda X, r, i=i, j=j: X[:, i] * X[:, j])
        theta_names.append(f"x_{i} * x_{j}")

    theta_funs += [(lambda X, r, i=i: r[:, i]) for i in range(nref)]
    theta_names += [f"r_{i}" for i in range(nref)]

    theta_funs += [(lambda X, r, i=i: torch.cos(r[:, i])) for i in range(nref)]
    theta_names += [f"cos(r_{i})" for i in range(nref)]

    theta_funs += [(lambda X, r, i=i: torch.sin(r[:, i])) for i in range(nref)]
    theta_names += [f"sin(r_{i})" for i in range(nref)]

    theta_funs += [(lambda X, r, i=i: torch.cos(2*r[:, i])) for i in range(nref)]
    theta_names += [f"cos(2*r_{i})" for i in range(nref)]

    theta_funs += [(lambda X, r, i=i: torch.sin(2*r[:, i])) for i in range(nref)]
    theta_names += [f"sin(2*r_{i})" for i in range(nref)]

    theta_funs += [(lambda X, r, i=i, j=j: X[:, i] * r[:, j]) for i in range(n_features) for j in range(nref)]
    theta_names += [f"x_{i} * r_{j}" for i in range(n_features) for j in range(nref)]

    theta_funs += [(lambda X, r, i=i, j=j: X[:, i]**2 * r[:, j]) for i in range(n_features) for j in range(nref)]
    theta_names += [f"x_{i}^2 * r_{j}" for i in range(n_features) for j in range(nref)]

    theta_funs += [(lambda X, r, i=i, j=j: X[:, i] * r[:, j]**2) for i in range(n_features) for j in range(nref)]
    theta_names += [f"x_{i} * r_{j}^2" for i in range(n_features) for j in range(nref)]

    # Create function library
    theta_library = FunctionLibrary(theta_funs, 1, nref, theta_names)
    # Return the SINDy model
    return SINDy(library=theta_library, policy_name=policy_name, seed=seed)