# fx_library.py
import torch
from SparseDPC.src.sindy.library import FunctionLibrary
from SparseDPC.src.sindy.sindy import SINDy


# fx_library.py  ────────────────────────────────────────────────────────────
import itertools, torch
from .library import FunctionLibrary
from .sindy    import SINDy


def fx_library(
        main_idx:   int,
        nx:         int,
        nu:         int,
        seed:       int,
        device:     torch.device,
        *,
        max_degree: int  = 2,      # highest total polynomial degree
        add_sqrt:   bool = True,
        add_u_prod: bool = True,   # include u_p·u_q
        add_fourier: bool = False,  # include sin/cos terms
        max_freq:   int  = 1       # sin(k·x), cos(k·x) with k = 1…max_freq
):
    """
    SINDy function library generator.

    Contents
    --------
    • All monomials in x up to `max_degree`
    • sin(k·x_j) and cos(k·x_j) for k ≤ `max_freq`  (if `add_fourier`)
    • Linear controls u_k
    • Cross terms  x_j·u_k
    • Optional:  u_p·u_q    (`add_u_prod`)
    • Optional:  sqrt(x_j)  (`add_sqrt`)
    """
    theta_funs, theta_names = [], []

    # shorthand helpers
    def x(i): return f"x{i}"
    def u(k): return f"u_{k}"

    # ------------------------------------------------------------------ #
    # 1) monomials  (main state first)
    # ------------------------------------------------------------------ #
    theta_funs.append(lambda X, U, i=main_idx: X[:, i])
    theta_names.append(x(main_idx))

    for j in range(nx):
        if j != main_idx:
            theta_funs.append(lambda X, U, j=j: X[:, j])
            theta_names.append(x(j))

    for deg in range(2, max_degree + 1):
        for combo in itertools.combinations_with_replacement(range(nx), deg):

            def _poly_fun(combo):
                def _f(X, U, combo=combo):
                    out = torch.ones(X.size(0), device=X.device)
                    for idx in combo:
                        out = out * X[:, idx]
                    return out
                return _f

            counts = {i: combo.count(i) for i in set(combo)}
            name   = "*".join(f"{x(k)}" if p == 1 else f"{x(k)}^{p}"
                              for k, p in sorted(counts.items()))

            theta_funs.append(_poly_fun(combo))
            theta_names.append(name)

    # ------------------------------------------------------------------ #
    # 2) optional Fourier basis
    # ------------------------------------------------------------------ #
    if add_fourier and max_freq >= 1:
        for j in range(nx):
            for k in range(1, max_freq + 1):
                theta_funs.append(lambda X, U, j=j, k=k: torch.sin(k * X[:, j]))
                theta_names.append(f"sin({k}·{x(j)})")

                theta_funs.append(lambda X, U, j=j, k=k: torch.cos(k * X[:, j]))
                theta_names.append(f"cos({k}·{x(j)})")

    # ------------------------------------------------------------------ #
    # 3) control inputs u_k
    # ------------------------------------------------------------------ #
    for k in range(nu):
        theta_funs.append(lambda X, U, k=k: U[:, k])
        theta_names.append(u(k))

    if add_u_prod and nu >= 2:
        for p in range(nu):
            for q in range(p + 1, nu):
                theta_funs.append(lambda X, U, p=p, q=q: U[:, p] * U[:, q])
                theta_names.append(f"{u(p)}*{u(q)}")

    # ------------------------------------------------------------------ #
    # 4) x_j * u_k interactions
    # ------------------------------------------------------------------ #
    for k in range(nu):
        for j in range(nx):
            theta_funs.append(lambda X, U, j=j, k=k: X[:, j] * U[:, k])
            theta_names.append(f"{x(j)}*{u(k)}")

    # ------------------------------------------------------------------ #
    # 5) optional sqrt terms
    # ------------------------------------------------------------------ #
    if add_sqrt:
        for j in range(nx):
            theta_funs.append(lambda X, U, j=j: torch.sqrt(torch.clamp(X[:, j], 1e-6)))
            theta_names.append(f"sqrt({x(j)})")

    # ------------------------------------------------------------------ #
    # build FunctionLibrary + SINDy model
    # ------------------------------------------------------------------ #
    lib = FunctionLibrary(theta_funs,
                          n_features=1,
                          n_control=nu,
                          function_names=theta_names)

    return SINDy(library=lib,
                 main_idx=main_idx,
                 seed=seed,
                 device=device)


import itertools

def fx_policy_library(nx: int, nref: int, policy_name: str, seed: int, device: torch.device):
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
    return SINDy(library=theta_library, policy_name=policy_name, seed=seed, device=device)