import re
import torch
import torch.nn as nn
import copy
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import casadi
import re
import neuromancer.psl as psl


def convert_to_latex(names):
    latex_names = []
    for name in names:
        # === Step 1: Shift variable indices (e.g. x0 -> x1, u_0 -> u1) ===
        def shift(match):
            var, idx = match.group(1), match.group(2)
            return f"{var}{int(idx) + 1}"

        name = re.sub(r'\b(x|u)_?(\d+)\b', shift, name)

        # === Step 2: Replace with LaTeX formatting ===
        name = name.replace('*', r' \cdot ')
        name = re.sub(r'sin\((.*?)\)', r'\\sin(\1)', name)
        name = re.sub(r'cos\((.*?)\)', r'\\cos(\1)', name)
        name = re.sub(r'sqrt\((.*?)\)', r'\\sqrt{\1}', name)
        name = re.sub(r'\^(\d+)', r'^{\1}', name)
        name = re.sub(r'\bx(\d+)\b', r'x_{\1}', name)
        name = re.sub(r'\bu(\d+)\b', r'u_{\1}', name)

        latex_names.append(rf"${name}$")

    return latex_names


import re

import re

def convert_to_latex_policy(names):
    """
    Convert sparse policy basis names into proper LaTeX with:
      - automatic index shifting (x0→x1, x1→x2, etc.)
      - correct sqrt conversion
      - correct trig conversion
      - correct multiplication
      - correct superscripts and subscripts
    """

    latex_names = []

    # --- 1) Determine if shifting is needed ---
    shifting_needed = any(re.search(r'\bx_?0\b', n) for n in names)

    for name in names:
        original = name

        # ======================================================
        # 2) SHIFTING STAGE
        # ======================================================
        if shifting_needed:

            # r0 or r_0 → r
            name = re.sub(r'r_?0(?=\b|[^0-9])', 'r', name)

            # x0, x_0, x1, x_1 → shifted
            def shift_x(m):
                idx = int(m.group(1))
                return f"x{idx+1}"
            name = re.sub(r'x_?(\d+)', shift_x, name)

        # ======================================================
        # 3) LATEX CONVERSION STAGE
        # ======================================================

        # Replace multiplication
        name = name.replace('*', r' \cdot ')

        # sqrt(...)
        name = re.sub(r'sqrt\((.*?)\)', r'\\sqrt{\1}', name)

        # sin/cos
        name = re.sub(r'sin\((.*?)\)', r'\\sin(\1)', name)
        name = re.sub(r'cos\((.*?)\)', r'\\cos(\1)', name)

        # Powers
        name = re.sub(r'\^(\d+)', r'^{\1}', name)

        # Subscripts AFTER sqrt and trig
        name = re.sub(r'x(\d+)', r'x_{\1}', name)
        name = re.sub(r'u(\d+)', r'u_{\1}', name)
        name = re.sub(r'r(\d+)', r'r_{\1}', name)

        # Wrap in math mode
        latex_names.append(f"${name}$")

    return latex_names


import torch
import torch.nn.functional as F
import numpy as np


def run_multi_seed_rollouts(
    systems: dict,
    nsteps: int,
    nx: int,
    nu: int,
    nref: int,
    xmin: float,
    xmax: float,
    umin: float,
    umax: float,
    device: torch.device,
    step_length: int,            # <--- added
    n_seeds: int = 1,
    seed0: int = 0,
    include_mpc: bool = False,
    mpc_fn=None,                 # e.g., mpc_solve_adaptive_horizon
    mpc_kwargs=None              # dict passed to mpc_fn
):
    """
    Runs rollouts for multiple controllers (Neuromancer systems + optional MPC)
    using the same step reference construction as in the manual code.
    """

    if mpc_kwargs is None:
        mpc_kwargs = {}

    # --- Setup Neuromancer systems ---
    for sys in systems.values():
        sys.nsteps = nsteps

    Y_cols = {name: [] for name in systems.keys()}
    U_cols = {name: [] for name in systems.keys()}
    losses = {name: [] for name in systems.keys()}

    if include_mpc:
        Y_cols["MPC"] = []
        U_cols["MPC"] = []
        losses["MPC"] = []

    # --- Rollout loop ---
    for s in range(seed0, seed0 + n_seeds):
        torch.manual_seed(s)
        rng = np.random.default_rng(s)

        # ----- Build reference (exactly like your manual code) -----
        # np_ref_1: shape (nsteps+2, 1)
        np_ref_1 = psl.signals.step(
            nsteps + 2,
            1,
            min=0.2,
            max=0.8,
            randsteps=nsteps // step_length,
            rng=rng,
        )
        R_1 = torch.tensor(np_ref_1, dtype=torch.float32).reshape(1, nsteps + 2, 1)
        torch_ref = torch.cat([R_1, R_1], dim=-1).to(device)  # (1, nsteps+2, nref=2)

        # ----- Random initial condition in [xmin, xmax] -----
        xn0 = torch.rand(1, 1, nx, dtype=torch.float32) * (xmax - xmin) + xmin
        xn0 = xn0.to(device)

        data = {
            "xn": xn0,
            "x1_n": xn0[:, :, 0:1],
            "x2_n": xn0[:, :, 1:2],
            "r": torch_ref,
        }

        # === Neuromancer Systems ===
        for name, sys in systems.items():
            out = sys(data)

            xn_out = out["xn"].detach().cpu().reshape(nsteps + 1, nx)
            u_out = out["u"].detach().cpu().reshape(nsteps, nu)

            Y_cols[name].append(xn_out)
            U_cols[name].append(u_out)

            # MSE vs reference (first nsteps+1 steps)
            ref_target = torch_ref[:, :nsteps + 1].cpu()
            mse = F.mse_loss(out["xn"].cpu(), ref_target).item()
            losses[name].append(mse)

        # === MPC baseline ===
        if include_mpc and mpc_fn is not None:
            mpc_out = mpc_fn(
                data,
                **mpc_kwargs,
            )  # expected to return numpy arrays for 'xn', 'u'

            xn_out = np.array(mpc_out["xn"]).reshape(-1, nx)
            u_out = np.array(mpc_out["u"]).reshape(-1, nu)

            # Align lengths
            xn_out = xn_out[: nsteps + 1]
            u_out = u_out[:nsteps]

            Y_cols["MPC"].append(xn_out)
            U_cols["MPC"].append(u_out)

            pred = torch.tensor(xn_out, dtype=torch.float32)
            ref = torch_ref[:, :nsteps + 1].reshape(nsteps + 1, nref).cpu()
            losses["MPC"].append(F.mse_loss(pred, ref).item())

    # --- Aggregate Results ---
    controller_labels = list(systems.keys())
    if include_mpc:
        controller_labels.append("MPC")

    Y_list, U_list = [], []
    for name in controller_labels:
        Ys = [
            y.numpy() if isinstance(y, torch.Tensor) else y
            for y in Y_cols[name]
        ]
        Us = [
            u.numpy() if isinstance(u, torch.Tensor) else u
            for u in U_cols[name]
        ]
        Y = np.concatenate(Ys, axis=1)  # (nsteps+1, nx * n_seeds)
        U = np.concatenate(Us, axis=1)  # (nsteps,   nu * n_seeds)
        Y_list.append(Y)
        U_list.append(U)

    # Just use the last constructed reference (all seeds share construction logic)
    R_np = (
        torch_ref[:, :nsteps + 1]
        .detach()
        .cpu()
        .reshape(nsteps + 1, nref)
        .numpy()
    )

    losses_vec = [float(np.mean(losses[name])) for name in controller_labels]

    return Y_list, U_list, controller_labels, R_np, losses_vec


import time
import numpy as np
import torch
import casadi

def mpc_solve_adaptive_horizon(
    data, two_tank,
    N_max=50, nsim=750, step_length=150, N_min=2,
    Qy=1.0, Ru=1e-4, Rdu=1e-3, Rdu_cross=None,
    softplus_beta=50.0, p=1e-6
):
    """
    Variable-horizon MPC that rebuilds the NLP whenever N_k changes.
    Adds both in-horizon move penalty (Rdu) and cross-step move penalty (Rdu_cross)
    on the first control vs the previously applied control.

    Horizon rule at time k (before end-of-ref clipping):
        remaining = step_length - (k % step_length)
        N_k = clip(remaining, N_min, N_max)

    The actual N_k used is also clipped to the remaining reference:
        N_k = min(N_k, T_ref - k - 1)

    Assumes reference length >= nsim + 1.
    """
    if Rdu_cross is None:
        Rdu_cross = Rdu  # small but nonzero cross-step smoothing

    # ---------- Extract IC and reference ----------
    x0_init = data['xn'][0, 0].detach().cpu().numpy().astype(float)  # (2,)
    ref_traj = data['r'][0].detach().cpu().numpy().astype(float)     # (T,2)
    T_ref = ref_traj.shape[0]
    assert T_ref >= nsim + 1, "ref_traj must have length at least nsim + 1."

    # ---------- Scalars as floats ----------
    def _to_float(x):
        if isinstance(x, (float, int)): return float(x)
        if hasattr(x, "item"):          return float(x.item())
        return float(x)

    c1 = _to_float(two_tank.c1)
    c2 = _to_float(two_tank.c2)
    dt = _to_float(two_tank.ts)

    nx, nu = 2, 2
    umin, umax = 0.0, 1.0
    xmin, xmax = 0.0, 1.0

    # ---------- Softplus to keep sqrt-argument positive ----------
    def softplus(z):
        return (1.0 / softplus_beta) * casadi.log1p(casadi.exp(softplus_beta * z))

    # ---------- Builder for a solver of size N_k ----------
    def build_solver(N_k):
        opti = casadi.Opti()
        X = opti.variable(nx, N_k + 1)
        U = opti.variable(nu, N_k)

        x0_param    = opti.parameter(nx)
        ref_param   = opti.parameter(nx, N_k + 1)
        u_prev_param= opti.parameter(nu)   # NEW: previous applied control

        def ode_equations(x, u):
            h1 = softplus(x[0])  # strictly positive inside sqrt
            h2 = softplus(x[1])
            pump  = u[0]
            valve = u[1]
            dhdt1 = c1 * (1.0 - valve) * pump - c2 * casadi.sqrt(h1 + p)
            dhdt2 = c1 * valve * pump + c2 * casadi.sqrt(h1 + p) - c2 * casadi.sqrt(h2 + p)
            return casadi.vertcat(dhdt1, dhdt2)

        # Multiple-shooting (Euler)
        for i in range(N_k):
            fi = ode_equations(X[:, i], U[:, i])
            opti.subject_to(X[:, i + 1] == X[:, i] + dt * fi)

        # Bounds
        opti.subject_to(opti.bounded(umin, U, umax))
        opti.subject_to(opti.bounded(xmin, X, xmax))
        opti.subject_to(X[:, 0] == x0_param)

        # Cost
        cost = 0
        # Cross-step smoothing on the first control vs previous applied control:
        cost += Rdu_cross * casadi.sumsqr(U[:, 0] - u_prev_param)

        for i in range(N_k):
            cost += Qy * casadi.sumsqr(X[:, i] - ref_param[:, i]) \
                  + Ru * casadi.sumsqr(U[:, i])
            if i > 0:
                cost += Rdu * casadi.sumsqr(U[:, i] - U[:, i - 1])

        cost += Qy * casadi.sumsqr(X[:, N_k] - ref_param[:, N_k])
        opti.minimize(cost)

        # IPOPT
        opti.solver('ipopt', {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 500,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.bound_push": 1e-6,
            "ipopt.bound_frac": 1e-6,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.hessian_approximation": "limited-memory",
        })

        # Numeric ODE function (for rollout)
        Xsym = casadi.MX.sym('x', nx)
        Usym = casadi.MX.sym('u', nu)
        f_ode = casadi.Function('f_ode', [Xsym, Usym], [ode_equations(Xsym, Usym)])

        return {
            "opti": opti, "X": X, "U": U,
            "x0_param": x0_param, "ref_param": ref_param,
            "u_prev_param": u_prev_param,
            "f_ode": f_ode
        }

    # ---------- Simulation ----------
    Xs = [x0_init.copy()]
    Us_hist = []
    times = []

    x0 = x0_init.copy()
    prev = None
    prev_N = None
    solver = None

    # Initialize previous control (used in cross-step penalty).
    # Choose center of bounds as a neutral starting guess.
    u_prev = 0.5 * np.ones(nu)

    for k in range(nsim):
        # Rule-based N_k, then clip to remaining reference
        rem_set = step_length - (k % step_length)            # 1..step_length
        N_k = int(np.clip(rem_set, N_min, N_max))
        N_k = int(min(N_k, T_ref - k - 1))                   # ensure k+N_k+1 <= T_ref
        if N_k < 1:
            # No more preview; just hold previous control (or zeros)
            Us_hist.append(u_prev.copy())
            # rollout one step with held input
            # To keep it simple here, skip model rollout if end-of-ref (optional)
            Xs.append(x0.copy())
            times.append(0.0)
            continue

        # Build/refactor solver if size changed
        if (solver is None) or (prev_N != N_k):
            solver = build_solver(N_k)
            prev = None
            prev_N = N_k

        opti       = solver["opti"]
        X_var      = solver["X"]
        U_var      = solver["U"]
        x0_par     = solver["x0_param"]
        ref_par    = solver["ref_param"]
        u_prev_par = solver["u_prev_param"]
        f_ode      = solver["f_ode"]

        # Parameters for this step
        win = ref_traj[k : k + N_k + 1, :]       # (N_k+1, nx)
        opti.set_value(x0_par, x0)
        opti.set_value(ref_par, win.T)
        opti.set_value(u_prev_par, u_prev)

        # Warm start
        if prev is None:
            opti.set_initial(X_var, np.tile(x0.reshape(-1, 1), (1, N_k + 1)))
            opti.set_initial(U_var, np.tile(u_prev.reshape(-1,1), (1, N_k)))
        else:
            prev_X, prev_U = prev
            X_guess = np.hstack([prev_X[:, 1:], prev_X[:, [-1]]])
            U_guess = np.hstack([prev_U[:, 1:], prev_U[:, [-1]]])
            opti.set_initial(X_var, X_guess)
            opti.set_initial(U_var, U_guess)

        # Solve
        t0 = time.time()
        sol = opti.solve()
        elapsed = time.time() - t0

        X_sol = sol.value(X_var)   # (nx, N_k+1)
        U_sol = sol.value(U_var)   # (nu, N_k)

        # Apply first control and rollout one step
        u0 = U_sol[:, 0]
        xdot = np.array(f_ode(x0, u0)).reshape(-1)
        x0 = (x0 + dt * xdot).astype(float)

        # Log + update cross-step penalty reference
        Xs.append(x0.copy())
        Us_hist.append(u0.copy())
        times.append(elapsed)
        prev = (X_sol, U_sol)
        u_prev = u0.copy()  # for next step's cross-step penalty

    # ---------- Pack results ----------
    Xs_arr = np.array(Xs)        # (nsim+1, nx)
    Us_arr = np.array(Us_hist)   # (nsim,   nu)
    results = {
        'xn': torch.tensor(Xs_arr[None, ...], dtype=torch.float32),
        'u':  torch.tensor(Us_arr[None, ...], dtype=torch.float32),
        'r':  torch.tensor(ref_traj[None, ...], dtype=torch.float32),
        'times': np.array(times, dtype=float),
    }
    return results

def prune_model(model, tol: float = 0.0, *, in_place: bool = False, reinit: bool = False, reinit_std: float = 0.01):
    """
    Prune terms whose |coef| <= tol and remove the corresponding entries from the library.
    Mirrors the 'mask -> active_idx -> slice library + coef' approach in your reference.

    Args:
        model: object with
            - .coef: (n_terms,) or (n_terms, out_dim) tensor / nn.Parameter
            - .library.function_names or .library.functions_names: list[str]
            - optional .library.library: list[callable/objs]
            - optional .library.functions: list[callable]
            - optional .library.shape: tuple
            - optional .library.term_indices: list[...]  (pruned if present)
        tol: threshold; keep a row if any |coef[row]| > tol (for 2D).
        in_place: modify the passed model in place; otherwise returns a deep-copied pruned model.
        reinit: if True, reinitialize the kept coefficients with N(0, reinit_std^2).
        reinit_std: std for reinitialization when reinit=True.

    Returns:
        pruned model (same object if in_place=True, otherwise a deep-copied one).
    """
    m = model if in_place else copy.deepcopy(model)

    with torch.no_grad():
        coef = m.coef
        if coef.ndim == 1:
            mask = torch.abs(coef) > tol                   # (n_terms,)
        elif coef.ndim == 2:
            mask = torch.abs(coef).amax(dim=1) > tol       # (n_terms,)
        else:
            raise ValueError(f"Unsupported coef ndim={coef.ndim}")

        active_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if active_idx.numel() == 0:
            # keep at least one term to avoid degenerate model
            print("Warning: all terms would be pruned; keeping the largest-magnitude term.")
            if coef.ndim == 1:
                keep_idx = torch.argmax(torch.abs(coef)).view(-1)
            else:
                keep_idx = torch.argmax(torch.abs(coef).amax(dim=1)).view(-1)
            active_idx = keep_idx

        idx = active_idx.tolist()

        # --- slice coefficients
        if reinit:
            new_coef = torch.randn_like(coef[idx, ...]) * reinit_std if coef.ndim == 2 \
                       else torch.randn_like(coef[idx]) * reinit_std
        else:
            new_coef = coef[idx, ...] if coef.ndim == 2 else coef[idx]

        # preserve Parameter-ness and requires_grad
        if isinstance(m.coef, nn.Parameter):
            m.coef = nn.Parameter(new_coef, requires_grad=model.coef.requires_grad)
        else:
            m.coef = new_coef

        # --- resolve names attribute
        names_attr = 'function_names' if hasattr(m.library, 'function_names') else (
                     'functions_names' if hasattr(m.library, 'functions_names') else None)
        if names_attr is not None:
            names = getattr(m.library, names_attr)
            setattr(m.library, names_attr, [names[i] for i in idx])

        # --- prune parallel library structures if present
        if hasattr(m.library, 'library'):
            lib_list = getattr(m.library, 'library')
            setattr(m.library, 'library', [lib_list[i] for i in idx])

        if hasattr(m.library, 'functions'):
            funcs = getattr(m.library, 'functions')
            setattr(m.library, 'functions', [funcs[i] for i in idx])

        if hasattr(m.library, 'term_indices'):
            term_idx = getattr(m.library, 'term_indices')
            setattr(m.library, 'term_indices', [term_idx[i] for i in idx])

        # --- update shape if provided
        if hasattr(m.library, 'shape'):
            old_shape = getattr(m.library, 'shape')
            # assume shape = (n_terms, something)
            setattr(m.library, 'shape', (len(idx), old_shape[1] if len(old_shape) > 1 else 1))

    return m


def _make_ref(nsteps, step_length, xmin, xmax, seed):
    rng = np.random.default_rng(seed)
    np_refs = psl.signals.step(nsteps+2, 1, min=0.2, max=0.8, randsteps=nsteps//step_length, rng=rng)
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps + 2, 1)

    return torch.cat([R, R], dim=-1)  # 2-state ref

def _extract_states(xn_tensor, nx):
    if isinstance(xn_tensor, torch.Tensor):
        arr = xn_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(xn_tensor)
    if arr.ndim == 3:
        arr = arr[0]
    assert arr.shape[-1] == nx, f"expected last dim {nx}, got {arr.shape}"
    return arr

def _tracking_mse(x_traj, r_traj):
    diff = x_traj - r_traj[1:,:]
    return float(np.mean(diff**2))

def _mean_violation(x_traj, xmin, xmax):
    below = np.maximum(0.0, xmin - x_traj)
    above = np.maximum(0.0, x_traj - xmax)
    return float(np.mean(below + above))

import time
import numpy as np
import pandas as pd
import torch

def evaluate_controllers_over_seeds(
    systems_dict,
    nsteps,
    step_length,
    nx,
    xmin=0.0,
    xmax=1.0,
    n_seeds=200,
    torch_device=None,
    *,
    # --- MPC hooks (optional) ---
    mpc_solver=None,          # e.g., mpc_solve_adaptive_horizon_rebuild
    mpc_model=None,           # e.g., gt_model (Two-Tank params)
    mpc_params=None           # dict of kwargs for mpc_solver (e.g., N_max, N_min, Qy, Ru, Rdu, Rdu_cross, etc.)
):
    """
    Evaluates controllers over multiple seeds.
    - If a controller name starts with 'MPC', this will call `mpc_solver(data, mpc_model, nsim=nsteps, step_length=step_length, **mpc_params)`.
    - Otherwise it calls the controller object directly: cl_system(data).

    Notes:
      * Assumes helpers `_make_ref`, `_extract_states`, `_tracking_mse`, `_mean_violation` exist.
      * For the MPC 'rebuild on N_k' implementation, the reference only needs shape (1, nsteps+1, nx).
    """
    if mpc_params is None:
        mpc_params = {}

    results = []

    for name, cl_system in systems_dict.items():
        print(f"\n=== Evaluating controller: {name} ===")
        tracking_list, viol_list, times = [], [], []

        is_mpc = name.startswith("MPC")
        if is_mpc:
            if mpc_solver is None:
                raise ValueError("Controller name starts with 'MPC' but `mpc_solver` was not provided.")
            if mpc_model is None:
                raise ValueError("Controller name starts with 'MPC' but `mpc_model` was not provided.")

        for seed in range(n_seeds):
            if (seed + 1) % 20 == 0 or seed == 0:
                print(f"  Seed {seed+1}/{n_seeds}...")

            torch.manual_seed(seed)

            # Initial state and reference
            x0 = torch.rand(1, 1, nx, dtype=torch.float32)
            R = _make_ref(nsteps, step_length, xmin, xmax, seed)   # expected: (1, nsteps+1, nx)

            data = {'xn': x0, 'r': R}

            # Non-MPC controllers in your code rely on the nsteps attribute
            if not is_mpc and hasattr(cl_system, 'nsteps'):
                cl_system.nsteps = nsteps

            # Send tensors to device if requested
            if torch_device is not None:
                data = {k: (v.to(torch_device) if isinstance(v, torch.Tensor) else v)
                        for k, v in data.items()}

            # Run the controller and time it
            t0 = time.time()
            if is_mpc:
                call_kwargs = {'nsim': nsteps, 'step_length': step_length}
                call_kwargs.update(mpc_params)
                print("MPC on SEED: ", seed)
                out = mpc_solver(data, mpc_model, **call_kwargs)
            else:
                out = cl_system(data)
            elapsed = time.time() - t0
            times.append(elapsed)

            # Extract and score
            x_traj = _extract_states(out['xn'], nx)
            r_traj = R.detach().cpu().numpy()[0]

            tracking_list.append(_tracking_mse(x_traj, r_traj))
            viol_list.append(_mean_violation(x_traj, xmin, xmax))

        # Aggregate
        tracking_arr = np.array(tracking_list, dtype=float)
        viol_arr = np.array(viol_list, dtype=float)
        times_arr = np.array(times, dtype=float)

        results.append({
            'controller': name,
            'tracking_mse_mean': float(tracking_arr.mean()),
            'tracking_mse_std': float(tracking_arr.std(ddof=0)),
            'tracking_mse_max': float(tracking_arr.max()),
            'tracking_mse_min': float(tracking_arr.min()),
            'violation_mse_mean': float(viol_arr.mean()),
            'violation_mse_std': float(viol_arr.std(ddof=0)),
            'violation_mse_max': float(viol_arr.max()),
            'violation_mse_min': float(viol_arr.min()),
            'time_mean': float(times_arr.mean()),
            'time_std': float(times_arr.std(ddof=0)),
            'time_max': float(times_arr.max()),
            'time_min': float(times_arr.min()),
        })

        print(f"  Done with {name}: "
              f"MSE={tracking_arr.mean():.4e}±{tracking_arr.std():.4e}, "
              f"Violation={viol_arr.mean():.4e}±{viol_arr.std():.4e}, "
              f"Time={times_arr.mean():.4e}±{times_arr.std():.4e}")

    return pd.DataFrame(results).set_index('controller')
