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
def convert_to_latex_policy(names):
    """
    Correctly shifts:
    - x_0 -> x_1
    - x_1 -> x_2
    - r_0 -> r
    and applies LaTeX formatting for all terms,
    including inside composite expressions.
    """
    latex_names = []

    for name in names:
        # Step 1: Safely handle r_0 first
        name = re.sub(r'r_0', r'r', name)

        # Step 2: Temporary tags to avoid double-replacement
        name = name.replace('x_0', 'xTEMP0')
        name = name.replace('x_1', 'xTEMP1')

        # Step 3: Now shift safely
        name = name.replace('xTEMP0', 'x_1')
        name = name.replace('xTEMP1', 'x_2')

        # Step 4: LaTeX formatting
        name = name.replace('*', r' \cdot ')  # Multiplications
        name = re.sub(r'sin\((.*?)\)', r'\\sin(\1)', name)
        name = re.sub(r'cos\((.*?)\)', r'\\cos(\1)', name)
        name = re.sub(r'\^(\d+)', r'^{\1}', name)

        # Step 5: Add LaTeX subscripts for any x_N (after shift)
        name = re.sub(r'(?<![a-zA-Z])x_(\d+)', r'x_{\1}', name)

        latex_names.append(rf"${name}$")

    return latex_names

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


class GenericTrigPolicy(nn.Module):
    """
    Library-driven policy:
        - function_names: list[str] like ["cos(2*x_0)", "cos(2*x_1)", "cos(2*r_0)", "sin(2*r_0)", ...]
        - coefs: torch.Tensor of shape (n_funcs,)
    Forward:
        - X: (batch, nx)
        - R: (batch, nref)
        -> u: (batch, 1) == sum_j coefs[j] * f_j(X,R)
    """

    _allowed = {
        # torch ops
        "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
        "exp": torch.exp, "sqrt": torch.sqrt, "abs": torch.abs,
        "log": torch.log,
        # basic math
        "pi": torch.pi,
        # allow Python's pow that delegates to torch.pow via our mapping:
        "pow": torch.pow,
    }

    def __init__(self, function_names, coefs: torch.Tensor):
        super().__init__()
        assert isinstance(function_names, (list, tuple)) and len(function_names) > 0
        self.function_names = list(function_names)

        # Store coefficients (not trainable). To make trainable: use nn.Parameter instead.
        self.register_buffer("coefs", coefs.clone().detach().flatten())
        assert self.coefs.numel() == len(self.function_names), "coefs length must match #functions"

        # Compile expressions to callables
        self._compiled = [self._compile_expr(expr) for expr in self.function_names]

    @staticmethod
    def _prepare_expr(expr: str) -> str:
        """
        Convert a user expression into a torch-evaluable Python expression:
          - x_i -> X[:, i]
          - r_i -> R[:, i]
          - '^' -> '**'
        """
        s = expr.strip()
        s = s.replace("^", "**")

        # Replace x_k and r_k with tensor indexing (word boundaries!)
        s = re.sub(r"\bx_(\d+)\b", r"X[:, \1]", s)
        s = re.sub(r"\br_(\d+)\b", r"R[:, \1]", s)
        return s

    def _compile_expr(self, expr: str):
        """
        Returns a function f(X, R) -> (batch,) evaluating the expression with torch ops.
        Uses restricted eval namespace.
        """
        code = self._prepare_expr(expr)
        # Restricted globals, controlled locals at call-time
        def f(X, R):
            # ensure 2D
            assert X.ndim == 2 and R.ndim == 2, "X and R must be (batch, features)"
            return eval(code, {"__builtins__": {}}, {**self._allowed, "X": X, "R": R})
        return f

    def forward(self, X: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        batch = X.size(0)
        # Evaluate all basis functions -> (batch, n_funcs)
        feats = []
        for f in self._compiled:
            val = f(X, R)
            # ensure shape (batch,)
            if val.ndim == 2 and val.shape[1] == 1:
                val = val.squeeze(-1)
            feats.append(val)
        Phi = torch.stack(feats, dim=-1)  # (batch, n_funcs)

        # coefs shape to (1, n_funcs) for broadcast
        w = self.coefs.view(1, -1).to(Phi)
        u = (Phi * w).sum(dim=-1, keepdim=True)  # (batch, 1)
        return u

class UnifiedTwoPolicy(nn.Module):
    """
    Wrap two single-output policies into one 2-output policy:
      u1 = model_u1(X, R)
      u2 = model_u2(X, R)
      U  = clamp([u1, u2], umin, umax)

    Each sub-model must return shape (batch, 1).
    """
    def __init__(self, model_u1: nn.Module, model_u2: nn.Module, umin: float, umax: float):
        super().__init__()
        self.model_u1 = model_u1
        self.model_u2 = model_u2
        self.umin = float(umin)
        self.umax = float(umax)

    def forward(self, X: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        u1 = self.model_u1(X, R)   # (batch, 1)
        u2 = self.model_u2(X, R)   # (batch, 1)
        U = torch.cat([u1, u2], dim=1)  # (batch, 2)
        return torch.clamp(U, min=self.umin, max=self.umax)

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


def combined_controller_and_two_policy_heatmaps(
    Y_list, U_list,
    controller_labels,
    fx_policy_SINDy, fx_policy_WB,
    fx_policy_labels,
    R=None,
    style_map=None,
    TS_FIG_WIDTH=4, TS_FIG_HEIGHT=2,
    CELL_SIZE=0.3,
    FONT_SIZE_COEFS=5,
    FONT_SIZE_COEFS_LABELS=8,
    save_name="CombinedPlot.pdf",
    title=None,
    xmin=None, xmax=None,
    umin=None, umax=None
):
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    function_names_latex = convert_to_latex_policy(fx_policy_SINDy[0].library.function_names)
    n_terms = len(function_names_latex)
    heatmap_width = n_terms * CELL_SIZE
    heatmap_height = 0.8
    gap_between_heatmaps = 0.05
    gap_between_blocks = 0.01

    fig_height = TS_FIG_HEIGHT + 2 * heatmap_height + gap_between_heatmaps + gap_between_blocks
    fig = plt.figure(figsize=(TS_FIG_WIDTH, fig_height))

    nsteps = Y_list[0].shape[0] - 1
    time = np.arange(nsteps + 1)

    top_margin = 0.62
    gap = 0.01
    plot_height = 0.2
    plot_width = 0.8
    left_margin = (1 - plot_width) / 2
    heatmap_width_figunits = heatmap_width / TS_FIG_WIDTH
    heatmap_left = (1 - heatmap_width_figunits) / 2

    # === Time series axes ===
    # Shrink the width of each plot since now they are side-by-side
    plot_width = 0.38  # about half (0.38 + 0.38 + small gap)

    ax1 = fig.add_axes([left_margin, top_margin, plot_width, plot_height])
    gap_between_ts_plots = 0.07
    ax2 = fig.add_axes([left_margin + plot_width + gap_between_ts_plots, top_margin, plot_width, plot_height])

    # === Legend just above ax1 ===
    legend_ax = fig.add_axes([left_margin, top_margin + plot_height + 0.001, 2*plot_width + 0.04, 0.05])
    legend_ax.axis("off")

    legend_handles = []
    if R is not None:
        h_ref, = ax1.plot(time, R[:, 0], '--', linewidth=0.6, color='red', label="Ref")
        legend_handles.append(h_ref)

    for i in range(len(Y_list)):
        label = controller_labels[i]
        linestyle, linewidth, alpha, color = style_map[label]

        ax1.plot(time, Y_list[i][:, 0], linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=color)
        ax1.plot(time, Y_list[i][:, 1], linestyle="--", linewidth=linewidth, alpha=alpha, color=color)

        ax2.plot(time[:-1], U_list[i][:, 0], linestyle=linestyle, linewidth=linewidth, alpha=alpha,  color=color)
        ax2.plot(time[:-1], U_list[i][:, 1], linestyle="--", linewidth=linewidth, alpha=alpha, color=color)

        legend_handles.append(Line2D([0], [0], color=color, label=label, linewidth=1.3, linestyle=linestyle))

    legend_ax.legend(legend_handles, [h.get_label() for h in legend_handles],
                     loc='center', ncol=len(legend_handles), fontsize=12,
                     frameon=False, handlelength=2.5, columnspacing=1.5)

    # === Formatting plots ===
    if xmin is not None: ax1.hlines(xmin, xmin=0, xmax=nsteps, colors='black', linewidth=0.4, linestyle='--')
    if xmax is not None: ax1.hlines(xmax, xmin=0, xmax=nsteps, colors='black', linewidth=0.4, linestyle='--')
    if umin is not None: ax2.hlines(umin, xmin=0, xmax=nsteps, colors='black', linewidth=0.4, linestyle='--')
    if umax is not None: ax2.hlines(umax, xmin=0, xmax=nsteps, colors='black', linewidth=0.4, linestyle='--')

    ax1.set_ylabel(r"$x$", fontsize=15, rotation=0, labelpad=16)
    ax1.set_xlabel(r"$t$", fontsize=15, labelpad=8)
    ax2.set_ylabel(r"$u$", fontsize=15, rotation=0, labelpad=16)
    ax2.set_xlabel(r"$t$", fontsize=15, labelpad=8)

    ax1.set_xlim([0, nsteps])
    ax2.set_xlim([0, nsteps])
    ax1.tick_params(axis='both', labelsize=10)
    ax2.tick_params(axis='both', labelsize=10)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # === Heatmap (single) ===
    coef_SINDy = np.vstack([fx.coef.detach().cpu().numpy().T for fx in fx_policy_SINDy])

    ax3_bottom = 0.15 - gap_between_blocks
    ax3 = fig.add_axes([heatmap_left, ax3_bottom, heatmap_width_figunits, heatmap_height])

    sns_plot = sns.heatmap(
        coef_SINDy,
        annot=False,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        vmin=-4, vmax=4,
        square=True,
        linewidths=0.5,
        linecolor='black',
        xticklabels=function_names_latex,
        yticklabels=fx_policy_labels,
        cbar=False,  # we'll add a single external colorbar
        annot_kws={"size": FONT_SIZE_COEFS},
        ax=ax3
    )

    ax3.tick_params(axis='x', rotation=90, labelsize=FONT_SIZE_COEFS_LABELS, pad=1)
    ax3.tick_params(axis='y', rotation=0, labelsize=FONT_SIZE_COEFS_LABELS, pad=5)

    ax3.text(
        -2.5, coef_SINDy.shape[0]/2,
        "Sparse\nDictionary\nPolicy",
        va='center', ha='center',
        rotation=0,
        fontsize=9,
        transform=ax3.transData
    )

    ax3.add_patch(patches.Rectangle((0, 0), coef_SINDy.shape[1], coef_SINDy.shape[0],
                                    linewidth=0.75, edgecolor='black', facecolor='none',
                                    clip_on=False, zorder=10))

    # --- Colorbar for the single heatmap ---
    pos3 = ax3.get_position()
    cbar_ax = fig.add_axes([
        pos3.x1 + 0.01,  # just to the right of the heatmap
        pos3.y0,         # align vertically with the heatmap
        0.015,           # width
        pos3.height      # height
    ])
    cbar = plt.colorbar(sns_plot.collections[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6)

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)

    if save_name is not None:
        file_format = save_name.split('.')[-1]
        fig.savefig(save_name, format=file_format, bbox_inches="tight")

    plt.show()

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
