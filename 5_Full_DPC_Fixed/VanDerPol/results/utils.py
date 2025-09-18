import re
import casadi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib import patches
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import pandas as pd
import seaborn as sns
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

def prune_terms_by_name(sindy_model, substring="r_0"):
    """
    Remove all library functions (and corresponding coefficient rows)
    whose name contains `substring`. Operates in-place.

    Returns: list of pruned function names.
    """
    lib = sindy_model.library
    names = list(lib.function_names) if lib.function_names is not None else None
    if names is None:
        print("No function names available on the library; nothing to prune.")
        return []

    # indices to keep / drop
    keep_idx = [i for i, n in enumerate(names) if substring not in n]
    drop_idx = [i for i, n in enumerate(names) if substring in n]

    if not drop_idx:
        print(f"No terms matched substring '{substring}'.")
        return []

    # Safety: don't delete everything
    if len(keep_idx) == 0:
        raise RuntimeError(f"Pruning '{substring}' would remove ALL terms; aborting.")

    # prune coefficients
    device = sindy_model.coef.device
    with torch.no_grad():
        new_coef = sindy_model.coef.detach().clone()[keep_idx, :]
    sindy_model.coef = torch.nn.Parameter(new_coef.to(device), requires_grad=True)

    # prune library callables + names + shape
    lib.library        = [lib.library[i]        for i in keep_idx]
    lib.function_names = [lib.function_names[i] for i in keep_idx]
    lib.shape          = (len(keep_idx), lib.shape[1])

    # if the library had an active mask, drop it (now stale)
    if hasattr(lib, "active_idx"):
        delattr(lib, "active_idx")

    pruned_names = [names[i] for i in drop_idx]
    #print(f"Pruned {len(pruned_names)} terms containing '{substring}': {pruned_names}")
    return sindy_model

def vdp_mpc_solve(data, vdp_system, N=50, nsim=100, Rdu=0.001):
    """
    MPC solver for the forced Van der Pol system driving to the origin.

    Args:
        data: Dictionary with keys:
            'xn': initial state, torch tensor of shape (batch, time, 2)
        vdp_system: VanDerPolControl object (PyTorch) with attribute mu
        N: MPC horizon length
        nsim: number of simulation steps

    Returns:
        Dictionary containing state trajectory, control inputs, and solve times.
    """

    # Extract initial condition
    x0_init = data['xn'][0, 0].detach().cpu().numpy()

    # System parameters
    mu = vdp_system.mu
    dt = vdp_system.ts if hasattr(vdp_system, 'ts') else 0.1  # default dt if not defined

    # System dimensions
    nx, nu = 2, 1

    # Bounds
    umin, umax = -5., 5.
    xmin, xmax = -5., 5.

    # Define system dynamics
    def ode_equations(x, u):
        dx1 = x[1]
        dx2 = mu * (1 - x[0]**2) * x[1] - x[0] + u
        return casadi.vertcat(dx1, dx2)

    # Set up MPC optimization
    opti = casadi.Opti()
    X = opti.variable(nx, N+1)
    U = opti.variable(N)
    x0_param = opti.parameter(nx)

    # Constraints
    opti.subject_to(opti.bounded(umin, U, umax))
    opti.subject_to(opti.bounded(xmin, X, xmax))
    opti.subject_to(X[:, 0] == x0_param)

    # Objective (drive x to 0)
    cost = 0
    for i in range(N):
        cost += casadi.sumsqr(X[:, i])
        opti.subject_to(X[:, i+1] == X[:, i] + dt * ode_equations(X[:, i], U[i]))
        if i > 0:
            cost += Rdu * casadi.sumsqr(U[i] - U[i - 1])

    opti.minimize(cost)

    opti.solver('ipopt', {
    'ipopt.print_level': 0,    # 0 = no IPOPT output
    'print_time': False        # don’t print total solve time
    })

    # Storage
    Xs = [x0_init]
    Us = []
    times = []

    # Simulation
    x0 = x0_init.copy()
    for k in range(nsim):
        opti.set_value(x0_param, x0)
        start_time = time.time()
        sol = opti.solve()
        elapsed_time = time.time() - start_time

        u_val = sol.value(U)[0]
        x0 = x0 + ode_equations(x0, u_val).full().flatten() * dt

        Xs.append(x0)
        Us.append(u_val)
        times.append(elapsed_time)

    results = {
        'xn': torch.tensor(np.expand_dims(np.array(Xs), axis=0), dtype=torch.float32),
        'u': torch.tensor(np.expand_dims(np.array(Us), axis=0), dtype=torch.float32),
        'times': np.array(times)
    }
    return results


def run_multi_seed_rollouts(
    systems,                  # dict: {"SD_SINDy": SD_SINDy_system, "NN_WB": NN_WB_system, ...}
    fx_policy_SINDy,          # list of policy modules (for the heatmap, unchanged)
    nsteps,
    nx, nu, nref,
    xmin, xmax, umin, umax,
    device,
    n_seeds=5,
    seed0=5,
    include_mpc=False,
    mpc_fn=None,              # e.g. vdp_mpc_solve
    mpc_kwargs=None           # dict passed to mpc_fn
):
    """
    Returns (Y_list, U_list, controller_labels, R, losses)
    where each Y_list[i] has shape (nsteps+1, nx * n_seeds) and each U_list[i] has shape (nsteps, nu * n_seeds).
    """
    if mpc_kwargs is None: mpc_kwargs = {}

    # Set horizons once
    for sys in systems.values():
        sys.nsteps = nsteps

    # Accumulators per controller
    Y_cols = {name: [] for name in systems.keys()}
    U_cols = {name: [] for name in systems.keys()}
    losses = {name: [] for name in systems.keys()}

    # Optional MPC
    if include_mpc:
        Y_cols["MPC"] = []
        U_cols["MPC"] = []
        losses["MPC"] = []

    # Same zero reference you used
    R_torch = torch.zeros(1, nsteps+1, nx, dtype=torch.float32, device=device)

    # Loop seeds
    for s in range(seed0, seed0 + n_seeds):
        torch.manual_seed(s)

        # Random initial condition in [xmin, xmax]
        xn0 = torch.rand(1, 1, nx, dtype=torch.float32) * (xmax - xmin) + xmin

        data = {
            'xn':   xn0.to(device),
            'x1_n': xn0[:, :, 0:1].to(device),
            'x2_n': xn0[:, :, 1:2].to(device),
            'r':    R_torch.clone()
        }

        # Run each controller
        for name, sys in systems.items():
            out = sys(data)
            Y_cols[name].append(out['xn'].detach().cpu().reshape(nsteps + 1, nx))
            U_cols[name].append(out['u'].detach().cpu().reshape(nsteps, nu))
            # tracking to zero (same as your snippet)
            losses[name].append(F.mse_loss(out['xn'], torch.zeros_like(out['xn'])).item())

        # MPC (optional)
        if include_mpc:
            mpc_out = mpc_fn(data, **mpc_kwargs)
            Y_cols["MPC"].append(mpc_out['xn'].detach().cpu().reshape(nsteps + 1, nx))
            U_cols["MPC"].append(mpc_out['u'].detach().cpu().reshape(nsteps, nu))
            losses["MPC"].append(F.mse_loss(mpc_out['xn'], torch.zeros_like(mpc_out['xn'])).item())

    # Stack per controller across seeds along feature axis so plotter draws many lines with same color
    controller_labels = list(systems.keys())
    if include_mpc:
        controller_labels += ["MPC"]

    Y_list = []
    U_list = []
    for name in controller_labels:
        Y = np.concatenate([y.numpy() for y in Y_cols[name]], axis=1)  # (nsteps+1, nx * n_seeds)
        U = np.concatenate([u.numpy() for u in U_cols[name]], axis=1)  # (nsteps,   nu * n_seeds)
        Y_list.append(Y)
        U_list.append(U)

    # Reference for the plotter
    R_np = R_torch.detach().cpu().reshape(nsteps + 1, nref).numpy()

    # Package losses in same order as controller_labels
    losses_vec = [float(np.mean(losses[name])) for name in controller_labels]

    return Y_list, U_list, controller_labels, fx_policy_SINDy, R_np, losses_vec

def plot_multi_seed_summary(
    Y_list, U_list, controller_labels, R,
    nx, nu, n_seeds,
    style_map,
    xmin=None, xmax=None, umin=None, umax=None,
    band_quantiles=(0.10, 0.90),
    agg="median",
    highlight_label="SD_SINDy",   # emphasize your sparse policy
    show_samples=0,
    TS_FIG_WIDTH=7.0, TS_FIG_HEIGHT=4.2,
    title=None, save_name=None,
    # NEW: controls for bold band edges
    band_edge_color='auto',       # 'auto' or a color like 'black'
    band_edge_lw_add=1.2,         # how much thicker than center line
    band_edge_alpha=0.9
):
    def _reshape_concat_to_tensor(arr, nfeat, n_seeds):
        T = arr.shape[0]
        return arr.reshape(T, n_seeds, nfeat).transpose(0, 2, 1)  # (T, nfeat, n_seeds)

    def _summary_curves(A, q_low=0.10, q_high=0.90, agg="median"):
        if agg == "median":
            center = np.median(A, axis=-1)
        elif agg == "mean":
            center = A.mean(axis=-1)
        else:
            raise ValueError("agg must be 'median' or 'mean'")
        low = np.quantile(A, q_low, axis=-1)
        high = np.quantile(A, q_high, axis=-1)
        return {"center": center, "low": low, "high": high}

    def _plot_band_with_outline(ax, x, low, high, center, *,
                                color, ls, lw_center, z,
                                alpha_fill, alpha_center,
                                edge_color='auto', edge_lw_add=1.2, edge_alpha=0.9):
        """
        Draws a shaded band between low/high with bold outlines and a center line.
        edge_color='auto' -> uses the fill color; else use string like 'black'.
        """
        ax.fill_between(x, low, high, color=color, alpha=alpha_fill, zorder=z - 1)
        ec = color if edge_color == 'auto' else edge_color
        ax.plot(x, low, color=ec, lw=lw_center + edge_lw_add, alpha=edge_alpha, zorder=z)
        ax.plot(x, high, color=ec, lw=lw_center + edge_lw_add, alpha=edge_alpha, zorder=z)
        ax.plot(x, center, ls=ls, lw=lw_center, color=color, alpha=alpha_center, zorder=z)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    T = Y_list[0].shape[0]
    time = np.arange(T)

    fig = plt.figure(figsize=(TS_FIG_WIDTH, TS_FIG_HEIGHT))
    left, top = 0.08, 0.90
    w, h = 0.38, 0.26         # each subplot height
    gap_x = 0.10
    gap_y = 0.08

    # Axes: states, controls, control rate
    ax_x  = fig.add_axes([left,               top - h,           w, h])
    ax_u  = fig.add_axes([left + w + gap_x,   top - h,           w, h])
    ax_du = fig.add_axes([left + w + gap_x,   top - 2*h - gap_y, w, h])  # Δu below controls
    leg_ax = fig.add_axes([left, top + 0.01, w*2 + gap_x, 0.07]); leg_ax.axis("off")

    # Optional reference on states
    if R is not None and R.ndim == 2 and R.shape[0] == T:
        ax_x.plot(time, R[:, 0], linestyle=":", color="red", linewidth=1.2, label="Ref")

    # Shade admissible regions (with visible boundaries)
    if (xmin is not None) and (xmax is not None):
        ax_x.axhspan(xmin, xmax, color="0.92", zorder=0)
        ax_x.axhline(xmin, color='black', lw=1.4, ls='--', zorder=1)
        ax_x.axhline(xmax, color='black', lw=1.4, ls='--', zorder=1)
    if (umin is not None) and (umax is not None):
        ax_u.axhspan(umin, umax, color="0.92", zorder=0)
        ax_u.axhline(umin, color='black', lw=1.4, ls='--', zorder=1)
        ax_u.axhline(umax, color='black', lw=1.4, ls='--', zorder=1)

    # Plot each controller
    handles, labels = [], []
    for i, label in enumerate(controller_labels):
        ls, lw_base, alpha, color = style_map[label]
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]

        # Emphasize the sparse policy
        if label == highlight_label:
            lw = lw_base + 0.8
            band_alpha = 0.22
            center_alpha = 1.0
            z = 5
        else:
            lw = lw_base + 0.2
            band_alpha = 0.10
            center_alpha = 0.95
            z = 3

        # Y: (T, nx, n_seeds)
        Y = _reshape_concat_to_tensor(Y_list[i], nx, n_seeds)
        Ysum = _summary_curves(Y, q_low=band_quantiles[0], q_high=band_quantiles[1], agg=agg)
        for s in range(nx):
            _plot_band_with_outline(
                ax_x, time, Ysum["low"][:, s], Ysum["high"][:, s], Ysum["center"][:, s],
                color=color, ls=ls, lw_center=lw, z=z,
                alpha_fill=band_alpha, alpha_center=center_alpha,
                edge_color=band_edge_color, edge_lw_add=band_edge_lw_add, edge_alpha=band_edge_alpha
            )

        if show_samples > 0:
            idx = np.linspace(0, n_seeds-1, min(show_samples, n_seeds)).astype(int)
            for s in range(nx):
                ax_x.plot(time, Y[:, s, idx], color=color, alpha=0.12, lw=0.6, zorder=1)

        # U: (T-1, nu, n_seeds)
        U = _reshape_concat_to_tensor(U_list[i], nu, n_seeds)
        Usum = _summary_curves(U, q_low=band_quantiles[0], q_high=band_quantiles[1], agg=agg)
        for a in range(nu):
            _plot_band_with_outline(
                ax_u, time[:-1], Usum["low"][:, a], Usum["high"][:, a], Usum["center"][:, a],
                color=color, ls=ls, lw_center=lw, z=z,
                alpha_fill=band_alpha, alpha_center=center_alpha,
                edge_color=band_edge_color, edge_lw_add=band_edge_lw_add, edge_alpha=band_edge_alpha
            )

        # Δu (smoothness): |u_t - u_{t-1}|, summarize
        dU = np.abs(np.diff(U, axis=0))                # (T-2, nu, n_seeds)
        dUsum = _summary_curves(dU, q_low=band_quantiles[0], q_high=band_quantiles[1], agg=agg)
        for a in range(nu):
            _plot_band_with_outline(
                ax_du, time[1:-1], dUsum["low"][:, a], dUsum["high"][:, a], dUsum["center"][:, a],
                color=color, ls=ls, lw_center=lw, z=z,
                alpha_fill=band_alpha, alpha_center=center_alpha,
                edge_color=band_edge_color, edge_lw_add=band_edge_lw_add, edge_alpha=band_edge_alpha
            )

        # legend handle
        handles.append(plt.Line2D([0],[0], color=color, linestyle=ls, linewidth=lw))
        labels.append(label)

    # Legend (add Ref if drawn)
    if R is not None:
        handles = [plt.Line2D([0],[0], color="red", linestyle=":")] + handles
        labels  = ["Ref"] + labels
    leg_ax.legend(handles, labels, loc="center", ncol=len(labels), frameon=False, fontsize=11)

    # Cosmetics
    ax_x.set_ylabel(r"$x$", fontsize=13, rotation=0, labelpad=12)
    ax_u.set_ylabel(r"$u$", fontsize=13, rotation=0, labelpad=12)
    ax_du.set_ylabel(r"$|\Delta u|$", fontsize=13, rotation=0, labelpad=12)

    ax_x.set_xlabel(r"$t$", fontsize=12)
    ax_u.set_xlabel(r"$t$", fontsize=12)
    ax_du.set_xlabel(r"$t$", fontsize=12)

    ax_x.set_xlim([0, T-1]); ax_u.set_xlim([0, T-1]); ax_du.set_xlim([0, T-1])
    for ax in (ax_x, ax_u, ax_du):
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    if title:
        fig.suptitle(title, fontsize=14, y=0.99)

    if save_name:
        fig.savefig(save_name, bbox_inches="tight", dpi=300)

    plt.show()

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
    gap_between_blocks = 0.05

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

        legend_handles.append(Line2D([0], [0], color=color, label=label, linewidth=1.3, linestyle=linestyle))

    legend_ax.legend(legend_handles, [h.get_label() for h in legend_handles],
                     loc='center', ncol=len(legend_handles), fontsize=12,
                     frameon=False, handlelength=2.5, columnspacing=1.5)

    # === Formatting plots ===
    #if xmin is not None: ax1.hlines(xmin, xmin=0, xmax=nsteps, colors='black', linewidth=0.4, linestyle='--')
    #if xmax is not None: ax1.hlines(xmax, xmin=0, xmax=nsteps, colors='black', linewidth=0.4, linestyle='--')
    if umin is not None: ax2.hlines(umin, xmin=0, xmax=nsteps, colors='black', linewidth=0.6, linestyle='--')
    if umax is not None: ax2.hlines(umax, xmin=0, xmax=nsteps, colors='black', linewidth=0.6, linestyle='--')

    ax1.set_ylabel(r"$x$", fontsize=15, rotation=0, labelpad=16)
    ax1.set_xlabel(r"$t$", fontsize=15, labelpad=8)
    ax2.set_ylabel(r"$u$", fontsize=15, rotation=0, labelpad=16)
    ax2.set_xlabel(r"$t$", fontsize=15, labelpad=8)

    ax1.set_xlim([0, nsteps])
    ax2.set_xlim([0, nsteps])
    ax1.tick_params(axis='both', labelsize=10)
    ax2.tick_params(axis='both', labelsize=10)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(2))
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
        vmin=-5, vmax=5,
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


def evaluate_controllers_over_seeds(
        systems_dict,
        nsteps,
        nx,
        xmin=-3.5,
        xmax=3.5,
        n_seeds=200,
        torch_device=None,
        *,
        # --- MPC hooks (optional) ---
        mpc_solver=None,  # e.g., mpc_solve_adaptive_horizon_rebuild
        mpc_model=None,  # e.g., gt_model (Two-Tank params)
        mpc_params=None  # dict of kwargs for mpc_solver (e.g., N_max, N_min, Qy, Ru, Rdu, Rdu_cross, etc.)
):
    """
    Evaluates controllers over multiple seeds.
    - If a controller name starts with 'MPC', this will call `mpc_solver(data, mpc_model, nsim=nsteps, step_length=step_length, **mpc_params)`.
    - Otherwise it calls the controller object directly: cl_system(data).

    Notes:
      * Assumes helpers `_make_ref`, `_extract_states`, `_tracking_mse`, `_mean_violation` exist.
      * For the MPC 'rebuild on N_k' implementation, the reference only needs shape (1, nsteps+1, nx).
    """

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
        diff = x_traj - r_traj
        return float(np.mean(diff ** 2))

    def _mean_violation(x_traj, xmin, xmax):
        below = np.maximum(0.0, xmin - x_traj)
        above = np.maximum(0.0, x_traj - xmax)
        return float(np.mean(below + above))

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
                print(f"  Seed {seed + 1}/{n_seeds}...")

            torch.manual_seed(seed)

            # Initial state and reference
            x0 = torch.rand(1, 1, nx, dtype=torch.float32)
            R = torch.zeros(1, nsteps + 1, nx, dtype=torch.float32, device=device)

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
                call_kwargs = {'nsim': nsteps}
                call_kwargs.update(mpc_params)
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
