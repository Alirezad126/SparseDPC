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
from matplotlib.ticker import FuncFormatter

# A function to plot before and after updates
# expects a function `convert_to_latex_policy(names)` in scope

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
    umin=None, umax=None,
    ACTIVE_TOL=1e-8, VMIN=-6, VMAX=6
):
    # ---------- helpers ----------
    def _names_attr(lib):
        if hasattr(lib, 'function_names'): return 'function_names'
        if hasattr(lib, 'functions_names'): return 'functions_names'
        raise AttributeError("Library needs function_names/functions_names")
    def _get_names(m): return getattr(m.library, _names_attr(m.library))
    def _coef1xN(t):
        a = t.detach().cpu().numpy()
        if a.ndim == 2: a = a[:, 0]
        return a.reshape(1, -1)
    def _ordered_union(pols):
        seen, out = set(), []
        for p in pols:
            for n in _get_names(p):
                if n not in seen: seen.add(n); out.append(n)
        return out
    def _aligned_row(p, union):
        names = _get_names(p); idx = {n:i for i,n in enumerate(names)}
        row = np.zeros((1, len(union)))
        c = _coef1xN(p.coef)
        for j,n in enumerate(union):
            i = idx.get(n)
            if i is not None: row[0, j] = c[0, i]
        return row

    # --- helpers: bold borders ---
    def _bold_spines(ax, lw=0.5, color="black"):
        for side in ("left","right","top","bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(lw)
            ax.spines[side].set_edgecolor(color)

    def _latex_text(s: str) -> str:
        """Wrap plain text in LaTeX math mode with \\text{...}, preserving spaces."""
        if "$" in s:       # already LaTeX, don't double-wrap
            return s
        # escape LaTeX specials
        for k, v in {
            "\\": r"\textbackslash{}",
            "{": r"\{", "}": r"\}",
            "#": r"\#", "$": r"\$", "%": r"\%",
            "&": r"\&", "_": r"\_", "^": r"\^{}", "~": r"\~{}",
        }.items():
            s = s.replace(k, v)
        return rf"$\text{{{s}}}$"
        # put this near the top (you already set usetex=True)

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",   # needed for \text
    })


    # ---------- align + active mask (shared x-ticks) ----------
    union = _ordered_union(list(fx_policy_SINDy) + list(fx_policy_WB))
    coef_S_full = np.vstack([_aligned_row(fx, union) for fx in fx_policy_SINDy])
    coef_W_full = np.vstack([_aligned_row(fx, union) for fx in fx_policy_WB])

    active = (np.abs(coef_S_full).max(0) > ACTIVE_TOL) | (np.abs(coef_W_full).max(0) > ACTIVE_TOL)
    if not active.any():
        active[np.argmax(np.abs(np.r_[coef_S_full, coef_W_full]).max(0))] = True

    coef_S, coef_W = coef_S_full[:, active], coef_W_full[:, active]
    function_names_latex = [n for n,k in zip(convert_to_latex_policy(union), active) if k]
    n_terms = len(function_names_latex)

    # ---------- sizing ----------
    heatmap_width = n_terms * CELL_SIZE
    heatmap_height = 0.8
    gap_between_heatmaps = 0.01
    gap_between_blocks   = 0.01

    fig_height = TS_FIG_HEIGHT + 2*heatmap_height + gap_between_heatmaps + gap_between_blocks
    fig = plt.figure(figsize=(TS_FIG_WIDTH, fig_height))

    # ---------- time series (unchanged layout) ----------
    nsteps = Y_list[0].shape[0] - 1
    time = np.arange(nsteps + 1)

    top_margin = 0.62
    plot_height = 0.2
    plot_width = 0.8
    left_margin = (1 - plot_width) / 2
    heatmap_width_figunits = heatmap_width / TS_FIG_WIDTH

    plot_width = 0.38
    ax1 = fig.add_axes([left_margin, top_margin, plot_width, plot_height])
    gap_between_ts_plots = 0.07
    ax2 = fig.add_axes([left_margin + plot_width + gap_between_ts_plots, top_margin, plot_width, plot_height])

    legend_ax = fig.add_axes([left_margin, top_margin + plot_height + 0.001, 2*plot_width + 0.04, 0.05])
    legend_ax.axis("off")

    legend_handles = []
    if R is not None:
        h_ref, = ax1.plot(time, R[:, 0], '--', linewidth=0.7, color='red', label=r"$\mathrm{Ref}$")
        legend_handles.append(h_ref)

    for i in range(len(Y_list)):
        label = controller_labels[i]
        cfg = style_map[label]

        # Accept dict or tuple:
        if isinstance(cfg, dict):
            linestyle = cfg.get("linestyle", "-")
            linewidth = cfg.get("linewidth", 1.5)
            alpha     = cfg.get("alpha", 1.0)
            color     = cfg.get("color", plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10])
        else:
            # tuple formats: (linestyle, linewidth, alpha, color?) or (linestyle, linewidth, alpha)
            if len(cfg) >= 4:
                linestyle, linewidth, alpha, color = cfg[:4]
            else:
                linestyle, linewidth, alpha = cfg[:3]
                color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]

        # plot with the color from style_map
        ax1.plot(time, Y_list[i][:, 0], linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=color)
        ax1.plot(time, Y_list[i][:, 1], linestyle="--", linewidth=linewidth, alpha=alpha, color=color)

        for k in range(U_list[i].shape[1]):
            ax2.plot(time[:-1], U_list[i][:, k], linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=color)

        legend_handles.append(Line2D(
            [0], [0],
            color=color, linewidth=1.3, linestyle=linestyle,
            label=_latex_text(label)
        ))

    legend_ax.legend(
        legend_handles, [h.get_label() for h in legend_handles],
        loc='center', ncol=len(legend_handles), fontsize=12,
        frameon=False, handlelength=2.5, columnspacing=1.5
    )

    #if xmin is not None: ax1.hlines(xmin, 0, nsteps, colors='black', linewidth=0.4, linestyle='--')
    #if xmax is not None: ax1.hlines(xmax, 0, nsteps, colors='black', linewidth=0.4, linestyle='--')
    if umin is not None: ax2.hlines(umin, 0, nsteps, colors='black', linewidth=0.4, linestyle='--')
    if umax is not None: ax2.hlines(umax, 0, nsteps, colors='black', linewidth=0.4, linestyle='--')
    ax1.set_ylabel(r"$x$", fontsize=15, rotation=0, labelpad=16); ax1.set_xlabel(r"$t$", fontsize=15, labelpad=8)
    ax2.set_ylabel(r"$u$", fontsize=15, rotation=0, labelpad=16); ax2.set_xlabel(r"$t$", fontsize=15, labelpad=8)
    ax1.set_xlim([0, nsteps]); ax2.set_xlim([0, nsteps])
    ax1.set_ylim([xmin, xmax])
    ax1.tick_params(axis='both', labelsize=10); ax2.tick_params(axis='both', labelsize=10)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    _bold_spines(ax1, lw=0.8)
    _bold_spines(ax2, lw=0.8)
    # ---------- heatmaps directly under TS (side-by-side, aligned to TS) ----------
    hm_w = heatmap_width / TS_FIG_WIDTH
    hm_h = heatmap_height / fig_height

    V_GAP = 0.05        # vertical gap under TS row (figure units)
    BOTTOM_SHIFT = 0.02 # extra downward shift (figure units)
    SHRINK_GAP_FRAC = 0.1  # 0..1 pull the two heatmaps toward each other

    bbox1 = ax1.get_position()
    bbox2 = ax2.get_position()
    c1 = bbox1.x0 + bbox1.width / 2.0
    c2 = bbox2.x0 + bbox2.width / 2.0
    hm_bottom = min(bbox1.y0, bbox2.y0) - V_GAP - hm_h - BOTTOM_SHIFT
    gap_current = (c2 - c1) - hm_w
    pull = max(0.0, SHRINK_GAP_FRAC * gap_current / 2.0)

    ax3_left = c1 - hm_w/2.0 + pull
    ax4_left = c2 - hm_w/2.0 - pull
    ax3 = fig.add_axes([ax3_left, hm_bottom, hm_w, hm_h])  # left heatmap (SINDy)
    ax4 = fig.add_axes([ax4_left, hm_bottom, hm_w, hm_h])  # right heatmap (WB)

    # ---------- heatmaps (rectangular cells + LaTeX annotations) ----------
    DECIMALS = 3
    CELL_ASPECT = 0.6
    CELL_BORDER = 0.5
    LABEL_PAD = 30

    sns_plots = []
    for coef, ax, label in [(coef_S, ax3, r"$\mathrm{Before}$"),
                            (coef_W, ax4, r"$\mathrm{After}$")]:
        annot_str = np.array([[rf"${v:.{DECIMALS}f}$" for v in row] for row in coef])
        hm = sns.heatmap(
            coef,
            annot=annot_str, fmt="",
            cmap="RdBu_r", center=0, vmin=VMIN, vmax=VMAX,
            square=False, linewidths=CELL_BORDER, linecolor='black',
            xticklabels=function_names_latex, yticklabels=fx_policy_labels,
            cbar=False, annot_kws={"size": FONT_SIZE_COEFS}, ax=ax
        )
        sns_plots.append(hm)
        ax.set_aspect(CELL_ASPECT)
        ax.tick_params(axis='x', rotation=90, labelsize=FONT_SIZE_COEFS_LABELS, pad=1)
        ax.tick_params(axis='y', rotation=0,  labelsize=FONT_SIZE_COEFS_LABELS, pad=5)
        ax.add_patch(patches.Rectangle((0,0), coef.shape[1], coef.shape[0],
                                       fill=False, ec='black', lw=0.75, zorder=10, clip_on=False))
        ax.annotate(label, xy=(0, 0.5), xycoords='axes fraction',
                    xytext=(-LABEL_PAD, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=12, annotation_clip=False)

    # shared colorbar (to the right of the right heatmap) with LaTeX ticks
    posR = ax4.get_position()
    cbar_ax = fig.add_axes([posR.x1 + 0.01, posR.y0, 0.015, posR.height])
    cbar = plt.colorbar(sns_plots[0].collections[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: rf"${v:.{DECIMALS}f}$"))
    cbar.update_ticks()

    if title: fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    if save_name:
        file_format = save_name.split('.')[-1]
        fig.savefig(save_name, format=file_format, bbox_inches="tight")
    plt.show()
