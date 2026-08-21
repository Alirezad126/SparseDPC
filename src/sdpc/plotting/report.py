"""Report plots and tables: barrier evolution, coefficient heatmaps, metric tables.

These support the "safety-barrier evolution" and cross-method comparison figures, and turn
the aggregated eval metrics into a readable table.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

__all__ = ["plot_barrier_evolution", "plot_coef_heatmap", "format_metrics_table"]


def plot_barrier_evolution(logs: List[Dict], *, title: str = "Safety-barrier evolution"):
    """Plot barrier loss, min predicted margin, and #safety iterations over the run."""
    import matplotlib.pyplot as plt

    t = [l["t"] for l in logs]
    barrier = [l.get("barrier_loss", float("nan")) for l in logs]
    margin = [l.get("min_pred_margin", float("nan")) for l in logs]
    iters = [l.get("safety_iters", 0) for l in logs]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, barrier, color="crimson"); axes[0].set_ylabel(r"$J_h$")
    axes[0].set_title(title); axes[0].grid(True, ls="--", alpha=0.5)
    axes[1].plot(t, margin, color="steelblue"); axes[1].axhline(0.0, color="k", ls="--", alpha=0.5)
    axes[1].set_ylabel("min margin"); axes[1].grid(True, ls="--", alpha=0.5)
    axes[2].bar(t, iters, color="seagreen"); axes[2].set_ylabel("#safety iters")
    axes[2].set_xlabel("time step"); axes[2].grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    return fig, axes


def plot_coef_heatmap(policy, *, title: str = "Active policy coefficients"):
    """Heatmap of the (dense) sparse-policy coefficient matrix with term labels."""
    import matplotlib.pyplot as plt

    C = policy.coef.detach().cpu().numpy()  # (n_terms, n_out)
    names = policy.function_names
    active = np.where(np.abs(C).sum(axis=1) > 0)[0]
    C = C[active]
    labels = [names[i] for i in active]

    fig, ax = plt.subplots(figsize=(4, max(3, 0.3 * len(active))))
    im = ax.imshow(C, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(C).max(), vmax=np.abs(C).max())
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    out_names = policy.policy_name or [f"u{j}" for j in range(policy.n_out)]
    ax.set_xticks(range(policy.n_out)); ax.set_xticklabels(out_names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig, ax


def format_metrics_table(summary: Dict[str, Dict[str, float]],
                         metrics: List[str] = None) -> str:
    """Render the aggregated metric summary as a plain-text table (mean +/- std)."""
    metrics = metrics or [
        "final_tracking_mse", "steps_to_reach", "total_control_action_l1",
        "control_smoothness_rms", "num_violations", "min_safety_margin",
        "runtime_per_step_s", "sparsity_total_terms",
    ]
    header = ["method"] + metrics
    widths = [max(len(h), 16) for h in header]
    lines = [" | ".join(h.ljust(w) for h, w in zip(header, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for method, s in summary.items():
        cells = [method.ljust(widths[0])]
        for k, w in zip(metrics, widths[1:]):
            mean, std = s.get(f"{k}_mean"), s.get(f"{k}_std")
            cells.append(("--" if mean is None else f"{mean:.3g}+-{std:.2g}").ljust(w))
        lines.append(" | ".join(cells))
    return "\n".join(lines)
