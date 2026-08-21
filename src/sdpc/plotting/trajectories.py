"""Trajectory, obstacle, state, and control plotting helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

__all__ = [
    "plot_trajectories_with_ellipse",
    "plot_training_sample_with_ellipse",
    "make_test_sample_discrete",
    "make_test_sample",
    "plot_states_and_controls",
    "plot_trajectories_with_obstacles",
    "plot_controller_evaluation",
]


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _sample_outside_rotated_ellipse(rng, rect, ellipse, *, max_tries=200_000):
    (x0, y0), (x1, y1) = rect
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("sampling rectangles must have positive width and height")

    p = float(ellipse["p"])
    b = float(ellipse["b"])
    c = float(ellipse["c"])
    d = float(ellipse["d"])
    theta = float(ellipse.get("theta", 0.0))
    if b <= 0.0 or p <= 0.0:
        raise ValueError("ellipse parameters require b > 0 and p > 0")
    ct, st = np.cos(theta), np.sin(theta)
    radius_sq = (0.5 * p) ** 2

    for _ in range(0, max_tries, 4096):
        size = min(4096, max_tries)
        xs = rng.uniform(x0, x1, size=size)
        ys = rng.uniform(y0, y1, size=size)
        dx, dy = xs - c, ys - d
        xr = ct * dx + st * dy
        yr = -st * dx + ct * dy
        valid = b * xr ** 2 + yr ** 2 >= radius_sq
        if valid.any():
            idx = int(np.flatnonzero(valid)[0])
            return float(xs[idx]), float(ys[idx])
    raise RuntimeError("could not sample a point outside the obstacle")


def make_test_sample_discrete(
    *,
    nx,
    nsteps,
    device,
    ellipse,
    seed=0,
    init_rect=((-1.0, -1.0), (1.0, 1.0)),
    ref_rect=((-1.0, -1.0), (1.0, 1.0)),
):
    """Sample a deterministic initial state and constant reference outside an ellipse."""
    if nx != 2:
        raise ValueError("the discrete obstacle sample expects the state [x, y]")
    rng = np.random.default_rng(seed)
    x0, y0 = _sample_outside_rotated_ellipse(rng, init_rect, ellipse)
    xr, yr = _sample_outside_rotated_ellipse(rng, ref_rect, ellipse)
    xn = torch.tensor([[[x0, y0]]], dtype=torch.float32, device=device)
    ref = torch.tensor([xr, yr], dtype=torch.float32, device=device)
    return {"xn": xn, "r": ref.reshape(1, 1, 2).repeat(1, nsteps + 1, 1)}


def make_test_sample(
    *,
    nx,
    nsteps,
    device,
    ellipse,
    seed=7,
    init_vel_std=0.0,
    init_rect=((-1.0, -1.0), (1.0, 1.0)),
    ref_rect=((-1.0, -1.0), (1.0, 1.0)),
):
    """Compatibility wrapper for the two-state continuous obstacle sample."""
    del init_vel_std
    return make_test_sample_discrete(
        nx=nx,
        nsteps=nsteps,
        device=device,
        ellipse=ellipse,
        seed=seed,
        init_rect=init_rect,
        ref_rect=ref_rect,
    )


def plot_training_sample_with_ellipse(
    loader,
    sample_idx=0,
    pos_idx=(0, 2),
    *,
    p=None,
    b=None,
    c=None,
    d=None,
    theta=None,
    title="Training sample with obstacle",
    init_rect=None,
    ref_rect=None,
    xmin=-1.0,
    xmax=1.0,
):
    """Plot one policy-training initial/reference pair and its keep-out ellipse."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle

    batch = next(iter(loader))
    xn, ref = _np(batch["xn"]), _np(batch["r"])

    def _parameter(name, supplied, default=None):
        if supplied is not None:
            return float(supplied)
        if name in batch:
            return float(_np(batch[name][sample_idx]).reshape(-1)[0])
        return default

    obstacle = {
        "p": _parameter("obs_p", p),
        "b": _parameter("obs_b", b),
        "c": _parameter("obs_c", c),
        "d": _parameter("obs_d", d),
        "theta": _parameter("obs_theta", theta, 0.0),
    }
    if any(obstacle[key] is None for key in ("p", "b", "c", "d")):
        raise ValueError("ellipse parameters must be supplied or present in the batch")

    fig, ax = plt.subplots(figsize=(6, 6))
    x0 = xn[sample_idx, 0, list(pos_idx)]
    target = ref[sample_idx, 0, :2]
    ax.scatter(*x0, marker="o", facecolor="white", edgecolor="black", label="initial")
    ax.scatter(*target, marker="*", color="black", s=70, label="reference")
    ax.plot([x0[0], target[0]], [x0[1], target[1]], linestyle=":", color="0.45")

    width = float(obstacle["p"]) / np.sqrt(float(obstacle["b"]))
    height = float(obstacle["p"])
    ax.add_patch(Ellipse(
        (obstacle["c"], obstacle["d"]), width, height,
        angle=np.degrees(obstacle["theta"]), facecolor="tab:purple",
        edgecolor="tab:purple", alpha=0.18, hatch="////",
    ))
    for rect, color, label in (
        (init_rect, "tab:blue", "initial region"),
        (ref_rect, "tab:orange", "reference region"),
    ):
        if rect is None:
            continue
        (rx0, ry0), (rx1, ry1) = rect
        rx0, rx1 = sorted((rx0, rx1))
        ry0, ry1 = sorted((ry0, ry1))
        ax.add_patch(Rectangle(
            (rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False,
            edgecolor=color, linestyle="--", label=label,
        ))
    ax.set(xlim=(xmin, xmax), ylim=(xmin, xmax), xlabel=r"$x_1$", ylabel=r"$x_2$", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_trajectories_with_ellipse(
    traj_list,
    r,
    *,
    p,
    b,
    c,
    d,
    theta=0.0,
    batch_idx=0,
    xmin=-1.0,
    xmax=1.0,
    title="Closed-loop trajectories with obstacle",
    show_ref_path=True,
    show_u_plot=False,
):
    """Plot trajectories against one rotated keep-out ellipse."""
    del show_ref_path
    fig, ax = plot_trajectories_with_obstacles(
        traj_list,
        [{"p": p, "b": b, "c": c, "d": d, "theta": theta}],
        r,
        batch_idx=batch_idx,
        xmin=xmin,
        xmax=xmax,
        title=title,
    )
    if show_u_plot:
        import matplotlib.pyplot as plt

        plt.close(fig)
        fig, (ax, ax_u) = plt.subplots(1, 2, figsize=(11, 5))
        obstacle = {"p": p, "b": b, "c": c, "d": d, "theta": theta}
        _draw_obstacle_scene(
            ax, [obstacle], obstacle_band=0.0, regions=None, publication=False,
            xmin=xmin, xmax=xmax, bound_lo=None, bound_hi=None,
        )
        for item in traj_list:
            x = _np(item.get("traj", item.get("x")))[batch_idx]
            color = item.get("color")
            label = item.get("label", "trajectory")
            ax.plot(x[:, 0], x[:, 1], color=color, linestyle=item.get("linestyle", "-"), label=label)
            if "u" in item:
                ax_u.plot(_np(item["u"])[batch_idx], color=color, linestyle=item.get("linestyle", "-"), label=label)
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        ax_u.set(xlabel="time step", ylabel="control")
        ax_u.grid(True, linestyle="--", alpha=0.4)
        ax_u.legend()
        fig.tight_layout()
        return fig, (ax, ax_u)
    return fig, (ax, None)


_OBSTACLE_PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _obstacle_view_limits(xmin, xmax, state_bounds, constraint_pad):
    bound_lo = bound_hi = None
    if state_bounds is not None:
        bound_lo, bound_hi = map(float, state_bounds)
        if bound_lo >= bound_hi:
            raise ValueError("state_bounds must satisfy lower < upper")
        if xmin is None:
            xmin = bound_lo - float(constraint_pad)
        if xmax is None:
            xmax = bound_hi + float(constraint_pad)
    return xmin, xmax, bound_lo, bound_hi


def _draw_obstacle_scene(
    ax,
    obstacles,
    *,
    obstacle_band,
    regions,
    publication,
    xmin,
    xmax,
    bound_lo,
    bound_hi,
):
    from matplotlib.patches import Ellipse, Rectangle

    for i, obstacle in enumerate(obstacles):
        p = float(obstacle["p"])
        b = float(obstacle["b"])
        theta_deg = np.degrees(float(obstacle.get("theta", 0.0)))
        center = (float(obstacle["c"]), float(obstacle["d"]))
        label = None if publication else ("hard obstacle" if i == 0 else None)
        ax.add_patch(Ellipse(
            center, width=p / np.sqrt(b), height=p, angle=theta_deg,
            facecolor="tab:purple", edgecolor="tab:purple",
            alpha=0.14 if publication else 0.22, hatch="////",
            linewidth=0.8 if publication else 1.5, label=label,
        ))
        band = float(obstacle.get("band", obstacle_band))
        if band > 0.0:
            level = (p / 2.0) ** 2 + band
            clearance_label = None if publication else (
                "barrier / PSF clearance" if i == 0 else None
            )
            ax.add_patch(Ellipse(
                center, width=2.0 * np.sqrt(level / b), height=2.0 * np.sqrt(level),
                angle=theta_deg, fill=False, edgecolor="tab:purple", linestyle="--",
                linewidth=0.9 if publication else 1.2,
                alpha=0.65 if publication else 0.8, label=clearance_label,
            ))

    for region in regions or []:
        center = np.asarray(region["center"], dtype=float)
        side = float(region["side"])
        lower = center - side / 2.0
        ax.add_patch(Rectangle(
            lower, side, side, fill=False, linestyle=region.get("linestyle", ":"),
            linewidth=region.get("linewidth", 1.5),
            edgecolor=region.get("color", "0.35"),
            alpha=region.get("alpha", 1.0), zorder=region.get("zorder", 1.5),
            label=region.get("label"),
        ))

    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
    if bound_lo is not None:
        shade = "0.94"
        ax.axvspan(xmin, bound_lo, color=shade, zorder=0)
        ax.axvspan(bound_hi, xmax, color=shade, zorder=0)
        ax.axhspan(xmin, bound_lo, color=shade, zorder=0)
        ax.axhspan(bound_hi, xmax, color=shade, zorder=0)
        ax.add_patch(Rectangle(
            (bound_lo, bound_lo), bound_hi - bound_lo, bound_hi - bound_lo,
            fill=False, edgecolor="0.35", linewidth=1.0,
            linestyle=(0, (4, 2.5)), zorder=2,
        ))


def _draw_obstacle_markers(
    ax, r, first_x, *, batch_idx, publication, reference_marker_size, show_initial
):
    if r is not None:
        ref = _np(r)[batch_idx]
        ax.scatter(
            ref[-1, 0], ref[-1, 1], color="black", s=reference_marker_size,
            marker="*", label=None if publication else "reference", zorder=5,
        )
    if show_initial and first_x is not None:
        ax.scatter(
            first_x[0, 0], first_x[0, 1], s=24, marker="o",
            facecolor="white", edgecolor="black", linewidth=0.8, zorder=6,
        )


def _finish_obstacle_axes(fig, ax, handles, labels, *, title, publication):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    if publication:
        ax.tick_params(direction="in", top=False, right=False, width=0.6, length=3)
        ax.grid(color="0.90", linewidth=0.45, linestyle="-")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01),
            ncol=max(len(labels), 1), frameon=False, handlelength=2.2,
            columnspacing=1.3,
        )
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.91)
    else:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")


def plot_trajectories_with_obstacles(
    traj_list: List[Dict],
    obstacles: List[Dict],
    r=None,
    *,
    obstacle_band: float = 0.0,
    regions: Optional[List[Dict]] = None,
    batch_idx: int = 0,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    reference_marker_size: float = 80.0,
    title: str = "Closed-loop trajectories with obstacles",
    publication: bool = False,
    state_bounds: Optional[tuple[float, float]] = None,
    constraint_pad: float = 0.4,
    show_initial: bool = False,
    output_stem: Optional[str] = None,
    output_dpi: int = 600,
):
    """Plot XY trajectories against rotated obstacles and optional box constraints.

    ``publication=True`` applies compact journal typography and a method-only legend.
    ``state_bounds=(lower, upper)`` renders the infeasible exterior and a dashed
    constraint boundary. If limits are omitted, ``constraint_pad`` supplies the view
    around those bounds. ``output_stem`` writes PDF and ``*_600dpi.png`` copies.
    """
    import matplotlib.pyplot as plt
    xmin, xmax, bound_lo, bound_hi = _obstacle_view_limits(
        xmin, xmax, state_bounds, constraint_pad
    )
    style = _OBSTACLE_PUBLICATION_STYLE if publication else {}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.4, 4.8) if publication else (7.5, 7.5))
        _draw_obstacle_scene(
            ax, obstacles, obstacle_band=obstacle_band, regions=regions,
            publication=publication, xmin=xmin, xmax=xmax,
            bound_lo=bound_lo, bound_hi=bound_hi,
        )

        trajectory_handles = []
        trajectory_labels = []
        first_x = None
        for item in traj_list:
            x = _np(item.get("traj", item.get("x")))[batch_idx]
            if first_x is None:
                first_x = x
            line, = ax.plot(
                x[:, 0], x[:, 1], color=item.get("color"),
                linestyle=item.get("linestyle", "-"),
                linewidth=item.get("linewidth", 2.0),
                label=item.get("label", "trajectory"),
            )
            trajectory_handles.append(line)
            trajectory_labels.append(item.get("label", "trajectory"))
            if item.get("show_start", True):
                ax.scatter(
                    x[0, 0], x[0, 1], color=item.get("color"), s=32, marker="o",
                    zorder=item.get("zorder", 3.0),
                )

        _draw_obstacle_markers(
            ax, r, first_x, batch_idx=batch_idx, publication=publication,
            reference_marker_size=reference_marker_size, show_initial=show_initial,
        )
        _finish_obstacle_axes(
            fig, ax, trajectory_handles, trajectory_labels,
            title=title, publication=publication,
        )
        if not publication:
            fig.tight_layout()

        if output_stem is not None:
            stem = Path(output_stem)
            stem.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(stem.with_suffix(".pdf"))
            png_path = stem.parent / f"{stem.name}_{output_dpi}dpi.png"
            fig.savefig(png_path, dpi=output_dpi)
    return fig, ax


def plot_states_and_controls(
    traj_list: List[Dict],
    r_traj=None,
    *,
    batch_idx: int = 0,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    state_margin: Optional[float] = None,
    title: str = "Closed-loop states and controls",
):
    """Time-series of states (with reference/bounds) and controls for one or more methods.

    Each dict in ``traj_list``: ``{"x": (B,T,nx), "u": (B,T,nu), "label": str, "color"?}``.

    ``state_margin`` displays the box-safety activation region. For a margin ``delta``,
    the effective limits are ``xmin + delta`` and ``xmax - delta``; the hard bounds
    remain visible separately.
    """
    import matplotlib.pyplot as plt

    nx = _np(traj_list[0]["x"])[batch_idx].shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for item in traj_list:
        X = _np(item["x"])[batch_idx]
        U = _np(item["u"])[batch_idx]
        label = item.get("label", "")
        color = item.get("color")
        ls = item.get("linestyle", "-")
        for i in range(X.shape[1]):
            axes[0].plot(X[:, i], ls, color=color, alpha=0.9, label=f"{label} x{i}")
        for j in range(U.shape[1]):
            axes[1].plot(U[:, j], ls, color=color, alpha=0.9, label=f"{label} u{j}")

    if r_traj is not None:
        R = _np(r_traj)[batch_idx]
        for i in range(R.shape[1]):
            axes[0].plot(R[:, i], "k:", alpha=0.6, label=f"ref{i}")
    margin = None if state_margin is None else float(state_margin)
    if margin is not None and margin < 0.0:
        raise ValueError("state_margin must be nonnegative")
    if margin and xmin is not None and xmax is not None:
        if float(xmin) + margin > float(xmax) - margin:
            raise ValueError("state_margin leaves no feasible interval between xmin and xmax")

    if xmin is not None:
        axes[0].axhline(xmin, color="r", ls="--", alpha=0.55, label="hard bound")
        if margin:
            lower_margin = float(xmin) + margin
            axes[0].axhspan(float(xmin), lower_margin, color="darkorange", alpha=0.08, zorder=0)
            axes[0].axhline(
                lower_margin, color="darkorange", ls=":", alpha=0.8,
                label=f"safety margin ({margin:g})",
            )
    if xmax is not None:
        axes[0].axhline(xmax, color="r", ls="--", alpha=0.55)
        if margin:
            upper_margin = float(xmax) - margin
            axes[0].axhspan(upper_margin, float(xmax), color="darkorange", alpha=0.08, zorder=0)
            axes[0].axhline(upper_margin, color="darkorange", ls=":", alpha=0.8)

    axes[0].set_ylabel("state"); axes[0].set_title(title)
    axes[0].grid(True, ls="--", alpha=0.5); axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("control"); axes[1].set_xlabel("time step")
    axes[1].grid(True, ls="--", alpha=0.5); axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig, axes


def plot_controller_evaluation(system, cfg: Dict, examples: Dict, scenario: Dict):
    """Plot one shared MPC/SD-DPC/NN-DPC evaluation scenario."""
    styles = {
        "mpc": {"label": "MPC", "color": "tab:blue", "linestyle": "-"},
        "sd_dpc": {"label": "SD-DPC", "color": "green", "linestyle": "--"},
        "nn_dpc": {"label": "NN-DPC", "color": "darkorange", "linestyle": "-."},
    }
    entries = []
    for name in ("mpc", "sd_dpc", "nn_dpc"):
        if name in examples:
            entries.append({
                "x": examples[name]["x_traj"],
                "u": examples[name]["u_traj"],
                **styles[name],
            })

    obstacle = system.obstacle(cfg)
    if obstacle is not None:
        trajectories = [
            {"traj": item["x"], **{k: item[k] for k in ("label", "color", "linestyle")}}
            for item in entries
        ]
        return plot_trajectories_with_obstacles(
            trajectories,
            [obstacle],
            r=scenario["r"],
            xmin=system.xmin,
            xmax=system.xmax,
            title=f"{system.name}: MPC vs. SD-DPC vs. NN-DPC",
        )
    return plot_states_and_controls(
        entries,
        r_traj=scenario["r"],
        xmin=cfg.get("safe_xmin", system.xmin),
        xmax=cfg.get("safe_xmax", system.xmax),
        title=f"{system.name}: MPC vs. SD-DPC vs. NN-DPC",
    )
