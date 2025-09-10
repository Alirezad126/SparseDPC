import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import seaborn as sns
from typing import List, Optional, Sequence, Tuple, Dict
import torch
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
def _rect_corners_to_llwh(rect):
    """
    rect: ((x1, y1), (x2, y2)) with any corner order.
    returns: (x_ll, y_ll, width, height), and (xmin, xmax, ymin, ymax)
    """
    (x1, y1), (x2, y2) = rect
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return (xmin, ymin, xmax - xmin, ymax - ymin), (xmin, xmax, ymin, ymax)

def plot_training_sample_with_ellipse(
    loader,
    sample_idx: int = 0,
    pos_idx=(0, 2),           # (x_index, y_index). For nx=2 use (0,1)
    *,
    p: float | None = None,
    b: float | None = None,
    c: float | None = None,
    d: float | None = None,
    title: str = "Training sample: init vs. reference with ellipse obstacle",
    init_rect=None,           # ((x1,y1),(x2,y2)) start sampling region
    ref_rect=None,            # ((x1,y1),(x2,y2)) reference sampling region
    xmin=-1., xmax=1.         # plot limits (square)
):
    # get one batch
    batch = next(iter(loader))
    xn = batch["xn"].detach().cpu().numpy()         # (B, 1, nx)
    r  = batch["r"].detach().cpu().numpy()          # (B, N+1, nx)

    # read ellipse params if missing
    def get_param(name, fallback):
        if fallback is None and name in batch:
            return float(batch[name][sample_idx].reshape(-1)[0])
        return fallback

    p  = get_param("obs_p", p)
    b_ = get_param("obs_b", b)
    c_ = get_param("obs_c", c)
    d_ = get_param("obs_d", d)

    if None in (p, b_, c_, d_):
        raise ValueError("Ellipse params p,b,c,d must be provided or present in the batch as obs_*.")

    # extract initial (single) and reference (assumed constant across horizon for xy)
    x_i = float(xn[sample_idx, 0, pos_idx[0]])
    y_i = float(xn[sample_idx, 0, pos_idx[1]])
    x_r = float(r[sample_idx, 0, 0])
    y_r = float(r[sample_idx, 0, 1])

    # ellipse semi-axes for b*(x-c)^2 + (y-d)^2 = (p/2)^2
    if b_ <= 0 or p < 0:
        raise ValueError("Require b > 0 and p >= 0 for a valid ellipse.")
    a_x = (p * 0.5) / np.sqrt(b_)
    a_y = (p * 0.5)

    fig, ax = plt.subplots(figsize=(6, 6))

    # plot initial and reference points
    ax.scatter([x_i], [y_i], marker="x", s=60, label="x_init")
    ax.scatter([x_r], [y_r], marker="*", s=80, label="ref")
    ax.plot([x_i, x_r], [y_i, y_r], linestyle="--", linewidth=1.0, alpha=0.6, label="init→ref")

    # draw keep-out ellipse (filled + outline)
    ell_fill = Ellipse((c_, d_), width=2*a_x, height=2*a_y, alpha=0.25, color="purple")
    ax.add_patch(ell_fill)
    ell_edge = Ellipse((c_, d_), width=2*a_x, height=2*a_y, fill=False, linewidth=1.5, label="obstacle")
    ax.add_patch(ell_edge)

    # --- NEW: sampling rectangles ---
    if init_rect is not None:
        (rx, ry, rw, rh), _ = _rect_corners_to_llwh(init_rect)
        ax.add_patch(Rectangle((rx, ry), rw, rh, fill=True, alpha=0.10, color="blue"))
        ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, linewidth=1.2, linestyle=":"))
        ax.text(rx, ry, " x_init_rect", va="top", ha="left", fontsize=9)

    if ref_rect is not None:
        (rx, ry, rw, rh), _ = _rect_corners_to_llwh(ref_rect)
        ax.add_patch(Rectangle((rx, ry), rw, rh, fill=True, alpha=0.10, color="orange"))
        ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, linewidth=1.2, linestyle="--"))
        ax.text(rx, ry, " ref_rect", va="top", ha="left", fontsize=9)

    # cosmetics
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(xmin, xmax); ax.set_ylim(xmin, xmax)
    ax.legend(loc="best")
    plt.show()

    return fig, ax


def make_test_sample(
    *,
    nx: int,
    nsteps: int,
    device: torch.device,
    ellipse: Dict[str, float],     # {"p":..., "b":..., "c":..., "d":...}
    seed: int = 7,
    init_vel_std: float = 0.0,     # e.g., 0.05 for a bit of initial motion
    # Rectangles: ((x_min, y_min), (x_max, y_max))
    init_rect: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, -1.0), (1.0, 1.0)),
    ref_rect:  Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, -1.0), (1.0, 1.0)),
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict for closed-loop rollouts:
        data = {"xn": (1,1,nx), "r": (1,nsteps+1,2)}
    - (x,y) for the initial state is sampled UNIFORMLY within init_rect, but OUTSIDE the ellipse.
    - Reference (x_ref,y_ref) is sampled UNIFORMLY within ref_rect, also OUTSIDE the ellipse,
      and repeated across time.
    - No margins or extra clearance.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Ellipse parameters
    p = float(ellipse["p"]); b = float(ellipse["b"])
    c = float(ellipse["c"]); d = float(ellipse["d"])
    assert b > 0.0 and p >= 0.0

    def outside_ellipse_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # True if (x,y) is OUTSIDE the ellipse b(x-c)^2 + (y-d)^2 = (p/2)^2
        return (b * (x - c) ** 2 + (y - d) ** 2) >= (0.5 * p) ** 2

    def _ordered_rect(rect: Tuple[Tuple[float, float], Tuple[float, float]]):
        (x0, y0), (x1, y1) = rect
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        if not (x1 > x0 and y1 > y0):
            raise ValueError("Rectangle must have positive width and height.")
        return x0, y0, x1, y1

    def sample_xy_in_rect_outside(rect: Tuple[Tuple[float, float], Tuple[float, float]],
                                  max_tries: int = 200_000,
                                  block: int = 4096) -> Tuple[float, float]:
        x0, y0, x1, y1 = _ordered_rect(rect)
        tries = 0
        while tries < max_tries:
            m = min(block, max_tries - tries)
            xs = rng.uniform(x0, x1, size=m)
            ys = rng.uniform(y0, y1, size=m)
            keep = outside_ellipse_np(xs, ys)
            if np.any(keep):
                # take the first valid point
                idx = np.nonzero(keep)[0][0]
                return float(xs[idx]), float(ys[idx])
            tries += m
        raise RuntimeError(
            "Could not sample a valid (x,y) outside the ellipse within the given rectangle. "
            "Adjust the rectangle or ellipse parameters."
        )

    # --- Initial state (x,y) sampled in init_rect but OUTSIDE ellipse ---
    x0, y0 = sample_xy_in_rect_outside(init_rect)
    xn = torch.zeros((1, 1, nx), dtype=torch.float32, device=device)
    # positions (assumes indices 0 and 2 are x,y)
    xn[:, :, 0] = x0
    if nx >= 3:
        xn[:, :, 2] = y0
    # optional small initial velocities (indices 1 and 3 if present)
    if nx >= 2 and init_vel_std > 0:
        xn[:, :, 1] = torch.randn(1, 1, device=device) * init_vel_std
    if nx >= 4 and init_vel_std > 0:
        xn[:, :, 3] = torch.randn(1, 1, device=device) * init_vel_std

    # --- Constant reference (x_ref,y_ref) sampled in ref_rect but OUTSIDE ellipse ---
    xr, yr = sample_xy_in_rect_outside(ref_rect)
    r = torch.zeros((1, nsteps + 1, 2), dtype=torch.float32, device=device)
    r[:, :, 0] = xr
    r[:, :, 1] = yr

    return {"xn": xn, "r": r}




def plot_trajectories_with_ellipse(
    traj_list: list[dict],
    r: torch.Tensor | np.ndarray,
    *,
    p: float,
    b: float,
    c: float,
    d: float,
    batch_idx: int = 0,
    xmin: float = -1.0,
    xmax: float = 1.0,
    title: str = "Closed-loop trajectories with keep-out ellipse",
    show_ref_path: bool = True,
):
    """
    Plots XY paths from multiple simulated trajectories plus the keep-out ellipse.

    Parameters
    ----------
    traj_list : list of dicts
        Each dict must contain:
          - "traj": (B, T, nx) tensor/array with states
          - "label": str for legend
        Optional:
          - "color": matplotlib color
          - "linestyle": e.g. "-", "--"
          - "alpha": float
    r : (B, T, 2) reference positions over time
    p, b, c, d : ellipse parameters
    batch_idx : which batch index to plot
    """

    # convert reference to numpy
    if isinstance(r, torch.Tensor):
        r = r.detach().cpu().numpy()
    R = r[batch_idx]  # (T, 2)
    xr, yr = R[:, 0], R[:, 1]

    # ellipse radii
    if b <= 0.0 or p < 0.0:
        raise ValueError("Require b > 0 and p >= 0 for a valid ellipse.")
    a_x = (p * 0.5) / np.sqrt(b)
    a_y = (p * 0.5)

    fig, ax = plt.subplots(figsize=(6, 6))

    # plot each trajectory
    for cfg in traj_list:
        X = cfg["traj"]
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = X[batch_idx]  # (T, nx)

        if X.shape[1] < 3:
            raise ValueError("Expected nx>=3 with positions at indices 0 (x) and 2 (y).")

        xs, ys = X[:, 0], X[:, 2]
        color     = cfg.get("color", None)
        linestyle = cfg.get("linestyle", "-")
        alpha     = cfg.get("alpha", 1.0)
        label     = cfg.get("label", "trajectory")

        ax.plot(xs, ys, lw=1.5, color=color, linestyle=linestyle,
                alpha=alpha, label=label)

        ax.scatter([xs[-1]], [ys[-1]], c=color, marker="*", s=120)
    ax.scatter([xs[0]], [ys[0]], c="green", marker="x", s=80, label="start point")
    # optional reference path
    if show_ref_path:
        ax.scatter([xr[-1]], [yr[-1]], c="purple", marker="o", s=60,
                   label="reference")

    # ellipse (fill + edge)
    ell_fill = Ellipse((c, d), width=2 * a_x, height=2 * a_y,
                       facecolor="tab:purple", edgecolor="none", alpha=0.5)
    ax.add_patch(ell_fill)
    ell_edge = Ellipse((c, d), width=2 * a_x, height=2 * a_y,
                       fill=False, edgecolor="tab:purple", lw=1, alpha=0.6,
                       label="obstacle")
    ax.add_patch(ell_edge)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.6)

    # limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.legend(loc="best")
    plt.show()
    return fig, ax

