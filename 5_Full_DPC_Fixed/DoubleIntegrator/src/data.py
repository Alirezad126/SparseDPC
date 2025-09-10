from neuromancer.dataset import DictDataset
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader


def get_policy_data(
    nsteps: int,
    n_samples: int,
    nx: int,
    device: torch.device,
    *,
    pos_idx: Tuple[int, int] = (0, 2),   # indices of x,y in the state vector
    batch_size: int = 200,
    seed: Optional[int] = None,
    # Rectangles are ((x_min, y_min), (x_max, y_max))
    init_rect: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, -1.0), (1.0, 1.0)),
    ref_rect:  Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, -1.0), (1.0, 1.0)),
    # Ellipse parameters for obstacle exclusion
    ellipse_configs: dict = {"p": 10.0, "b": 8.0, "c": 1.0, "d": 2.0},
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate training/dev datasets:
      - Initial positions sampled uniformly in init_rect; velocities = 0.
      - References sampled uniformly in ref_rect, held constant over horizon.
      - Any samples that fall INSIDE the obstacle ellipse are rejected.
    No margins or extra clearance.
    """

    # RNG
    if seed is None:
        gen_train = gen_dev = None
    else:
        gen_train = torch.Generator(device=device).manual_seed(seed)
        gen_dev   = torch.Generator(device=device).manual_seed(seed + 1)

    # Ellipse params
    p = float(ellipse_configs["p"])
    b = float(ellipse_configs["b"])
    c = float(ellipse_configs["c"])
    d = float(ellipse_configs["d"])
    assert b > 0.0 and p >= 0.0

    def outside_ellipse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # keep points whose quadratic form is >= (p/2)^2 (i.e., OUTSIDE the ellipse)
        return (b * (x - c)**2 + (y - d)**2) >= (p * 0.5)**2

    def sample_xy_in_rect_outside(
        n: int,
        rect: Tuple[Tuple[float, float], Tuple[float, float]],
        gen: Optional[torch.Generator],
        max_iters: int = 100,
        chunk: int = 4096,
    ) -> torch.Tensor:
        (x0, y0), (x1, y1) = rect
        # allow unordered corners
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        assert x1 > x0 and y1 > y0, "Rectangle must have positive width/height."

        out = torch.empty(0, 2, device=device)
        need = n
        it = 0
        while need > 0 and it < max_iters:
            m = max(need, chunk)
            xs = x0 + (x1 - x0) * torch.rand((m,), device=device, generator=gen)
            ys = y0 + (y1 - y0) * torch.rand((m,), device=device, generator=gen)
            keep = outside_ellipse(xs, ys)
            if keep.any():
                block = torch.stack([xs[keep], ys[keep]], dim=1)
                out = torch.cat([out, block], dim=0)
                need = n - out.shape[0]
            it += 1

        if out.shape[0] < n:
            raise RuntimeError(
                f"Could not sample {n} points outside the ellipse within the given rectangle "
                f"after {it} iterations. Consider changing the rectangle or the ellipse."
            )
        return out[:n]

    def _build_split(name: str, gen: Optional[torch.Generator]) -> "DictDataset":
        # Initial state: positions in init_rect but OUTSIDE ellipse; velocities = 0
        xy0 = sample_xy_in_rect_outside(n_samples, init_rect, gen)
        xn = torch.zeros((n_samples, 1, nx), device=device)
        xn[:, 0, pos_idx[0]] = xy0[:, 0]
        xn[:, 0, pos_idx[1]] = xy0[:, 1]

        # Reference positions (constant over horizon) in ref_rect, OUTSIDE ellipse
        xy_ref = sample_xy_in_rect_outside(n_samples, ref_rect, gen)
        r = torch.zeros((n_samples, nsteps + 1, 2), device=device)
        r[:, :, 0] = xy_ref[:, 0].unsqueeze(1).repeat(1, nsteps + 1)
        r[:, :, 1] = xy_ref[:, 1].unsqueeze(1).repeat(1, nsteps + 1)

        dct = {
            "xn": xn,                        # (n_samples, 1, nx)
            "r": r,                          # (n_samples, nsteps+1, 2)
            "obs_p": torch.full((n_samples, 1, 1), p, device=device),
            "obs_b": torch.full((n_samples, 1, 1), b, device=device),
            "obs_c": torch.full((n_samples, 1, 1), c, device=device),
            "obs_d": torch.full((n_samples, 1, 1), d, device=device),
        }
        return DictDataset(dct, name=name)

    train_data = _build_split("train", gen_train)
    dev_data   = _build_split("dev",   gen_dev)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=False, collate_fn=train_data.collate_fn)
    dev_loader   = DataLoader(dev_data, batch_size=batch_size,
                              shuffle=False, collate_fn=dev_data.collate_fn)
    return train_loader, dev_loader

