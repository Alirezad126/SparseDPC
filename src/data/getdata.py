import torch
from torch.utils.data import DataLoader
from neuromancer.dataset import DictDataset

def get_data(
    sys,
    nsim: int,
    nsteps: int,
    ts: float,
    bs: int,
    device: torch.device = torch.device("cpu"),
):
    """
    General‑purpose data generator that works for any nx (= sys.nx)

    Parameters
    ----------
    sys     : psl.system       – the dynamical system
    nsim    : int              – total simulated points
    nsteps  : int              – time steps per training sequence
    ts      : float            – simulation time step
    bs      : int              – DataLoader batch size
    device  : torch.device     – CPU / GPU destination for tensors
    """
    nx, nu = sys.nx, sys.nu
    nbatch  = nsim // nsteps          # full mini‑trajectories per split
    length  = nbatch * nsteps         # truncate so every split is complete

    # --- simulate three splits ------------------------------------------------
    train_sim, dev_sim, test_sim = [
        sys.simulate(nsim=nsim, ts=ts) for _ in range(3)
    ]

    # Helper -------------------------------------------------------------------
    def _tensorise(sim_dict, keep_1d=False):
        """Turn a raw sim split into a DictDataset‑compatible dict."""
        X_full = sim_dict["X"][:length].reshape(nbatch, nsteps, nx)
        U_full = sim_dict["U"][:length].reshape(nbatch, nsteps, nu)

        X_full_t = torch.tensor(X_full, dtype=torch.float32, device=device)
        U_full_t = torch.tensor(U_full, dtype=torch.float32, device=device)

        out = {
            "X":  X_full_t,                       # (B, T, nx)
            "xn": X_full_t[:, 0:1, :],            # first state in each traj
            "u":  U_full_t,
        }

        # Per‑state views -------------------------------------------------------
        if keep_1d:
            for i in range(nx):
                # slice keeps a channel dimension (shape: B, T, 1)
                xi = X_full_t[:, :, i : i + 1]
                out[f"x{i}"]   = xi
                out[f"x{i}_n"] = xi[:, 0:1, :]

        return out

    # Build loaders ------------------------------------------------------------
    train_dict = _tensorise(train_sim, keep_1d=True)
    dev_dict   = _tensorise(dev_sim,   keep_1d=True)

    train_data = DictDataset(train_dict, name="train")
    dev_data   = DictDataset(dev_dict,   name="dev")

    train_loader = DataLoader(
        train_data, batch_size=bs, shuffle=True, collate_fn=train_data.collate_fn
    )
    dev_loader = DataLoader(
        dev_data, batch_size=bs, shuffle=True, collate_fn=dev_data.collate_fn
    )

    # Test split: one long trajectory; keep_1d=True for consistency ------------
    test_length   = nbatch * nsteps
    test_dict_raw = {
        "X": test_sim["X"][:test_length].reshape(1, test_length, nx),
        "U": test_sim["U"][:test_length].reshape(1, test_length, nu),
    }
    test_dict = _tensorise(test_dict_raw, keep_1d=True)

    return train_loader, dev_loader, test_dict
