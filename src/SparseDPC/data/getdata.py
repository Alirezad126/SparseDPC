import torch
from typing import Tuple, Dict
from typing import Optional, Tuple
from neuromancer.dataset import DictDataset
from torch.utils.data import DataLoader
from neuromancer.dynamics.ode import ODESystem

def get_data(
    sys: ODESystem,
    nsim: int,
    nsteps: int,
    ts: float,
    bs: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    """
    Generate training, validation, and test data loaders from a system.

    Parameters:
        sys: Simulatable dynamical system
        nsim: Number of time steps to simulate
        nsteps: Sequence horizon per sample
        ts: Time step for simulation
        bs: Batch size
        device: Device to place tensors

    Returns:
        train_loader, dev_loader, test_dict
    """
    nx, nu = sys.nx, sys.nu
    nbatch = nsim // nsteps
    length = nbatch * nsteps

    train_sim, dev_sim, test_sim = [
        sys.simulate(nsim=nsim, ts=ts) for _ in range(3)
    ]

    def _tensorise(sim_dict: Dict[str, torch.Tensor], keep_1d: bool = False) -> Dict[str, torch.Tensor]:
        X = sim_dict["X"][:length].reshape(nbatch, nsteps, nx)
        U = sim_dict["U"][:length].reshape(nbatch, nsteps, nu)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        U_t = torch.tensor(U, dtype=torch.float32, device=device)

        out: Dict[str, torch.Tensor] = {
            "X": X_t,
            "xn": X_t[:, 0:1, :],
            "u": U_t,
        }

        if keep_1d:
            for i in range(nx):
                xi = X_t[:, :, i:i+1]
                out[f"x{i}"] = xi
                out[f"x{i}_n"] = xi[:, 0:1, :]

        return out

    train_dict = _tensorise(train_sim, keep_1d=True)
    dev_dict = _tensorise(dev_sim, keep_1d=True)

    train_data = DictDataset(train_dict, name="train")
    dev_data = DictDataset(dev_dict, name="dev")

    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True,
                              collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=bs, shuffle=True,
                            collate_fn=dev_data.collate_fn)

    test_length = nbatch * nsteps
    test_raw = {
        "X": test_sim["X"][:test_length].reshape(1, test_length, nx),
        "U": test_sim["U"][:test_length].reshape(1, test_length, nu),
    }
    test_dict = _tensorise(test_raw, keep_1d=True)

    return train_loader, dev_loader, test_dict


def get_policy_data(
    nsteps: int,
    n_samples: int,
    nx: int,
    device: torch.device,
    *,
    same_ref_for_all_states: bool = True,
    zero_refs: bool = False,
    batch_size: int = 200,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate synthetic policy training and dev datasets.

    Parameters:
        nsteps: Prediction horizon
        n_samples: Number of samples
        nx: Number of states
        device: CUDA or CPU device
        same_ref_for_all_states: Use same reference for all states
        zero_refs: All reference levels zero if True
        batch_size: DataLoader batch size
        seed: RNG seed for reproducibility

    Returns:
        train_loader, dev_loader
    """

    if seed is None:
        gen_train = gen_dev = None
    else:
        gen_train = torch.Generator(device=device).manual_seed(seed)
        gen_dev = torch.Generator(device=device).manual_seed(seed + 1)

    def _build_split(name: str, gen: Optional[torch.Generator]) -> DictDataset:
        if same_ref_for_all_states:
            if zero_refs:
                levels = torch.zeros(n_samples, 1, 1, device=device)
                ref = levels.repeat(1, nsteps + 1, nx)
            else:
                levels = torch.rand(n_samples, 1, 1, device=device, generator=gen)
                ref = levels.repeat(1, nsteps + 1, nx)
        else:
            levels = torch.rand(n_samples, 1, nx, device=device, generator=gen)
            ref = levels.repeat(1, nsteps + 1, 1)

        xn = torch.rand(n_samples, 1, nx, device=device, generator=gen)

        d: Dict[str, torch.Tensor] = {'xn': xn, 'r': ref}
        for i in range(nx):
            d[f"x{i}_n"] = xn[:, :, i:i+1]
            d[f"r{i}"] = ref[:, :, i:i+1]

        return DictDataset(d, name=name)

    train_data = _build_split('train', gen_train)
    dev_data = _build_split('dev', gen_dev)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=False, collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=batch_size,
                            shuffle=False, collate_fn=dev_data.collate_fn)

    return train_loader, dev_loader
