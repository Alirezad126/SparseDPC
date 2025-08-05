import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from neuromancer.loggers import BasicLogger
from neuromancer.problem import Problem
from neuromancer.callbacks import Callback


def move_batch_to_device(batch: dict, device: str = "cpu") -> dict:
    """
    Move batch data to target device.

    Args:
        batch: Dictionary of data, potentially including torch.Tensor values.
        device: Target device (e.g., "cuda", "cpu")

    Returns:
        Dictionary with tensors moved to the specified device.
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class SparseTrainer:
    """
    Trainer for sparse models with iterative pruning using learned coefficients (SINDy-style).

    Args:
        problem: Neuromancer Problem instance
        fx_models: List of sparse models (e.g., SINDy policies) with .coef and .library
        lr: Learning rate
        train_data: Training data loader
        dev_data: Validation data loader
        test_data: Test data loader
        optimizers: Optional list of optimizers (one per fx model)
        logger: Logger for metrics (e.g., wandb or basic)
        callback: Neuromancer Callback for training hooks
        lr_scheduler: Use ReduceLROnPlateau
        epochs: Number of epochs
        patience: Early stopping patience
        prune_every: Frequency of coefficient pruning
        threshold: Coefficient pruning threshold
        threshold_mult: Threshold growth factor after each prune
        prune_noise: Small noise added to surviving coefficients after prune
        device: Device for training
    """

    def __init__(self,
                 problem: Problem,
                 fx_models: list,
                 lr: float,
                 train_data: DataLoader = None,
                 dev_data: DataLoader = None,
                 test_data: DataLoader = None,
                 optimizers: list = None,
                 logger: BasicLogger = None,
                 callback: Callback = Callback(),
                 lr_scheduler: bool = False,
                 epochs: int = 1000,
                 epoch_verbose: int = 1,
                 patience: int = 5,
                 warmup: int = 0,
                 train_metric: str = "train_loss",
                 dev_metric: str = "dev_loss",
                 test_metric: str = "test_loss",
                 eval_metric: str = "dev_loss",
                 eval_mode: str = "min",
                 clip: float = 100.0,
                 multi_fidelity: bool = False,
                 device: str = "cpu",
                 threshold: float = 1e-3,
                 prune_every: int = 5,
                 threshold_mult: float = 1.07,
                 prune_noise: float = 0.1):

        self.model = problem
        self.fx_models = list(fx_models)
        self.lr = lr
        self.threshold = threshold
        self.threshold_mult = threshold_mult
        self.prune_every = prune_every
        self.prune_noise = prune_noise

        self.optimizers = optimizers or [
            Adam(fp.parameters(), lr=lr) for fp in self.fx_models
        ]
        assert len(self.optimizers) == len(self.fx_models)

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.logger = logger
        self.callback = callback
        self.epochs = epochs
        self.current_epoch = 0
        self.epoch_verbose = epoch_verbose
        self.train_metric = train_metric
        self.dev_metric = dev_metric
        self.test_metric = test_metric
        self.eval_metric = eval_metric
        self._eval_min = eval_mode == "min"
        self.clip = clip
        self.device = device
        self.multi_fidelity = multi_fidelity
        self.patience = patience
        self.warmup = warmup
        self.badcount = 0
        self.best_devloss = float("inf") if self._eval_min else 0.
        self.best_model = self.model.state_dict()
        self.prev_active_counts = [fp.coef.shape[0] for fp in self.fx_models]

        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizers[0], mode="min", factor=0.5, patience=100
        ) if lr_scheduler else None

    def prune_columns(self):
        """
        Prune low-magnitude terms from each SINDy model based on coefficient magnitude.
        Reinitialize weights with noise if number of active terms changed.
        """
        with torch.no_grad():
            active_lists = []
            changed = False

            for idx, (fp, prev) in enumerate(zip(self.fx_models, self.prev_active_counts)):
                print(f"\n--- Model {idx} BEFORE pruning ---")
                print(fp)

                coef = fp.coef.detach()
                mask = torch.any(torch.abs(coef) > self.threshold, dim=1)
                active = torch.nonzero(mask, as_tuple=False).view(-1)

                if active.numel() == 0:
                    print("   !! All terms pruned — keeping one.")
                    active = torch.arange(coef.size(0), device=coef.device)[:1]

                if active.numel() != prev:
                    changed = True

                keep = set(active.tolist())
                removed = [fp.library.function_names[j]
                           for j in range(len(fp.library.function_names))
                           if j not in keep]

                print(f"   Threshold={self.threshold:.2e}, Active={active.numel()}, Pruned: {removed}")

                # Prune library
                fp.library.library = [fp.library.library[k] for k in keep]
                fp.library.function_names = [fp.library.function_names[k] for k in keep]
                fp.library.shape = (active.numel(), fp.library.shape[1])
                active_lists.append(active)

            if changed:
                print("⇒ Active set changed. Adding small noise to retained coefficients.")
                for idx, (fp, active) in enumerate(zip(self.fx_models, active_lists)):
                    keep_coef = fp.coef[active].detach()
                    noise = self.prune_noise * torch.randn_like(keep_coef)
                    fp.coef = torch.nn.Parameter(keep_coef + noise, requires_grad=True)
                    self.optimizers[idx] = Adam(fp.parameters(), lr=self.lr)

            else:
                print("⇒ No change in active terms.")

            # ---- print AFTER state ----------------------------------------- #
            for idx, fp in enumerate(self.fx_models):
                print(f"\n---------- Model {idx} AFTER pruning  ----------")
                print(fp)

            self.prev_active_counts = [a.numel() for a in active_lists]
            self.threshold = np.clip(self.threshold * self.threshold_mult, 0, 1)

    def train(self) -> dict:
        """
        Full training loop with pruning, early stopping, logging.
        Returns:
            Best model state_dict based on eval_metric.
        """
        self.callback.begin_train(self)

        try:
            for ep in range(self.current_epoch, self.current_epoch + self.epochs):
                self.model.train()
                batch_losses = []

                for batch in self.train_data:
                    batch["epoch"] = ep
                    batch = move_batch_to_device(batch, self.device)
                    out = self.model(batch)

                    if self.multi_fidelity:
                        for node in self.model.nodes:
                            out[self.train_metric] += node.callable.get_alpha_loss()

                    for opt in self.optimizers:
                        opt.zero_grad()
                    out[self.train_metric].backward()
                    clip_grad_norm_(self.model.parameters(), self.clip)
                    for opt in self.optimizers:
                        opt.step()

                    batch_losses.append(out[self.train_metric])
                    self.callback.end_batch(self, out)

                out[f"mean_{self.train_metric}"] = torch.mean(torch.stack(batch_losses))
                self.callback.begin_epoch(self, out)

                if self.lr_scheduler:
                    self.lr_scheduler.step(out[f"mean_{self.train_metric}"])

                if ep % self.prune_every == 0 and ep > 0:
                    self.prune_columns()

                with torch.set_grad_enabled(self.model.grad_inference):
                    self.model.eval()
                    if self.dev_data:
                        dev_losses = []
                        for d in self.dev_data:
                            d = move_batch_to_device(d, self.device)
                            dev_out = self.model(d)
                            dev_losses.append(dev_out[self.dev_metric])
                        dev_out[f"mean_{self.dev_metric}"] = torch.mean(torch.stack(dev_losses))
                        out.update(dev_out)

                    self.callback.begin_eval(self, out)

                    self.best_model = {
                        k: v.clone().detach() for k, v in self.model.state_dict().items()
                    }
                    self.best_devloss = out[self.eval_metric]
                    self.badcount = 0

                    if self.logger:
                        self.logger.log_metrics(out, step=ep)

                    self.callback.end_eval(self, out)
                    self.callback.end_epoch(self, out)

                self.current_epoch = ep + 1

        except KeyboardInterrupt:
            print("Training interrupted.")

        self.callback.end_train(self, out)
        self.model.load_state_dict(self.best_model, strict=False)
        return self.best_model
