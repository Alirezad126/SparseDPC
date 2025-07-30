from copy import deepcopy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import wandb
import lightning.pytorch as pl

from neuromancer.loggers import BasicLogger
from neuromancer.problem import Problem
from neuromancer.callbacks import Callback
from neuromancer.problem import LitProblem
from neuromancer.dataset import LitDataModule

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger


def move_batch_to_device(batch, device="cpu"):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# class SparseDynamicsTrainer:
#     """
#     Trainer for sparse optimization with iterative thresholding and column pruning.
#     """
#     def __init__(
#         self,
#         problem: Problem,
#         fx,
#         lr,
#         train_data: torch.utils.data.DataLoader,
#         dev_data: torch.utils.data.DataLoader = None,
#         test_data: torch.utils.data.DataLoader = None,
#         optimizer: torch.optim.Optimizer = None,
#         logger: BasicLogger = None,
#         callback=Callback(),
#         lr_scheduler=False,
#         epochs=1000,
#         epoch_verbose=1,
#         patience=5,
#         warmup=0,
#         threshold_mult=1.07,
#         train_metric="train_loss",
#         dev_metric="dev_loss",
#         test_metric="test_loss",
#         eval_metric="dev_loss",
#         eval_mode="min",
#         clip=100.0,
#         multi_fidelity=False,
#         device="cpu",
#         threshold=1e-3,  # Threshold for pruning
#         prune_every=5 # Frequency of pruning
#
#     ):
#         self.model = problem
#         self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
#             problem.parameters(), 0.01, betas=(0.0, 0.9))
#         self.train_data = train_data
#         self.dev_data = dev_data
#         self.test_data = test_data
#         self.callback = callback
#         self.logger = logger
#         self.epochs = epochs
#         self.current_epoch = 0
#         self.epoch_verbose = epoch_verbose
#         self.train_metric = train_metric
#         self.dev_metric = dev_metric
#         self.test_metric = test_metric
#         self.eval_metric = eval_metric
#         self._eval_min = eval_mode == "min"
#         self.lr_scheduler = (
#             torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=100)
#             if lr_scheduler
#             else None
#         )
#         self.patience = patience
#         self.warmup = warmup
#         self.badcount = 0
#         self.clip = clip
#         self.best_devloss = float("inf") if self._eval_min else 0.
#         self.best_model = self.model.state_dict()
#         self.multi_fidelity = multi_fidelity
#         self.device = device
#         self.threshold = threshold
#         self.prune_every = prune_every  # Frequency of pruning
#         self.fx = fx  # Store SINDy model separately for pruning
#         self.lr = lr
#         self.threshold_mult = threshold_mult
#         self.prev_active_count = self.fx.coef.shape[0]
#
#     def prune_columns(self):
#         """
#         Prunes functions (columns in the library matrix) that have coefficients below the threshold.
#         Reinitializes the remaining active terms **only if** their count has changed since the last pruning.
#         """
#         with torch.no_grad():
#
#             print("Current terms:")
#             print(self.fx)
#
#             coef = self.fx.coef  # Get coefficients from fx (SINDy model)
#
#             # Identify active columns (functions that have nonzero impact)
#             mask_col = torch.any(torch.abs(coef) > self.threshold, dim=1)
#             active_idx = torch.nonzero(mask_col, as_tuple=False).view(-1)  # Ensures it's always 1D
#
#             if active_idx.numel() == 0:
#                 print("Warning: All terms have been pruned. Keeping at least one term.")
#                 return  # Prevent deleting all parameters
#
#             print(f"Pruning... Threshold: {self.threshold},  Active terms remaining: {active_idx.numel()}")
#
#             # Compute pruned (deleted) function names BEFORE updating
#             active_idx_list = active_idx.tolist()
#             all_functions = self.fx.library.function_names
#             deleted_idx = sorted(set(range(len(all_functions))) - set(active_idx_list))
#             deleted_functions = [all_functions[i] for i in deleted_idx]
#             print("Pruned terms:", deleted_functions)
#
#             # **Always prune: Update function library and coefficient matrix**
#             self.fx.library.library = [self.fx.library.library[i] for i in active_idx_list]
#             self.fx.library.function_names = [self.fx.library.function_names[i] for i in active_idx_list]
#             self.fx.library.shape = (active_idx.numel(), self.fx.library.shape[1])  # Update shape
#
#
#             if self.prev_active_count != active_idx.numel():
#                 print(
#                     f"Reinitializing parameters: Active terms changed from {self.prev_active_count} to {active_idx.numel()}.")
#                 # **Reinitialize only the remaining active coefficients**
#                 new_coef = torch.randn_like(coef[active_idx, :]) * 0.01
#                 self.fx.coef = torch.nn.Parameter(new_coef, requires_grad=True)
#
#                 # **Reinitialize optimizer to use the new parameters**
#                 self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#             else:
#                 print(f"No change in active terms ({self.prev_active_count}), skipping reinitialization.")
#
#             print("Updated terms:")
#             print(self.fx)
#
#             # **Always update previous count after pruning**
#             self.prev_active_count = active_idx.numel()
#
#             # Update threshold
#             self.threshold *= self.threshold_mult
#             self.threshold = np.clip(self.threshold, 0, 1)
#
#     def train(self):
#         """
#         Training loop with iterative thresholding and pruning every few epochs.
#         """
#         self.callback.begin_train(self)
#
#         try:
#             for i in range(self.current_epoch, self.current_epoch + self.epochs):
#                 #print(self.fx.coef)
#                 self.model.train()
#                 losses = []
#
#                 for t_batch in self.train_data:
#                     t_batch['epoch'] = i
#                     t_batch = move_batch_to_device(t_batch, self.device)
#
#                     output = self.model(t_batch)
#
#                     if self.multi_fidelity:
#                         for node in self.model.nodes:
#                             alpha_loss = node.callable.get_alpha_loss()
#                             output[self.train_metric] += alpha_loss
#
#                     self.optimizer.zero_grad()
#                     output[self.train_metric].backward()
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
#                     self.optimizer.step()
#
#                     losses.append(output[self.train_metric])
#                     self.callback.end_batch(self, output)
#
#                 output[f'mean_{self.train_metric}'] = torch.mean(torch.stack(losses))
#                 self.callback.begin_epoch(self, output)
#
#                 if self.lr_scheduler is not None:
#                     self.lr_scheduler.step(output[f'mean_{self.train_metric}'])
#
#                 # **Prune every X epochs**
#                 if i % self.prune_every == 0 and i > 0:
#                     self.prune_columns()
#
#                 with torch.set_grad_enabled(self.model.grad_inference):
#                     self.model.eval()
#                     if self.dev_data is not None:
#                         losses = []
#                         for d_batch in self.dev_data:
#                             d_batch = move_batch_to_device(d_batch, self.device)
#                             eval_output = self.model(d_batch)
#                             losses.append(eval_output[self.dev_metric])
#                         eval_output[f'mean_{self.dev_metric}'] = torch.mean(torch.stack(losses))
#                         output = {**output, **eval_output}
#
#                     self.callback.begin_eval(self, output)
#
#
#                     current_state = self.model.state_dict()
#                     self.best_model = {
#                         k: v.clone().detach() for k, v in current_state.items() if k in self.model.state_dict()
#                     }
#                     self.best_devloss = output[self.eval_metric]
#                     self.badcount = 0
#
#                     if self.logger is not None:
#                         self.logger.log_metrics(output, step=i)
#
#                     self.callback.end_eval(self, output)
#                     self.callback.end_epoch(self, output)
#
#                     if self.badcount > self.patience:
#                         print('Early stopping!!!')
#                         break
#
#                     self.current_epoch = i + 1
#
#         except KeyboardInterrupt:
#             print("Interrupted training loop.")
#
#         self.callback.end_train(self, output)
#
#         # **🔥 Fix: Load only matching parameters after pruning**
#         self.model.load_state_dict({
#             k: v for k, v in self.best_model.items() if k in self.model.state_dict()
#         }, strict=False)
#
#         return self.best_model
# class SparseDynamicsTrainer:
#     """
#     Trainer for sparse optimization with iterative thresholding and column pruning.
#     """
#     def __init__(
#         self,
#         problem: Problem,
#         fx,
#         lr,
#         train_data: torch.utils.data.DataLoader,
#         dev_data: torch.utils.data.DataLoader = None,
#         test_data: torch.utils.data.DataLoader = None,
#         optimizer: torch.optim.Optimizer = None,
#         logger: BasicLogger = None,
#         callback=Callback(),
#         lr_scheduler=False,
#         epochs=1000,
#         epoch_verbose=1,
#         patience=5,
#         warmup=0,
#         threshold_mult=1.07,
#         train_metric="train_loss",
#         dev_metric="dev_loss",
#         test_metric="test_loss",
#         eval_metric="dev_loss",
#         eval_mode="min",
#         clip=100.0,
#         multi_fidelity=False,
#         device="cpu",
#         threshold=1e-3,  # Threshold for pruning
#         prune_every=5 # Frequency of pruning
#
#     ):
#         self.model = problem
#         self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
#             problem.parameters(), 0.01, betas=(0.0, 0.9))
#         self.train_data = train_data
#         self.dev_data = dev_data
#         self.test_data = test_data
#         self.callback = callback
#         self.logger = logger
#         self.epochs = epochs
#         self.current_epoch = 0
#         self.epoch_verbose = epoch_verbose
#         self.train_metric = train_metric
#         self.dev_metric = dev_metric
#         self.test_metric = test_metric
#         self.eval_metric = eval_metric
#         self._eval_min = eval_mode == "min"
#         self.lr_scheduler = (
#             torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=100)
#             if lr_scheduler
#             else None
#         )
#         self.patience = patience
#         self.warmup = warmup
#         self.badcount = 0
#         self.clip = clip
#         self.best_devloss = float("inf") if self._eval_min else 0.
#         self.best_model = self.model.state_dict()
#         self.multi_fidelity = multi_fidelity
#         self.device = device
#         self.threshold = threshold
#         self.prune_every = prune_every  # Frequency of pruning
#         self.fx = fx  # Store SINDy model separately for pruning
#         self.lr = lr
#         self.threshold_mult = threshold_mult
#         self.prev_active_count = self.fx.coef.shape[0]
#
#     def prune_columns(self):
#         """
#         Prunes functions (columns in the library matrix) that have coefficients below the threshold.
#         Reinitializes the remaining active terms **only if** their count has changed since the last pruning.
#         """
#         with torch.no_grad():
#
#             print("Current terms:")
#             print(self.fx)
#
#             coef = self.fx.coef  # Get coefficients from fx (SINDy model)
#
#             # Identify active columns (functions that have nonzero impact)
#             mask_col = torch.any(torch.abs(coef) > self.threshold, dim=1)
#             active_idx = torch.nonzero(mask_col, as_tuple=False).view(-1)  # Ensures it's always 1D
#
#             if active_idx.numel() == 0:
#                 print("Warning: All terms have been pruned. Keeping at least one term.")
#                 return  # Prevent deleting all parameters
#
#             print(f"Pruning... Threshold: {self.threshold},  Active terms remaining: {active_idx.numel()}")
#
#             # Compute pruned (deleted) function names BEFORE updating
#             active_idx_list = active_idx.tolist()
#             all_functions = self.fx.library.function_names
#             deleted_idx = sorted(set(range(len(all_functions))) - set(active_idx_list))
#             deleted_functions = [all_functions[i] for i in deleted_idx]
#             print("Pruned terms:", deleted_functions)
#
#             # **Always prune: Update function library and coefficient matrix**
#             self.fx.library.library = [self.fx.library.library[i] for i in active_idx_list]
#             self.fx.library.function_names = [self.fx.library.function_names[i] for i in active_idx_list]
#             self.fx.library.shape = (active_idx.numel(), self.fx.library.shape[1])  # Update shape
#
#
#             if self.prev_active_count != active_idx.numel():
#                 print(
#                     f"Reinitializing parameters: Active terms changed from {self.prev_active_count} to {active_idx.numel()}.")
#                 # **Reinitialize only the remaining active coefficients**
#                 new_coef = torch.randn_like(coef[active_idx, :]) * 0.01
#                 self.fx.coef = torch.nn.Parameter(new_coef, requires_grad=True)
#
#                 # **Reinitialize optimizer to use the new parameters**
#                 self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#             else:
#                 print(f"No change in active terms ({self.prev_active_count}), skipping reinitialization.")
#
#             print("Updated terms:")
#             print(self.fx)
#
#             # **Always update previous count after pruning**
#             self.prev_active_count = active_idx.numel()
#
#             # Update threshold
#             self.threshold *= self.threshold_mult
#             self.threshold = np.clip(self.threshold, 0, 1)
#
#     def train(self):
#         """
#         Training loop with iterative thresholding and pruning every few epochs.
#         """
#         self.callback.begin_train(self)
#
#         try:
#             for i in range(self.current_epoch, self.current_epoch + self.epochs):
#                 #print(self.fx.coef)
#                 self.model.train()
#                 losses = []
#
#                 for t_batch in self.train_data:
#                     t_batch['epoch'] = i
#                     t_batch = move_batch_to_device(t_batch, self.device)
#
#                     output = self.model(t_batch)
#
#                     if self.multi_fidelity:
#                         for node in self.model.nodes:
#                             alpha_loss = node.callable.get_alpha_loss()
#                             output[self.train_metric] += alpha_loss
#
#                     self.optimizer.zero_grad()
#                     output[self.train_metric].backward()
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
#                     self.optimizer.step()
#
#                     losses.append(output[self.train_metric])
#                     self.callback.end_batch(self, output)
#
#                 output[f'mean_{self.train_metric}'] = torch.mean(torch.stack(losses))
#                 self.callback.begin_epoch(self, output)
#
#                 if self.lr_scheduler is not None:
#                     self.lr_scheduler.step(output[f'mean_{self.train_metric}'])
#
#                 # **Prune every X epochs**
#                 if i % self.prune_every == 0 and i > 0:
#                     self.prune_columns()
#
#                 with torch.set_grad_enabled(self.model.grad_inference):
#                     self.model.eval()
#                     if self.dev_data is not None:
#                         losses = []
#                         for d_batch in self.dev_data:
#                             d_batch = move_batch_to_device(d_batch, self.device)
#                             eval_output = self.model(d_batch)
#                             losses.append(eval_output[self.dev_metric])
#                         eval_output[f'mean_{self.dev_metric}'] = torch.mean(torch.stack(losses))
#                         output = {**output, **eval_output}
#
#                     self.callback.begin_eval(self, output)
#
#
#                     current_state = self.model.state_dict()
#                     self.best_model = {
#                         k: v.clone().detach() for k, v in current_state.items() if k in self.model.state_dict()
#                     }
#                     self.best_devloss = output[self.eval_metric]
#                     self.badcount = 0
#
#                     if self.logger is not None:
#                         self.logger.log_metrics(output, step=i)
#
#                     self.callback.end_eval(self, output)
#                     self.callback.end_epoch(self, output)
#
#                     if self.badcount > self.patience:
#                         print('Early stopping!!!')
#                         break
#
#                     self.current_epoch = i + 1
#
#         except KeyboardInterrupt:
#             print("Interrupted training loop.")
#
#         self.callback.end_train(self, output)
#
#         # **🔥 Fix: Load only matching parameters after pruning**
#         self.model.load_state_dict({
#             k: v for k, v in self.best_model.items() if k in self.model.state_dict()
#         }, strict=False)
#
#         return self.best_model



# --------------------------------------------------------------------------- #
class SparseTrainer:
    """
    Same sparse-pruning logic as before, but prints for each policy:
      • Current terms  (before pruning)
      • Threshold & #active after mask
      • Pruned term names
      • Updated terms (after pruning / possible re-init)
    """

    # --------------------------- INIT -------------------------------------- #
    def __init__(self,
                 problem: Problem,
                 fx_models: list,          # list of SINDy models
                 lr: float,
                 train_data=None,
                 dev_data=None,
                 test_data=None,
                 optimizers=None,
                 logger: BasicLogger = None,
                 callback=Callback(),
                 lr_scheduler=False,
                 epochs=1000,
                 epoch_verbose=1,
                 patience=5,
                 warmup=0,
                 train_metric="train_loss",
                 dev_metric="dev_loss",
                 test_metric="test_loss",
                 eval_metric="dev_loss",
                 eval_mode="min",
                 clip=100.0,
                 multi_fidelity=False,
                 device="cpu",
                 threshold=1e-3,
                 prune_every=5,
                 threshold_mult=1.07):    # ← multiplier from original snippet

        self.model        = problem
        self.fx_models  = list(fx_models)
        self.lr           = lr
        self.threshold    = threshold
        self.threshold_mult = threshold_mult
        self.prune_every  = prune_every

        self.optimizers   = (optimizers if optimizers is not None
                             else [torch.optim.Adam(fp.parameters(), lr=lr)
                                   for fp in self.fx_models])

        assert len(self.optimizers) == len(self.fx_models)

        # bookkeeping -------------------------------------------------------- #
        self.train_data   = train_data
        self.dev_data     = dev_data
        self.test_data    = test_data
        self.logger       = logger
        self.callback     = callback
        self.epochs       = epochs
        self.current_epoch = 0
        self.epoch_verbose = epoch_verbose
        self.train_metric  = train_metric
        self.dev_metric    = dev_metric
        self.test_metric   = test_metric
        self.eval_metric   = eval_metric
        self._eval_min     = eval_mode == "min"
        self.clip          = clip
        self.device        = device
        self.multi_fidelity = multi_fidelity
        self.patience      = patience
        self.warmup        = warmup
        self.badcount      = 0
        self.best_devloss  = float("inf") if self._eval_min else 0.
        self.best_model    = self.model.state_dict()
        self.prev_active_counts = [fp.coef.shape[0] for fp in self.fx_models]
        self.lr_scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                                self.optimizers[0], mode="min",
                                factor=0.5, patience=100)
                             if lr_scheduler else None)

    # ------------------------ PRUNE COLUMNS -------------------------------- #
    def prune_columns(self):
        """
        Enhanced diagnostic printing while preserving logic.
        If ANY model’s active set changes, all policies are re-initialised.
        """
        with torch.no_grad():
            active_lists, changed = [], False

            for idx, (fp, prev) in enumerate(zip(self.fx_models,
                                                 self.prev_active_counts)):
                print(f"\n---------- Model {idx} BEFORE pruning ----------")
                print(fp)                                     # Current terms

                coef = fp.coef.detach()
                mask = torch.any(torch.abs(coef) > self.threshold, dim=1)
                active = torch.nonzero(mask, as_tuple=False).view(-1)

                # avoid empty set
                if active.numel() == 0:
                    print("   !! All terms below threshold — keeping first one.")
                    active = torch.arange(coef.size(0), device=coef.device)[:1]

                now = active.numel()
                if now != prev:  changed = True

                print(f"   Threshold={self.threshold:.6g}  "
                      f"Active={now}  (prev={prev})")

                # report pruned names BEFORE update
                keep = set(active.tolist())
                removed_names = [fp.library.function_names[j]
                                 for j in range(len(fp.library.function_names))
                                 if j not in keep]
                print("   Pruned terms:", removed_names if removed_names else "[]")

                # ---- prune library ---------------------------------------- #
                keep_list = list(keep)
                fp.library.library        = [fp.library.library[k]        for k in keep_list]
                fp.library.function_names = [fp.library.function_names[k] for k in keep_list]
                fp.library.shape          = (now, fp.library.shape[1])

                active_lists.append(active)

            # ---- Re-init ALL policies if any changed ----------------------- #
            if changed:
                print("\n⇒ Active-set changed in at least one model; re-initialising ALL.")
                for idx, (fp, active) in enumerate(zip(self.fx_models, active_lists)):
                    new_coef = torch.randn_like(fp.coef[active]) * 1e-2
                    fp.coef  = torch.nn.Parameter(new_coef, requires_grad=True)
                    self.optimizers[idx] = torch.optim.Adam(fp.parameters(), lr=self.lr)
            else:
                print("\n⇒ No change in active sets; coefficients kept.")

            # ---- print AFTER state ----------------------------------------- #
            for idx, fp in enumerate(self.fx_models):
                print(f"\n---------- Model {idx} AFTER pruning  ----------")
                print(fp)

            # save counts & advance threshold
            self.prev_active_counts = [a.numel() for a in active_lists]
            self.threshold = np.clip(self.threshold * self.threshold_mult, 0, 1)

    # ----------------------------- TRAIN ----------------------------------- #
    def train(self):
        self.callback.begin_train(self)

        try:
            for ep in range(self.current_epoch, self.current_epoch + self.epochs):
                self.model.train()
                batch_losses = []

                for batch in self.train_data:
                    batch["epoch"] = ep
                    batch = move_batch_to_device(batch, self.device)
                    out   = self.model(batch)

                    if self.multi_fidelity:
                        for node in self.model.nodes:
                            out[self.train_metric] += node.callable.get_alpha_loss()

                    # zero, backward, step for every optimiser
                    for opt in self.optimizers:  opt.zero_grad()
                    out[self.train_metric].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    for opt in self.optimizers:  opt.step()

                    batch_losses.append(out[self.train_metric])
                    self.callback.end_batch(self, out)

                out[f"mean_{self.train_metric}"] = torch.mean(torch.stack(batch_losses))
                self.callback.begin_epoch(self, out)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(out[f"mean_{self.train_metric}"])

                if ep % self.prune_every == 0 and ep > 0:
                    self.prune_columns()

                # ----- DEV -------------------------------------------------- #
                with torch.set_grad_enabled(self.model.grad_inference):
                    self.model.eval()
                    if self.dev_data is not None:
                        dev_losses = []
                        for d in self.dev_data:
                            d = move_batch_to_device(d, self.device)
                            dev_out = self.model(d)
                            dev_losses.append(dev_out[self.dev_metric])
                        dev_out[f"mean_{self.dev_metric}"] = torch.mean(torch.stack(dev_losses))
                        out = {**out, **dev_out}

                    self.callback.begin_eval(self, out)

                    state = self.model.state_dict()
                    self.best_model  = {k: v.clone().detach() for k, v in state.items()}
                    self.best_devloss = out[self.eval_metric]
                    self.badcount     = 0

                    if self.logger is not None:
                        self.logger.log_metrics(out, step=ep)

                    self.callback.end_eval(self, out)
                    self.callback.end_epoch(self, out)

                self.current_epoch = ep + 1

        except KeyboardInterrupt:
            print("Training interrupted by user.")

        self.callback.end_train(self, out)
        self.model.load_state_dict(self.best_model, strict=False)
        return self.best_model
