import json
import os
from typing import List, Optional, Dict, Any

import numpy as np
import wandb
from neuromancer.loggers import BasicLogger


class CustomLogger(BasicLogger):
    """
    Extended logger for tracking sparse SINDy policy training progress.

    Logs the following:
    - Epoch-wise train/dev losses.
    - Number of active terms and expression strings for each SINDy policy.
    - JSON snapshots saved to disk per policy every `prune_every` epochs.
    - Sends metrics to Weights & Biases (wandb) for external tracking.

    Parameters
    ----------
    fx_policies : list
        List of SINDy-like models with `.coef` and `.library.function_names`.
    save_dir : str
        Directory to store per-policy JSON log files.
    pol_configs : dict
        Must contain key 'prune_every' (int): frequency to log SINDy snapshots.
    *args, **kwargs :
        Passed to base `BasicLogger`.
    """
    def __init__(
        self,
        fx_policies: Optional[List[Any]] = None,
        save_dir: Optional[str] = None,
        pol_configs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.fx_policies = fx_policies
        self.save_dir = save_dir
        self.pol_configs = pol_configs
        self.policy_logs = {}

        if fx_policies is not None and save_dir is not None:
            for i in range(len(fx_policies)):
                json_path = os.path.join(self.save_dir, f"policy{i}_log.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        self.policy_logs[i] = json.load(f)
                else:
                    self.policy_logs[i] = {"log": []}

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics and snapshot SINDy policies every `prune_every` epochs.

        Parameters
        ----------
        metrics : dict
            Dictionary of metric names and values (e.g. train_loss, dev_loss).
        step : int, optional
            Current training epoch.
        """
        super().log_metrics(metrics, step=step)

        filtered = {k: v for k, v in metrics.items() if k in ['train_loss', 'dev_loss']}
        if step is not None:
            filtered['epoch'] = step

            if self.pol_configs and step % self.pol_configs.get('prune_every', 1) == 0:
                for i, fp in enumerate(self.fx_policies or []):
                    coef = fp.coef.detach().cpu().numpy().flatten().tolist()
                    active = int(np.count_nonzero(coef))
                    expr = str(fp)

                    fnames = []
                    if hasattr(fp, "library") and hasattr(fp.library, "function_names"):
                        fnames = fp.library.function_names

                    filtered[f"active_terms_policy_{i}"] = active

                    log_entry = {
                        "epoch": step,
                        "active_terms": active,
                        "function_names": fnames,
                        "coef": coef,
                        "expr": expr
                    }
                    self.policy_logs[i]["log"].append(log_entry)

                    json_path = os.path.join(self.save_dir, f"policy{i}_log.json")
                    with open(json_path, "w") as f:
                        json.dump(self.policy_logs[i], f, indent=2)

        wandb.log(filtered)
