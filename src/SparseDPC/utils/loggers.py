import json
import os
from neuromancer.loggers import BasicLogger
import wandb
import numpy as np


class CustomLogger(BasicLogger):
    def __init__(self, fx_policies=None, save_dir=None, pol_configs: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fx_policies = fx_policies
        self.save_dir = save_dir
        self.pol_configs = pol_configs
        self.policy_logs = {}

        # Initialize or load policy logs
        for i in range(len(fx_policies)):
            json_path = os.path.join(self.save_dir, f"policy{i}_log.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    self.policy_logs[i] = json.load(f)
            else:
                self.policy_logs[i] = {"log": []}

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        filtered = {k: v for k, v in metrics.items() if k in ['train_loss', 'dev_loss']}
        if step is not None:
            filtered['epoch'] = step

            if step % self.pol_configs['prune_every'] == 0:
                for i, fp in enumerate(self.fx_policies):
                    coef = fp.coef.detach().cpu().numpy().flatten().tolist()
                    active = int(np.count_nonzero(coef))
                    expr = str(fp)
                    fnames = []

                    if hasattr(fp, "library") and hasattr(fp.library, "function_names"):
                        fnames = fp.library.function_names

                    filtered[f"active_terms_policy_{i}"] = active

                    # Append current snapshot
                    log_entry = {
                        "epoch": step,
                        "active_terms": active,
                        "function_names": fnames,
                        "coef": coef,
                        "expr": expr
                    }
                    self.policy_logs[i]["log"].append(log_entry)

                    # Save updated log to file
                    json_path = os.path.join(self.save_dir, f"policy{i}_log.json")
                    with open(json_path, "w") as f:
                        json.dump(self.policy_logs[i], f, indent=2)

        wandb.log(filtered)
