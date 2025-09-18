from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from neuromancer.loggers import BasicLogger
from neuromancer.problem import Problem
from neuromancer.callbacks import Callback
from neuromancer.system import Node, System
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.dynamics import ode, integrators

from SparseDPC.vectorized.sindy import *
import importlib
import SparseDPC.trainer.problem
import SparseDPC.sindy.sindy
import SparseDPC.sindy.fx_library

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
                 sindy: SINDyVectorized,
                 lr: float,
                 train_data: DataLoader = None,
                 dev_data: DataLoader = None,
                 test_data: DataLoader = None,
                 optimizers= None,
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
                 device : torch.device = torch.device("cpu"),
                 prune_every: int = 5,
                 prune_every_decay: int = 200,
                 prune_every_min: int = 10,
                 prune_noise: float = 0.05,
                 change_prune_every: bool = False,
                 threshold_mult: float = 1.07,
                 threshold: float = 1e-3,
                 threshold_max: float = 1e0,
                 l1_decay: float = 1.0,
                 lr_decay_step: int = 2000,
                 lr_decay_gamma: float = 0.8,):

        self.model = problem
        self.sindy = sindy
        self.lr = lr
        self.threshold = threshold
        self.threshold_max = threshold_max
        self.threshold_mult = threshold_mult
        self.prune_every = prune_every
        self.change_prune_every = change_prune_every
        self.prune_every_decay = prune_every_decay
        self.prune_every_min = prune_every_min
        self.prune_noise = prune_noise
        self.l1_decay = l1_decay

        self.optimizers = optimizers
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma
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
        self.prev_active_counts = [fp.shape[0] for fp in list(self.sindy.parameters())]

        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizers[0], mode="min", factor=0.5, patience=100
        ) if lr_scheduler else None

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

                    self.optimizers.zero_grad()
                    out[self.train_metric].backward()
                    clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizers.step()

                    batch_losses.append(out[self.train_metric])
                    self.callback.end_batch(self, out)

                out[f"mean_{self.train_metric}"] = torch.mean(torch.stack(batch_losses))
                self.callback.begin_epoch(self, out)

                if self.lr_scheduler:
                    self.lr_scheduler.step(out[f"mean_{self.train_metric}"])

                if ep % self.prune_every == 0 and ep > 0:
                    print("*"*75)
                    print("SINDy BEFORE pruning: ")

                    print("Terms BEFORE: ", [p.shape[0] for p in self.sindy.parameters()])
                    self.sindy.pretty_print()

                    self.sindy.prune_(thresholds = self.threshold, noise_after = self.prune_noise)
                    self.optimizers = torch.optim.AdamW(self.sindy.parameters(), lr=self.lr)
                    self.scheduler  = torch.optim.lr_scheduler.StepLR(self.optimizers, step_size=self.lr_decay_step, gamma=self.lr_decay_gamma)


                    print("*"*75)
                    print("SINDy AFTER pruning: ")
                    print("Terms AFTER: ", [p.shape[0] for p in self.sindy.parameters()])

                    self.sindy.pretty_print()
                    print("*"*75)
                    self.model.loss.objectives[-1].weight *= self.l1_decay
                    print("New L1 weight: ", self.model.loss.objectives[-1].weight)
                    print("New pruning threshold: ", self.threshold)
                    print("*"*75)

                    self.threshold = min(self.threshold_max, self.threshold * self.threshold_mult)
                    if self.change_prune_every:
                        self.prune_every = max(self.prune_every_min, self.prune_every - self.prune_every_decay)

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

class SparsePolicyBuilder2D:
    """
    """

    # ------------------------------- ctor ---------------------------------- #
    def __init__(
        self,
        *,
        dynamics_model: SINDyVectorized,
        policy: SINDyVectorized,
        ts: float,
        nsteps: int,
        train_loader,
        dev_loader,
        bounds: Dict[str, torch.Tensor | float],
        config: Dict[str, float | int | bool],
        obstacle_configs: Dict[str, float | int],
        refstep: int = 2,
        seed: int,
        device: torch.device,
        is_sindy: bool = True,
        gt_model: Optional[ode.ODESystem] = None,
        logger=None                     # may be None –> call set_logger later
    ) -> None:

        self.device        = device
        self._config       = config           # stash for later
        self._train_loader = train_loader
        self._dev_loader   = dev_loader
        self._bounds       = bounds
        self._obstacle_configs = obstacle_configs

        # 0.  generate policy block ------------------ #
        self.policy = policy


        # 1.  build system graph (policy → integrator) --------------------- #
        self._build_system(dynamics_model, ts, nsteps, bounds,
                           is_sindy=is_sindy, gt_model=gt_model)

        # 2.  objectives, constraints, Problem ---------------------------- #
        self._build_problem(bounds, config, obstacle_configs, refstep)

        # 3.  build trainer *IF* a logger is supplied ---------------------- #
        self.trainer = None
        if logger is not None:
            self.set_logger(logger)

        self.problem.show()

    # ====================================================================== #
    # internal helpers
    # ====================================================================== #

    def _build_system(
        self,
        dynamics_model: SINDyVectorized,
        ts: float,
        nsteps: int,
        bounds,
        *,
        is_sindy: bool,
        gt_model: Optional[ode.ODESystem]
    ):
        umin, umax = bounds["umin"], bounds["umax"]

        policy_node = Node(
            lambda xn, r: torch.clamp(self.policy(xn,r), umin, umax),
            ['xn', 'r'], ['u'],
            name="policy_combined"
        )

        integrator = integrators.Euler(dynamics_model, h=ts)
        integrator_node = Node(integrator, ['xn', 'u'], [f'xn'], name="x_integrator")
        self.system = System([policy_node, integrator_node], nsteps=nsteps)

    def _build_problem(self, bounds, cfg, obstacle_cfg, refstep):

        xmin, xmax, umin, umax = bounds["xmin"], bounds["xmax"], bounds["umin"], bounds["umax"]
        x   = variable('xn')        # states over horizon, shape: (T, B, nx)
        u   = variable('u')         # controls over horizon, shape: (T, B, nu)
        ref = variable('r')         # references, shape: (T, B, 2)  -> [x_ref, y_ref]

        # select positions from state to compare against ref
        x_pos = x[:, :, [0, 2]]
        x_vel = x[:, :, [1, 3]]
        x1 = variable('xn')[:, :, [0]]
        x2 = variable('xn')[:, :, [2]]

        # losses
        action_loss       = cfg["Q_u"]  * ((u == 0.0)          ^ 2)   # control penalty
        reference_loss_position    = cfg["Q_r"]  * ((ref[:, -refstep:, :]==x_pos[:, -refstep:, :])    ^ 2)   # track [x,y]
        reference_loss_velocity    = cfg["Q_r"]/2  * ((x_vel[:, -1:, :] == 0.0)    ^ 2)
        state_smoothing   = cfg["Q_dx"] * ((x[:, 1:, :] == x[:, :-1, :] ) ^ 2)   # Δx penalty
        control_smoothing = cfg["Q_du"] * ((u[:, 1:, :] == u[:, :-1, :] ) ^ 2)   # Δu penalty



        l1_policy = variable([x], lambda x: sum(torch.norm(p, p=1) for p in self.policy.Xi))
        l1_pen = cfg["l1_coef"] * (l1_policy == 0)
        l1_pen.name = f"loss_l1_policy"

        objectives = [reference_loss_position, reference_loss_velocity, action_loss, state_smoothing, control_smoothing]

        for n, nm in zip(objectives,
                         ["reference_loss_position", "reference_loss_velocity", "action_loss", "state_smoothing", "control_smoothing"]):
            n.name = nm

        #Contraints
        p = obstacle_cfg["p"] ; b = obstacle_cfg["b"] ; c = obstacle_cfg["c"]; d = obstacle_cfg["d"]
        constraints = [
            cfg["Q_con_obs"]  * ( (p / 2) ** 2 <= (b * (x1 - c) ** 2 + (x2 - d) ** 2)),
            cfg["Q_con_x"] * (x_pos > xmin),                      # lower
            cfg["Q_con_x"] * (x_pos < xmax),                      # upper
            cfg["Q_con_u"] * (u < umax),                      # lower
            cfg["Q_con_u"] * (u > umin),                      # upper

        ]
        for n, nm in zip(constraints,
                         ["obstacle_pen", "x_min", "x_max", "u_min", "u_max"]):
            n.name = nm

        loss = PenaltyLoss([*objectives, l1_pen], constraints)
        self.problem = Problem([self.system], loss)

    def _instantiate_trainer(self, logger):
        cfg = self._config
        lr  = cfg["lr"]
        opts = torch.optim.AdamW(self.policy.parameters(), lr=lr)

        self.trainer = SparseTrainer(
            problem       = self.problem,
            sindy         = self.policy,
            lr            = lr,
            train_data    = self._train_loader,
            dev_data      = self._dev_loader,
            optimizers    = opts,
            epochs        = cfg["epochs"],
            train_metric  = 'train_loss',
            eval_metric   = 'dev_loss',
            logger        = logger,
            device        = self.device,
            threshold     = cfg["threshold"],
            prune_every   = cfg["prune_every"],
            prune_every_min= cfg["prune_every_min"],
            prune_every_decay= cfg["prune_every_decay"],
            change_prune_every= cfg["change_prune_every"],
            threshold_mult = cfg["threshold_mult"],
            threshold_max = cfg["threshold_max"],
            prune_noise   = cfg["prune_noise"],
            lr_decay_gamma= cfg["lr_decay_gamma"],
            lr_decay_step = cfg["lr_decay_step"],
            l1_decay= cfg["l1_decay"],

        )


    # ====================================================================== #
    # public API
    # ====================================================================== #

    def set_logger(self, logger):
        """
        Plug-in a logger **after** the builder has been created.
        If a trainer already exists it will be replaced.
        """
        self._instantiate_trainer(logger)

    def train(self):
        if self.trainer is None:
            raise RuntimeError("Logger not set → trainer not built. "
                               "Call `set_logger(logger)` first.")
        best_state = self.trainer.train()
        self.trainer.model.load_state_dict(best_state)