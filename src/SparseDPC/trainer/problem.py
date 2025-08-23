# --------------------------------------------------------------------------- #
# builder_sparse_policy.py
# --------------------------------------------------------------------------- #
from __future__ import annotations
from typing import Dict, Sequence, List, Optional

import torch

import neuromancer.dynamics.ode as ode
from neuromancer.dynamics    import integrators as nm_int
from neuromancer.system     import Node, System
from neuromancer.loss       import PenaltyLoss
from neuromancer.problem    import Problem
from neuromancer.constraint import variable
from neuromancer.loggers    import BasicLogger

from SparseDPC.utils.integrator import OneElementEulerIntegrator
from SparseDPC.trainer.trainer   import SparseTrainer
from SparseDPC.sindy.fx_library   import fx_policy_library
from SparseDPC.utils.integrator   import FullStateEulerIntegrator

# --------------------------------------------------------------------------- #
# builder_sparse_dynamics.py
# --------------------------------------------------------------------------- #



class SparseDynamicsBuilder:
    """
    One *independent* sparse-identification trainer **per state coordinate**.

    • builds a tiny 1-state Euler integrator around each SINDy block
    • fits on-step + N-step prediction with L¹ sparsity
    • exposes:

        builder.models[i]   → SINDy model for xᵢ
        builder.trainers[i] → corresponding SparseTrainer
        builder.problems[i] → Neuromancer Problem

    Example
    -------
    ```python
    builder = SparseDynamicsBuilder(
        fx_models   = sindy_list,          # len == nx
        nx          = 2,
        ts          = 0.05,
        train_loader= train_loader,
        dev_loader  = dev_loader,
        config      = {
            'onestep_coef' : 1.0,
            'ref_coef'     : 2.0,
            'l1_coef'      : 3e-6,
            'lr'           : 2e-3,
            'epochs'       : 15_000,
            'threshold'    : 5e-3,
            'prune_every'  : 2_000,
            'warmup'       : 300,
            'patience'     : 50,
            'threshold_mult': 1.09,
            'prune_noise'  : 0.05,
        },
        device      = torch.device('cuda'),
        logger      = BasicLogger(args=None, savedir='logs')
    )

    builder.train_all()
    print(builder.models[0])          # pretty-print learned equation for x₀
    ```
    """

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        fx_models:   Sequence[torch.nn.Module],     # one per state
        nx:          int,
        ts:          float,
        train_loader,
        dev_loader,
        config:      Dict[str, float | int],
        device:      torch.device,
        logger:      BasicLogger | None = None
    ):
        self.nx         = nx
        self.ts         = ts
        self.fx_models  = list(fx_models)           # keep reference
        self.device     = device
        self.cfg        = config

        if logger is None:                          # default basic logger
            logger = BasicLogger(
                args=None, savedir='logs',
                verbosity=100, stdout=['train_loss', 'dev_loss']
            )
        self.shared_logger = logger

        # Containers ------------------------------------------------------
        self.problems:  List[Problem]       = []
        self.trainers:  List[SparseTrainer] = []
        self.models    = self.fx_models     # alias

        # Build trainer for every state ----------------------------------
        for idx in range(nx):
            problem   = self._make_problem(idx)
            optimizer = torch.optim.AdamW(problem.parameters(), lr=self.cfg['lr'])

            trainer = SparseTrainer(
                problem        = problem,
                fx_models      = [self.fx_models[idx]],
                lr             = self.cfg['lr'],
                train_data     = train_loader,
                dev_data       = dev_loader,
                optimizers     = [optimizer],
                epochs         = self.cfg['epochs'],
                train_metric   = 'train_loss',
                eval_metric    = 'dev_loss',
                logger         = self.shared_logger,
                device         = self.device,
                threshold      = self.cfg['threshold'],
                prune_every    = self.cfg['prune_every'],
                warmup         = self.cfg.get('warmup',     0),
                patience       = self.cfg.get('patience',   0),
                threshold_mult = self.cfg.get('threshold_mult', 1.0),
                prune_noise    = self.cfg.get('prune_noise',    0.0),
            )

            self.problems.append(problem)
            self.trainers.append(trainer)

    # ------------------------------------------------------------------ #
    # helper: build Neuromancer Problem for x_idx
    # ------------------------------------------------------------------ #
    def _make_problem(self, idx: int) -> Problem:
        """
        Only the *idx*-th state is integrated & penalised.
        """
        fx_i = self.fx_models[idx]

        integ = OneElementEulerIntegrator(fx_i, h=self.ts)
        node  = Node(
            integ,
            ['X', f'x{idx}_n', 'u'],                 # inputs
            [f'x{idx}_n'],                           # output overwrites old
            name=f"x{idx}_integrator"
        )

        dynamics = System([node])

        # --------------------- losses -----------------------------------
        x      = variable(f"x{idx}")
        x_hat  = variable(f'x{idx}_n')[:, :-1, :]

        w1 = self.cfg['onestep_coef']
        w2 = self.cfg['ref_coef']
        w3 = self.cfg['l1_coef']

        onestep = w1 * ((x_hat[:, 1, :] == x[:, 1, :]) ^ 2);  onestep.name = "onestep"
        nsteps  = w2 * ((x_hat == x) ^ 2);                    nsteps.name  = "ref_loss"

        l1_raw  = variable([x],
                           lambda y: torch.norm(list(fx_i.parameters())[0], p=1))
        l1_pen  = w3 * (l1_raw == 0);                         l1_pen.name  = "l1"

        loss = PenaltyLoss([nsteps, onestep, l1_pen], [])
        return Problem([dynamics], loss)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def train_all(self):
        """
        Train every coordinate’s SINDy model **once** with the provided
        loaders.  (No horizon-doubling here; simply call again if you want
        more iterations.)
        """
        for idx, trainer in enumerate(self.trainers):
            self.problems[idx].show()
            print(f"\n=== Training state x[{idx}] ===")
            best = trainer.train()
            trainer.model.load_state_dict(best)


class SparsePolicyBuilder:
    """
    Convenience wrapper that assembles

        policy → clamp → integrator → objectives/constraints → trainer

    Typical usage
    -------------
    ```python
    # 1) build everything *except* the trainer/logger
    builder = SparsePolicyBuilder(
        fx_models=fx_models, nx=nx, nu=nu, nref=1,
        ts=ts, nsteps=nsteps,
        train_loader=train_loader, dev_loader=dev_loader,
        bounds=bounds, config=config,
        seed=42, device=device,
        is_sindy=True, gt_model=None,           # leave default
        logger=None                             # <- defer logger
    )

    # 2) create a custom logger that needs the policies
    logger = CustomLogger(
        args=None, save_dir="logs/",
        verbosity=10, stdout=['train_loss', 'dev_loss'],
        fx_policies=builder.policies,
        pol_configs=config
    )

    # 3) hand the logger back and build the trainer
    builder.set_logger(logger)

    # 4) training
    best_state = builder.train()
    ```

    If you’re happy with a basic logger:

    ```python
    basic_logger = BasicLogger(...)

    builder = SparsePolicyBuilder(
        ..., logger=basic_logger   # trainer constructed immediately
    )
    best_state = builder.train()
    ```
    """

    # ------------------------------- ctor ---------------------------------- #
    def __init__(
        self,
        *,
        fx_models: Sequence[torch.nn.Module],
        nx: int,
        nref: int,
        nu: int,
        ts: float,
        nsteps: int,
        train_loader,
        dev_loader,
        bounds: Dict[str, torch.Tensor | float],
        config: Dict[str, float | int],
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

        # 0.  generate one policy block per input channel ------------------ #
        self.policies: List[torch.nn.Module] = [
            fx_policy_library(nx=nx, nref=nref,
                              policy_name=f"u_{k}",
                              seed=seed, device=device)
            for k in range(nu)
        ]

        # 1.  build system graph (policy → integrator) --------------------- #
        self._build_system(fx_models, ts, nsteps, bounds,
                           is_sindy=is_sindy, gt_model=gt_model)

        # 2.  objectives, constraints, Problem ---------------------------- #
        self._build_problem(bounds, config)

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
        fx_models: Sequence[torch.nn.Module],
        ts: float,
        nsteps: int,
        bounds,
        *,
        is_sindy: bool,
        gt_model: Optional[ode.ODESystem]
    ):
        umin, umax = bounds["umin"], bounds["umax"]

        policy_node = Node(
            lambda xn, r: torch.cat(
                [torch.clamp(fp(xn, r), umin, umax) for fp in self.policies],
                dim=-1
            ),
            ['xn', 'r'], ['u'],
            name="policy_combined"
        )

        integ = (FullStateEulerIntegrator(fx_models, ts)
                 if is_sindy else
                 nm_int.Euler(gt_model, h=torch.tensor(ts)))

        integ_node = Node(integ, ['xn', 'u'], ['xn'], name="x_integrator")
        self.system = System([policy_node, integ_node], nsteps=nsteps)

    def _build_problem(self, bounds, cfg):
        xmin, xmax = bounds["xmin"], bounds["xmax"]

        x   = variable('xn')
        ref = variable('r')

        reg = cfg["reg_coef"] * ((x == ref) ^ 2);  reg.name = "state_loss"

        l1_terms = []
        for k, fp in enumerate(self.policies):
            l1_raw = variable([x],
                              lambda y, fp=fp: torch.norm(list(fp.parameters())[0], p=1),
                              display_name=f"l1_pol{k}")
            l1_pen = cfg["l1_coef"] * (l1_raw == 0)
            l1_pen.name = f"loss_l1_pol{k}"
            l1_terms.append(l1_pen)

        c = cfg["const_coef"]
        constraints = [
            c * (x > xmin),                      # lower
            c * (x < xmax),                      # upper
            c * (x[:, [-1], :] > ref - 1e-2),    # terminal lower
            c * (x[:, [-1], :] < ref + 1e-2)     # terminal upper
        ]
        for n, nm in zip(constraints,
                         ["x_min", "x_max", "y_N_min", "y_N_max"]):
            n.name = nm

        loss = PenaltyLoss([reg, *l1_terms], constraints)
        self.problem = Problem([self.system], loss)

    def _instantiate_trainer(self, logger):
        cfg = self._config
        lr  = cfg["lr"]
        opts = [torch.optim.AdamW(fp.parameters(), lr=lr)
                for fp in self.policies]

        self.trainer = SparseTrainer(
            problem       = self.problem,
            fx_models     = self.policies,
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
            threshold_mult= cfg["threshold_mult"],
            prune_noise   = cfg["prune_noise"]
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
        return best_state


