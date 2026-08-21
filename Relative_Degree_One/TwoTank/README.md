# Two Tank: Constrained Level Tracking

This case study learns sparse pump/valve policies for two coupled liquid levels. It
includes nominal controller evaluation and safe online adaptation under parametric plant
perturbations.

```text
dx1/dt = c1 (1 - u2) u1 - c2 sqrt(x1)
dx2/dt = c1 u2 u1 + c2 sqrt(x1) - c2 sqrt(x2)
```

The nominal parameters are `c1 = 0.08` and `c2 = 0.04`; states and controls lie in
`[0, 1]` unless a tighter evaluation box is configured.

## Workflow

| Notebook | Purpose |
|---|---|
| [`01_system_id.ipynb`](notebooks/01_system_id.ipynb) | Identify sparse continuous dynamics |
| [`02_policy.ipynb`](notebooks/02_policy.ipynb) | Train and inspect SD-DPC/NN-DPC policies |
| [`03_unconstrained_adaptation.ipynb`](notebooks/03_unconstrained_adaptation.ipynb) | Reference-only adaptation |
| [`04_safe_adaptation.ipynb`](notebooks/04_safe_adaptation.ipynb) | Barrier adaptation and PSF baselines |
| [`04b_safe_adaptation_jacobian.ipynb`](notebooks/04b_safe_adaptation_jacobian.ipynb) | Analytic/autograd Jacobian verification and timing |
| [`04c_sparse_symbolic_vs_nn_autograd.ipynb`](notebooks/04c_sparse_symbolic_vs_nn_autograd.ipynb) | Sparse-symbolic versus neural-autograd comparison |
| [`05_evaluation.ipynb`](notebooks/05_evaluation.ipynb) | Nominal MPC/SD-DPC/NN-DPC evaluation |
| [`05b_safety_evaluation.ipynb`](notebooks/05b_safety_evaluation.ipynb) | Multi-trajectory safety evaluation |

Reference schedules may contain any number of piecewise-constant levels. Online
adaptation repeats the currently active reference over its predictive horizon and only
switches when the executed data sequence switches.

## Command-Line Runs

```bash
python scripts/TwoTank/system_id.py
python scripts/TwoTank/train_sd_dpc.py
python scripts/TwoTank/train_nn_dpc.py
python scripts/TwoTank/unconstrained_adaptation.py
python scripts/TwoTank/safe_adaptation.py
python scripts/TwoTank/evaluation.py
```

## Method Notes

- Continuous predictions support Euler and RK4 integration.
- Safety uses configurable lower/upper level bounds, optional safety margins, and an
  optional `du_max` control-rate limit.
- Autograd and analytic sensitivity updates share the same exact or straight-through
  action-clamp gradient mode.
- Safety evaluation stores every method trajectory and reports violations,
  convergence, smoothness, policy-only inference time, and full online runtime.

Configurations are in [`configs/`](configs); generated artifacts are in
[`results/`](results).
