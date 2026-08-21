# Double Integrator: Obstacle Avoidance

This case study learns a sparse two-input policy for point-to-point motion planning with
state, input, control-rate, and rotated-ellipse keep-out constraints. Online adaptation
compensates for a perturbed deployment model while predictive barriers or a predictive
safety filter (PSF) preserve feasibility.

The discrete model is

```text
x[k + 1] = A x[k] + B u[k],    x in R^2, u in R^2.
```

## Workflow

| Notebook | Purpose |
|---|---|
| [`01_system_id.ipynb`](notebooks/01_system_id.ipynb) | Identify sparse dynamics |
| [`02_policy.ipynb`](notebooks/02_policy.ipynb) | Train and inspect SD-DPC/NN-DPC policies |
| [`03_unconstrained_adaptation.ipynb`](notebooks/03_unconstrained_adaptation.ipynb) | Reference-only online adaptation |
| [`04_safe_adaptation.ipynb`](notebooks/04_safe_adaptation.ipynb) | Barrier adaptation and PSF baselines |
| [`04b_safe_adaptation_jacobian.ipynb`](notebooks/04b_safe_adaptation_jacobian.ipynb) | Symbolic versus autograd safety gradients |
| [`04c_sparse_symbolic_vs_nn_autograd.ipynb`](notebooks/04c_sparse_symbolic_vs_nn_autograd.ipynb) | Sparse-symbolic versus neural-autograd timing |
| [`04d_barrier_collapse_sweep.ipynb`](notebooks/04d_barrier_collapse_sweep.ipynb) | Adaptation stability sweep |
| [`04e_multi_obstacle_barrier_vs_psf.ipynb`](notebooks/04e_multi_obstacle_barrier_vs_psf.ipynb) | Multi-obstacle barrier/PSF comparison |
| [`05_evaluation.ipynb`](notebooks/05_evaluation.ipynb) | MPC, SD-DPC, and NN-DPC evaluation |

The YAML files in [`configs/`](configs) define the nominal problem, perturbation,
training objective, obstacle geometry, adaptation gains, barrier type, integration
method, safety bands, and evaluation scenarios.

## Command-Line Runs

```bash
python scripts/DoubleIntegrator/system_id.py
python scripts/DoubleIntegrator/train_sd_dpc.py
python scripts/DoubleIntegrator/train_nn_dpc.py
python scripts/DoubleIntegrator/unconstrained_adaptation.py
python scripts/DoubleIntegrator/safe_adaptation.py
python scripts/DoubleIntegrator/evaluation.py
```

## Method Notes

- The sparse dynamics model is a direct discrete-time map; no numerical integration is
  required for one-step prediction.
- The safety specification combines one or more rotated ellipses, a configurable state
  box, and an optional `du_max` control-rate constraint.
- Safe adaptation supports autograd and analytic sensitivity backends. The symbolic
  backend evaluates precompiled SINDy derivatives outside the online loop.
- CasADi/IPOPT provides the matched MPC and PSF baselines.

Generated artifacts are stored under [`results/`](results).
