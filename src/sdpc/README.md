# `sdpc` Framework

`sdpc` is the reusable implementation behind every experiment in this repository. It
separates system definitions, sparse models, training, adaptation, safety, baselines,
evaluation, and plotting from the case-study YAML files and notebooks.

## Package Map

| Module | Responsibility |
|---|---|
| `systems/` | Common `System` interface and Double Integrator, Two-Tank, and Van der Pol definitions |
| `sindy/` | Compiled function libraries, vectorized SINDy models, and checkpoint I/O |
| `training/` | Sparse system identification, SD-DPC training, NN-DPC training, pruning, and logging |
| `adaptation/` | Reference updates, predictive rollouts, analytic Jacobians, safe adaptation, and PSF |
| `safety/` | State constraints, rotated ellipses, box bounds, control-rate limits, and barrier losses |
| `baselines/` | CasADi/IPOPT MPC implementation |
| `eval/` | Scenario generation, rollouts, metrics, diagnostics, and case-specific evaluations |
| `data/` | System-ID data and constant, piecewise, or signal-generated references |
| `plotting/` | Publication plots, adaptation reports, and matched trajectory animations |
| `config.py`, `io/`, `registry.py` | YAML loading, reproducibility, run directories, and system lookup |

## Analytic Adaptation

For a sparse policy `u = pi(x, r; Xi)` and dynamics `x+ = F(x, u)`, the symbolic backend
evaluates

```text
dF/dx, dF/du, dpi/dx, dpi/dXi
```

directly from the compiled SINDy libraries. It propagates the closed-loop sensitivity

```text
S[k + 1] = (dF/dx + dF/du dpi/dx) S[k] + dF/du dpi/dXi
```

through the predictive horizon. Continuous systems use the corresponding Euler or RK4
tangent map. Safety gradients are accumulated from analytic constraint and barrier
derivatives, so symbolic runs do not retain or backpropagate through the rollout graph.

Autograd remains available for neural policies and as a numerical reference. Exact,
straight-through, and leaky straight-through action-clamp derivatives are selected by
the shared action-gradient mode used by reference and safety adaptation.

## Safety Model

`SafetySpec` combines differentiable state constraints `h_i(x)`, optional control-rate
limits, per-constraint activation bands, and weights. The framework provides box and
rotated-ellipse constraints plus `squared_hinge`, `relaxed_log`, and ReLU-style penalties.
Custom constraints can participate in symbolic adaptation by supplying an analytic
gradient function.

## Minimal Example

```python
import torch

from sdpc import make_system
from sdpc.adaptation import SafeAdaptationConfig, run_safe_adaptation

device = torch.device("cpu")
system = make_system("twotank", device=device)

# Load or train `policy`, then construct `data` and `cfg`.
spec = system.safety_specs(cfg)
result = run_safe_adaptation(
    policy,
    system.perturbed_plant(cfg),
    data,
    spec,
    SafeAdaptationConfig(horizon=20, safety_backend="symbolic"),
    derivative_model=system.perturbed_sindy_model(cfg),
    system=system,
    umin=system.umin,
    umax=system.umax,
)
```

## Adding a System

1. Subclass `systems.base.System`.
2. Implement the true dynamics, sparse library configs, data builders, and DPC loss.
3. Add safety and CasADi hooks when the case study requires adaptation, PSF, or MPC.
4. Register the class in `registry.py`.
5. Add case-specific YAML files, scripts, notebooks, and tests.

No adaptation or training module should require system-specific branching.
