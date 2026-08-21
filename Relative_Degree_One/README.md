# Relative-Degree-One Case Studies

This directory contains the relative-degree-one experiments built on the shared
[`sdpc`](../src/sdpc) framework. Each case study keeps only its system-specific YAML
configs, notebooks, checkpoints, metrics, and figures; reusable algorithms remain in
`src/sdpc`.

## Systems

| Case study | Task | Online adaptation | Safety methods |
|---|---|---:|---|
| [Double Integrator](DoubleIntegrator) | Two-dimensional obstacle avoidance | Yes | Barrier adaptation and PSF |
| [Two Tank](TwoTank) | Piecewise-constant liquid-level tracking | Yes | Box barriers and PSF |
| [Van der Pol](VanDerPol) | Two-input oscillator regulation | No | Evaluation constraints only |

The conventional one-input Van der Pol model is the relative-degree-two case study in
[`Relative_Degree_Two/VanDerPol`](../Relative_Degree_Two/VanDerPol).

## Experiment Structure

```text
<CaseStudy>/
|-- configs/       # system ID, policy, adaptation, and evaluation YAML
|-- notebooks/     # numbered interactive workflow
|-- results/       # checkpoints, metrics, trajectories, and figures
`-- README.md
```

Command-line stages live in [`scripts/<CaseStudy>/`](../scripts). Adapted systems provide
independent commands for system identification, SD-DPC training, NN-DPC training,
unconstrained adaptation, safe adaptation, and evaluation.

## Common Checkpoint Layout

```text
results/models/dynamics/run_N/saved_models/sindy.pt
results/models/policies/sd_dpc/run_N/saved_models/policy_sparse.pt
results/models/policies/nn_dpc/run_N/saved_models/policy_nn.pth
```

Run commands from the repository root after `pip install -e .`. The individual
case-study READMEs list their complete workflows.
