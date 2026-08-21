# Relative-Degree-Two Case Studies

This directory contains systems whose controlled output has relative degree two. The
shared implementation remains in [`src/sdpc`](../src/sdpc); this directory contains the
case-specific configurations, notebooks, checkpoints, and evaluation outputs.

| Case study | Model | Workflow |
|---|---|---|
| [Van der Pol](VanDerPol) | Conventional forced oscillator with one input | System ID, SD-DPC, NN-DPC, MPC evaluation |

The command-line entry points are in [`scripts/VanDerPol`](../scripts/VanDerPol).
