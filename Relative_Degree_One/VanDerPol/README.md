# Van der Pol: Relative Degree One

This directory retains the two-input controlled Van der Pol formulation:

```text
dx0/dt = x1 + u0
dx1/dt = mu (1 - x0^2) x1 - x0 + u1
```

It is registered as `vanderpol_relative_degree_one`, which keeps its two-output policy
and checkpoints separate from the conventional one-input system. The workflow covers
sparse system identification, SD-DPC and NN-DPC policy training, and matched fixed-policy
evaluation; online adaptation is intentionally excluded.

## Run

```bash
python scripts/VanDerPolRelativeDegreeOne/system_id.py
python scripts/VanDerPolRelativeDegreeOne/train_sd_dpc.py
python scripts/VanDerPolRelativeDegreeOne/train_nn_dpc.py
python scripts/VanDerPolRelativeDegreeOne/evaluation.py
```

The interactive workflow is in [`notebooks/`](notebooks), with experiment settings in
[`configs/`](configs). For the conventional single-input oscillator, use
[`Relative_Degree_Two/VanDerPol`](../../Relative_Degree_Two/VanDerPol).
