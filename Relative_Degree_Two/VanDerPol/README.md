# Van der Pol: Relative Degree Two

This case study uses the conventional single-input controlled oscillator:

```text
dx0/dt = x1
dx1/dt = mu (1 - x0^2) x1 - x0 + u,    mu = 1.
```

For the controlled output `y = x0`, the input first appears in the second derivative.
The learned policy is therefore a one-output state-feedback controller that regulates
the oscillator to the origin.

## Included Stages

- Sparse SINDy system identification
- STE-based SD-DPC policy training
- Matched NN-DPC policy training
- Fixed-controller MPC/SD-DPC/NN-DPC evaluation

This example deliberately has no online-adaptation stage.

## Run

```bash
python scripts/VanDerPol/system_id.py
python scripts/VanDerPol/train_sd_dpc.py
python scripts/VanDerPol/train_nn_dpc.py
python scripts/VanDerPol/evaluation.py
```

The numbered notebooks are in [`notebooks/`](notebooks), and their YAML inputs are in
[`configs/`](configs). New checkpoints follow the repository-wide model layout under
`results/models/`.
