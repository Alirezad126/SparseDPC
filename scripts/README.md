# Experiment Scripts

The command-line interface is organized by case study and stage. Every entry point runs
one experiment only; there is no aggregate launcher that silently executes the complete
repository.

```text
scripts/
|-- DoubleIntegrator/
|   |-- system_id.py
|   |-- train_sd_dpc.py
|   |-- train_nn_dpc.py
|   |-- unconstrained_adaptation.py
|   |-- safe_adaptation.py
|   `-- evaluation.py
|-- TwoTank/                    # same six stages
|-- VanDerPol/                  # conventional one-input model; no adaptation
|-- VanDerPolRelativeDegreeOne/ # retained two-input model; no adaptation
`-- _stages.py                  # shared implementation, not a user entry point
```

## Usage

Run scripts from the repository root after installing the package in editable mode:

```bash
pip install -e .
python scripts/DoubleIntegrator/train_sd_dpc.py
```

All stage scripts accept the same optional overrides:

```text
--config PATH    use a different YAML file
--device DEVICE  torch device, for example cpu or cuda:0
--seed INTEGER   override the YAML seed
```

Example:

```bash
python scripts/TwoTank/safe_adaptation.py \
    --config Relative_Degree_One/TwoTank/configs/safe.yaml \
    --device cpu --seed 12
```

Each entry point selects its system and default case-study YAML before calling the shared
stage implementation in `_stages.py`. Training runs create a new `run_N` checkpoint
directory; existing models are not overwritten.
