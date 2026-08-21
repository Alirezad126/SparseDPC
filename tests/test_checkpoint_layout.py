from pathlib import Path

from sdpc.io.checkpoints import (
    find_nn_checkpoint,
    find_policy_checkpoint,
    new_policy_run_dir,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_uniform_policy_run_layout(tmp_path):
    results = tmp_path / "results"

    sparse_run = new_policy_run_dir(results, "sparse")
    nn_run = new_policy_run_dir(results, "nn")

    assert sparse_run == results / "models" / "policies" / "sd_dpc" / "run_1"
    assert nn_run == results / "models" / "policies" / "nn_dpc" / "run_1"


def test_discovery_prefers_latest_uniform_checkpoint_over_legacy(tmp_path):
    results = tmp_path / "results"
    _touch(results / "models" / "run_12" / "SparseDPC" / "saved_models" / "policy_sparse.pt")
    _touch(results / "models" / "run_12" / "NN-DPC" / "saved_models" / "policy_nn.pth")
    sparse = _touch(
        results / "models" / "policies" / "sd_dpc" / "run_2" /
        "saved_models" / "policy_sparse.pt"
    )
    nn = _touch(
        results / "models" / "policies" / "nn_dpc" / "run_3" /
        "saved_models" / "policy_nn.pth"
    )

    assert find_policy_checkpoint(results) == sparse
    assert find_nn_checkpoint(results) == nn


def test_discovery_skips_incomplete_latest_run(tmp_path):
    results = tmp_path / "results"
    complete = _touch(
        results / "models" / "policies" / "sd_dpc" / "run_2" /
        "saved_models" / "policy_sparse.pt"
    )
    (results / "models" / "policies" / "sd_dpc" / "run_3").mkdir(parents=True)

    assert find_policy_checkpoint(results) == complete


def test_legacy_discovery_uses_numeric_run_order(tmp_path):
    results = tmp_path / "results"
    _touch(results / "models" / "run_9" / "SparseDPC" / "saved_models" / "policy_sparse.pt")
    latest = _touch(
        results / "models" / "run_10" / "SparseDPC" / "saved_models" / "policy_sparse.pt"
    )

    assert find_policy_checkpoint(results) == latest


def test_static_legacy_nn_checkpoint_remains_loadable(tmp_path):
    results = tmp_path / "results"
    legacy = _touch(
        results / "models" / "neural" / "NN-DPC" / "saved_models" / "policy_nn.pth"
    )

    assert find_nn_checkpoint(results) == legacy
