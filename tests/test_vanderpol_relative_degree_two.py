from pathlib import Path

import torch

from sdpc.config import load_config
from sdpc.registry import make_system


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "Relative_Degree_Two" / "VanDerPol"


def test_active_vanderpol_is_conventional_single_input_system():
    system = make_system("vanderpol")
    assert system.nx == 2
    assert system.nu == 1

    x = torch.tensor([[0.7, -0.4], [-0.2, 0.3]])
    u_a = torch.tensor([[0.2], [-0.1]])
    u_b = torch.tensor([[0.8], [0.5]])
    f_a = system.true_ode(x, u_a)
    f_b = system.true_ode(x, u_b)

    assert torch.allclose(f_a[:, 0], x[:, 1])
    assert torch.allclose(f_a[:, 0], f_b[:, 0])
    assert torch.allclose(f_b[:, 1] - f_a[:, 1], u_b[:, 0] - u_a[:, 0])


def test_active_vanderpol_exact_sindy_matches_true_rk4_step():
    system = make_system("vanderpol")
    cfg = {"perturbation": {"mu": 1.35}}
    model = system.perturbed_sindy_model(cfg)
    x = torch.tensor([[0.6, -0.7], [0.3, 0.4]])
    u = torch.tensor([[0.2], [0.6]])

    assert model.library.n_control == 1
    assert torch.allclose(
        system.discrete_step(model)(x, u),
        system.perturbed_plant(cfg)(x, u),
        atol=2.0e-6,
        rtol=1.0e-5,
    )


def test_active_vanderpol_system_id_data_has_one_action_channel():
    system = make_system("vanderpol")
    train_loader, _, test_data = system.make_sysid_data(
        {"nsim": 20, "nsteps": 2, "bs": 10}, torch.device("cpu")
    )
    train_batch = next(iter(train_loader))

    assert train_batch["u"].shape == (10, 2, 1)
    assert test_data["u"].shape == (10, 2, 1)


def test_legacy_two_input_vanderpol_has_explicit_registry_name():
    assert make_system("vanderpol_relative_degree_one").nu == 2


def test_relative_degree_two_case_has_training_and_fixed_evaluation_only():
    config_names = {path.name for path in (CASE_DIR / "configs").glob("*.yaml")}
    notebook_names = {path.name for path in (CASE_DIR / "notebooks").glob("*.ipynb")}
    policy = load_config(CASE_DIR / "configs" / "policy.yaml")
    nn_policy = load_config(CASE_DIR / "configs" / "policy_nn.yaml")
    evaluation = load_config(CASE_DIR / "configs" / "eval.yaml")

    assert config_names == {"sysid.yaml", "policy.yaml", "policy_nn.yaml", "eval.yaml"}
    assert notebook_names == {"01_system_id.ipynb", "02_policy.ipynb", "05_evaluation.ipynb"}
    assert policy["action_gradient_mode"] == "leaky_straight_through"
    assert policy["proximal_l1"] is True
    assert policy["lr_decay_enabled"] is True
    assert nn_policy["policy_type"] == "nn"
    assert nn_policy["nsteps"] == policy["nsteps"]
    assert evaluation["methods"] == ["mpc", "sd_dpc", "nn_dpc"]
