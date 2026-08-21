from pathlib import Path

import numpy as np
import torch

from sdpc.config import load_config
from sdpc.baselines import mpc_available, solve_mpc
from sdpc.eval import dpc_loss_reference
from sdpc.eval.metrics import terminal_tracking, violation_stats
from sdpc.registry import make_system


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIRS = {
    "DoubleIntegrator": REPO_ROOT / "Relative_Degree_One" / "DoubleIntegrator",
    "TwoTank": REPO_ROOT / "Relative_Degree_One" / "TwoTank",
    "VanDerPol": REPO_ROOT / "Relative_Degree_Two" / "VanDerPol",
}


def _configs(folder):
    config_dir = CASE_DIRS[folder] / "configs"
    return (
        load_config(config_dir / "policy.yaml"),
        load_config(config_dir / "policy_nn.yaml"),
        load_config(config_dir / "eval.yaml"),
    )


def test_eval_and_nn_inherit_double_integrator_sd_dpc_loss():
    policy, nn, evaluation = _configs("DoubleIntegrator")
    keys = ("Q_r", "Q_u", "Q_uf", "Q_dx", "Q_du", "Q_con_obs", "Q_con_x", "Q_con_u")
    assert nn["nsteps"] == evaluation["nsteps"] == policy["nsteps"] == 100
    assert {key: nn["weights"][key] for key in keys} == {
        key: policy["weights"][key] for key in keys
    }
    assert {key: evaluation["weights"][key] for key in keys} == {
        key: policy["weights"][key] for key in keys
    }
    assert evaluation["mpc"]["mode"] == "dpc_open_loop"
    assert dpc_loss_reference(make_system("double_integrator"), evaluation)["mpc_horizon"] == 100


def test_eval_and_nn_inherit_vanderpol_sd_dpc_loss():
    policy, nn, evaluation = _configs("VanDerPol")
    keys = ("Q_r", "Q_u", "Q_dx", "Q_du", "Q_con")
    assert nn["nsteps"] == evaluation["nsteps"] == policy["nsteps"] == 50
    assert {key: nn["weights"][key] for key in keys} == {
        key: policy["weights"][key] for key in keys
    }
    assert {key: evaluation["weights"][key] for key in keys} == {
        key: policy["weights"][key] for key in keys
    }
    assert evaluation["mpc"]["mode"] == "dpc_open_loop"
    assert dpc_loss_reference(make_system("vanderpol"), evaluation)["mpc_horizon"] == 50


def test_double_integrator_default_true_model_is_nominal():
    system = make_system("double_integrator")
    x = torch.tensor([[1.0, -2.0]])
    u = torch.tensor([[0.2, 0.4]])
    assert torch.allclose(system.perturbed_plant({})(x, u), x + u)


def test_vanderpol_casadi_rk4_matches_true_step():
    system = make_system("vanderpol")
    hooks = system.casadi_hooks(
        {"perturbation": system.nominal_params()}, integration_method="rk4"
    )
    if hooks is None:
        return
    x = torch.tensor([[0.7, -0.4]], dtype=torch.float32)
    u = torch.tensor([[0.2]], dtype=torch.float32)
    expected = system.true_step(x, u, system.nominal_params()).numpy().reshape(-1)
    actual = np.asarray(hooks["f_casadi"](x.numpy().reshape(-1), u.numpy().reshape(-1))).reshape(-1)
    assert np.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_double_integrator_mpc_executes_complete_open_loop_blocks():
    if not mpc_available():
        return
    system = make_system("double_integrator")
    cfg = {
        "mode": "dpc_open_loop",
        "horizon": 10,
        "nsteps": 10,
        "refstep": 1,
        "match_dpc_constraints": True,
        "enforce_state_bounds": False,
        "enforce_safety_constraints": False,
        "warm_start": True,
        "max_iter": 200,
        "ellipse": {"p": 1.0, "b": 1.0, "c": 100.0, "d": 100.0},
        "weights": {
            "Q_r": 100.0, "Q_u": 25.0, "Q_uf": 1.0,
            "Q_dx": 0.0, "Q_du": 10.0,
            "Q_con_obs": 200.0, "Q_con_x": 1.0, "Q_con_u": 1.0,
        },
        "perturbation": system.nominal_params(),
    }
    data = {
        "xn": torch.zeros(1, 1, 2),
        "r": torch.cat([torch.zeros(1, 5, 2), torch.ones(1, 21, 2)], dim=1),
    }
    result = solve_mpc(system, data, cfg)
    final_error = torch.linalg.vector_norm(result["x_traj"][0, -1] - data["r"][0, -1])
    assert result["mpc_mode"] == "dpc_open_loop"
    assert result["mpc_horizon"] == cfg["nsteps"]
    assert result["num_solves"] == 3
    assert np.array_equal(result["block_starts"], [0, 5, 15])
    assert np.array_equal(result["applied_block_lengths"], [5, 10, 10])
    assert float(final_error) <= 1.0e-2


def test_double_integrator_hard_mpc_obstacle_has_positive_clearance():
    if not mpc_available():
        return
    system = make_system("double_integrator")
    cfg = {
        "mode": "dpc_open_loop", "horizon": 20, "nsteps": 20, "refstep": 1,
        "match_dpc_constraints": True, "enforce_state_bounds": True,
        "enforce_safety_constraints": True, "constraint_margin": 1.0e-4,
        "max_iter": 1000, "tol": 1.0e-8, "acceptable_tol": 1.0e-7,
        "ellipse": {"p": 2.0, "b": 1.0, "c": 0.0, "d": 0.0},
        "weights": {
            "Q_r": 100.0, "Q_u": 25.0, "Q_uf": 1.0,
            "Q_dx": 0.0, "Q_du": 10.0,
            "Q_con_obs": 200.0, "Q_con_x": 1.0, "Q_con_u": 1.0,
            "Q_con_du": 1.0,
        },
        "bands": {"obstacle": 0.0, "box": 0.0},
        "perturbation": system.nominal_params(),
    }
    data = {
        "xn": torch.tensor([[[2.0, -2.0]]]),
        "r": torch.tensor([[[-2.0, 2.0]]]).repeat(1, 21, 1),
    }
    result = solve_mpc(system, data, cfg)
    stats = violation_stats(result["x_traj"], system.safety_specs(cfg), result["u_traj"])
    assert stats["num_state_violations"] == 0
    assert stats["min_state_margin"] >= 0.5 * cfg["constraint_margin"]


def test_terminal_metrics_report_final_error_and_sustained_arrival():
    x = torch.tensor([[[1.0], [0.2], [0.04], [0.03], [0.02]]])
    r = torch.zeros_like(x)
    metrics = terminal_tracking(x, r, reach_tolerance=0.05, reach_hold_steps=3)
    assert np.isclose(metrics["final_tracking_mse"], 0.02 ** 2)
    assert metrics["reached_reference"] == 1.0
    assert metrics["steps_to_reach"] == 2.0
    assert metrics["final_reference_start_step"] == 0.0
    assert metrics["final_reference_stabilization_step"] == 2.0


def test_terminal_metrics_report_absolute_stabilization_after_reference_change():
    x = torch.tensor([[[0.0], [0.0], [1.3], [1.04], [1.03], [1.02]]])
    r = torch.tensor([[[0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]])
    metrics = terminal_tracking(x, r, reach_tolerance=0.05, reach_hold_steps=3)
    assert metrics["final_reference_start_step"] == 2.0
    assert metrics["steps_to_reach"] == 1.0
    assert metrics["final_reference_stabilization_step"] == 3.0
