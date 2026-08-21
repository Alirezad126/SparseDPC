import math

import numpy as np
import torch

from sdpc.adaptation import PSFConfig
from sdpc.adaptation.psf import _constraint_buffer, _state_margin_casadi
from sdpc.eval.metrics import adaptation_summary, control_smoothness, violation_stats
from sdpc.registry import make_system
from sdpc.safety import ControlRateConstraint, SafetySpec


def test_control_smoothness_metrics():
    u = torch.tensor([[[1.0, 2.0], [2.0, 2.0], [2.0, 4.0]]])
    metrics = control_smoothness(u)
    assert math.isclose(metrics["mean_du_l2"], 1.5)
    assert math.isclose(metrics["max_du_l2"], 2.0)
    assert math.isclose(metrics["control_smoothness_rms"], math.sqrt(2.5), rel_tol=1e-6)
    assert math.isclose(metrics["control_total_variation_l1"], 3.0)


def test_psf_runtime_summary_separates_filter_time():
    logs = [
        {"step_time": 0.3, "filter_time": 0.2, "solver_iters": 4, "correction_norm": 0.1},
        {"step_time": 0.5, "filter_time": 0.4, "solver_iters": 6, "correction_norm": 0.3},
    ]
    metrics = adaptation_summary(logs)
    assert math.isclose(metrics["runtime_total_s"], 0.8)
    assert math.isclose(metrics["runtime_per_step_s"], 0.4)
    assert math.isclose(metrics["filter_runtime_total_s"], 0.6)
    assert math.isclose(metrics["filter_runtime_per_step_s"], 0.3)
    assert math.isclose(metrics["mean_filter_iters"], 5.0)
    assert math.isclose(metrics["mean_filter_correction"], 0.2)


def test_control_rate_violation_uses_numerical_tolerance():
    spec = SafetySpec(control_rate=ControlRateConstraint(du_max=0.3))
    x = torch.zeros(1, 2, 1)
    u = torch.tensor([[[0.0], [0.30000001]]])
    metrics = violation_stats(x, spec, u)
    assert metrics["num_control_rate_violations"] == 0


def test_psf_constraint_metadata_and_group_buffers():
    system = make_system("double_integrator", device=torch.device("cpu"))
    cfg = {
        "ellipse": {"p": 8.0, "b": 9.0, "c": 0.0, "d": 0.0, "theta": -0.4},
        "bands": {},
    }
    spec = system.safety_specs(cfg)
    psf_cfg = PSFConfig(buffers={"obstacle": 0.5, "box": 0.1})
    assert _constraint_buffer(spec.state_constraints[0], psf_cfg) == 0.5
    assert _constraint_buffer(spec.state_constraints[1], psf_cfg) == 0.1

    import casadi as ca

    x = ca.DM([2.0, 1.0])
    casadi_margin = float(_state_margin_casadi(spec.state_constraints[0], x))
    torch_margin = float(spec.state_constraints[0].margin(torch.tensor([[2.0, 1.0]])).item())
    assert math.isclose(casadi_margin, torch_margin, rel_tol=1e-6, abs_tol=1e-6)


def test_casadi_dynamics_match_perturbed_torch_plants():
    import casadi as ca

    cases = [
        (
            "double_integrator",
            {"perturbation": {"A": [[1.0, 0.03], [-0.01, 1.0]], "B": [[0.4, 0.0], [0.0, 0.6]]}},
            "euler",
            torch.tensor([[1.2, -0.7]]),
            torch.tensor([[0.2, -0.1]]),
        ),
        (
            "twotank",
            {"perturbation": {"c1": 0.12, "c2": 0.02}},
            "rk4",
            torch.tensor([[0.6, 0.4]]),
            torch.tensor([[0.7, 0.3]]),
        ),
    ]
    for name, cfg, method, x, u in cases:
        system = make_system(name, device=torch.device("cpu"))
        f_casadi = system.casadi_hooks(cfg, integration_method=method)["f_casadi"]
        xs = ca.MX.sym("x", system.nx)
        us = ca.MX.sym("u", system.nu)
        fn = ca.Function(f"f_{name}", [xs, us], [f_casadi(xs, us)])
        actual = np.asarray(fn(x.numpy().reshape(-1), u.numpy().reshape(-1))).reshape(-1)
        expected = system.perturbed_plant(cfg)(x, u).detach().numpy().reshape(-1)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_twotank_psf_warm_start_holds_near_lower_boundary():
    cfg = {"perturbation": {"c1": 0.12, "c2": 0.02}}
    system = make_system("twotank", device=torch.device("cpu"))
    hooks = system.casadi_hooks(cfg, integration_method="rk4")
    x = np.asarray([4.4e-5, 0.113], dtype=float)
    u = hooks["psf_u_guess"](x)
    x_next = system.perturbed_plant(cfg)(
        torch.tensor(x, dtype=torch.float64).unsqueeze(0),
        torch.tensor(u, dtype=torch.float64).unsqueeze(0),
    )

    np.testing.assert_allclose(x_next.squeeze(0).numpy(), x, rtol=0.0, atol=1.0e-10)
    assert np.all((u >= system.umin) & (u <= system.umax))
