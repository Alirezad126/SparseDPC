import ast
from pathlib import Path

import torch

from sdpc.adaptation.analytic import (
    dtheta_dx,
    dtheta_dz,
    policy_jacobians,
    sindy_step,
    sindy_step_jacobians,
    sindy_vector_jacobians,
)
from sdpc.adaptation.barrier import (
    rollout_barrier_loss,
    rollout_box_barrier_loss,
    rollout_box_relu_loss,
)
from sdpc.adaptation.jacobian import SymbolicJacobian, compute_updates_symbolic
from sdpc.adaptation.rollout import predict_rollout
from sdpc.adaptation.reference import compute_updates_discrete_ref
from sdpc.adaptation.safe import (
    SafeAdaptationConfig,
    prepare_safe_adaptation,
    run_safe_adaptation,
)
from sdpc.adaptation.safety_jacobian import SymbolicSafetyJacobian
from sdpc.adaptation.unconstrained import (
    UnconstrainedAdaptationConfig,
    prepare_unconstrained_adaptation,
    run_unconstrained_adaptation,
)
from sdpc.safety.specs import SafetySpec, box_constraints
from sdpc.sindy import CompiledFunctionLibrary, SINDyVectorized
from sdpc.registry import make_system


torch.manual_seed(3)


def _model(lib_cfg, n_out, scale=0.05):
    lib = CompiledFunctionLibrary(**lib_cfg)
    model = SINDyVectorized(lib, n_out=n_out, device=torch.device("cpu"))
    with torch.no_grad():
        for p in model.Xi:
            p.copy_(scale * torch.randn_like(p))
    return model


def _batch_diag(J, B):
    return torch.stack([J[b, :, b, :] for b in range(B)], dim=0)


def test_library_derivatives_match_autograd():
    lib = CompiledFunctionLibrary(
        n_features=2, n_control=2, include_bias=True, max_degree=3,
        add_sqrt=True, add_xu=True, add_u_prod=True, add_fourier=True,
        max_freq=2, is_policy=True,
    )
    plan = lib.compile(None)
    x = torch.tensor([[0.7, 1.2], [1.4, 0.9]], requires_grad=True)
    r = torch.tensor([[0.2, -0.3], [0.5, 0.8]], requires_grad=True)

    Jx = torch.autograd.functional.jacobian(lambda xx: lib.evaluate_plan(xx, r, plan), x, vectorize=True)
    Jr = torch.autograd.functional.jacobian(lambda rr: lib.evaluate_plan(x, rr, plan), r, vectorize=True)

    assert torch.allclose(dtheta_dx(lib, plan, x.detach(), r.detach()), _batch_diag(Jx, 2), atol=1e-5, rtol=1e-4)
    assert torch.allclose(dtheta_dz(lib, plan, x.detach(), r.detach()), _batch_diag(Jr, 2), atol=1e-5, rtol=1e-4)


def test_sindy_step_jacobians_match_autograd():
    dyn = _model(
        dict(n_features=2, n_control=2, include_bias=True, max_degree=2,
             add_sqrt=False, add_xu=True, add_u_prod=True, add_fourier=True,
             max_freq=1, is_policy=False),
        n_out=2,
    )
    x = torch.tensor([[0.4, -0.2], [0.1, 0.3]], requires_grad=True)
    u = torch.tensor([[0.2, -0.1], [0.5, 0.4]], requires_grad=True)

    Jx = torch.autograd.functional.jacobian(lambda xx: dyn(xx, u), x, vectorize=True)
    Ju = torch.autograd.functional.jacobian(lambda uu: dyn(x, uu), u, vectorize=True)
    ax, au = sindy_vector_jacobians(dyn, x.detach(), u.detach())

    assert torch.allclose(ax, _batch_diag(Jx, 2), atol=1e-5, rtol=1e-4)
    assert torch.allclose(au, _batch_diag(Ju, 2), atol=1e-5, rtol=1e-4)


def test_policy_jacobians_match_autograd():
    policy = _model(
        dict(n_features=2, n_control=2, include_bias=True, max_degree=2,
             add_sqrt=False, add_xu=True, add_u_prod=True, add_fourier=True,
             max_freq=2, is_policy=True),
        n_out=2,
    )
    x = torch.tensor([[0.2, -0.4], [0.5, 0.1]], requires_grad=True)
    r = torch.tensor([[0.3, 0.6], [-0.2, 0.4]], requires_grad=True)

    Jx = torch.autograd.functional.jacobian(lambda xx: policy(xx, r), x, vectorize=True)
    dudx, duda = policy_jacobians(policy, x.detach(), r.detach(), umin=-10.0, umax=10.0)

    assert torch.allclose(dudx, _batch_diag(Jx, 2), atol=1e-5, rtol=1e-4)
    col = 0
    for k, Xi_k in enumerate(policy.Xi):
        n = Xi_k.numel()
        theta_k = policy.library.evaluate_plan(x.detach(), r.detach(), policy._plan).index_select(
            1, policy._state_local_idx[k]
        )
        assert torch.allclose(duda[:, k, col:col + n], theta_k, atol=1e-6)
        col += n


def test_symbolic_safety_gradient_matches_autograd():
    dyn = _model(
        dict(n_features=2, n_control=2, include_bias=True, max_degree=1,
             add_sqrt=False, add_xu=False, add_u_prod=False, add_fourier=False,
             is_policy=False),
        n_out=2,
        scale=0.15,
    )
    policy_ag = _model(
        dict(n_features=2, n_control=2, include_bias=True, max_degree=1,
             add_sqrt=False, add_xu=False, add_u_prod=False, add_fourier=False,
             is_policy=True),
        n_out=2,
        scale=0.2,
    )
    policy_sym = _model(
        dict(n_features=2, n_control=2, include_bias=True, max_degree=1,
             add_sqrt=False, add_xu=False, add_u_prod=False, add_fourier=False,
             is_policy=True),
        n_out=2,
        scale=0.2,
    )
    policy_sym.load_state_dict(policy_ag.state_dict())

    spec = SafetySpec(state_constraints=box_constraints(-0.2, 0.2, idx=[0, 1], weight=1.0))
    x0 = torch.tensor([[0.45, -0.5]])
    r = torch.zeros(1, 4, 2)
    plant = lambda x, u: dyn(x, u)

    for p in policy_ag.Xi:
        p.grad = None
    x_roll, u_roll = predict_rollout(policy_ag, plant, x0, r, 4, umin=-10.0, umax=10.0, grad=True)
    loss, _, _ = rollout_barrier_loss(x_roll, spec, kind="squared_hinge")
    loss.backward()
    grad_ag = [(-p.grad.detach().clone()) for p in policy_ag.Xi]

    with torch.no_grad():
        x_roll2, u_roll2 = predict_rollout(policy_sym, plant, x0, r, 4, umin=-10.0, umax=10.0, grad=False)
    sj = SymbolicSafetyJacobian(dyn, policy_sym, umin=-10.0, umax=10.0)
    grad_sym, _ = sj.rollout_sensitivity_grad(x_roll2, u_roll2, r, spec, kind="squared_hinge")

    for ga, gs in zip(grad_ag, grad_sym):
        assert torch.allclose(gs, ga, atol=2e-5, rtol=2e-4)


def test_twotank_symbolic_straight_through_matches_autograd_at_saturation():
    system = make_system("twotank")
    dyn = system.perturbed_sindy_model({"perturbation": {"c1": 0.12, "c2": 0.02}})
    policy_ag = _model(system.policy_library_cfg(), n_out=system.nu, scale=0.0)
    policy_sym = _model(system.policy_library_cfg(), n_out=system.nu, scale=0.0)

    with torch.no_grad():
        for Xi_k, active in zip(policy_ag.Xi, policy_ag.active_idx):
            Xi_k.zero_()
            Xi_k[active.index(0)] = -0.25  # constant raw action below umin=0
    policy_sym.load_state_dict(policy_ag.state_dict())

    x0 = torch.tensor([[0.205, 0.205]])
    r = torch.full((1, 3, system.nx), 0.2)
    spec = SafetySpec(
        state_constraints=box_constraints(0.2, 0.8, idx=[0, 1], band=0.02)
    )

    def plant(x, u):
        return sindy_step(dyn, x, u, system=system, method="rk4")

    x_ag, u_ag = predict_rollout(
        policy_ag, plant, x0, r, 3,
        umin=system.umin, umax=system.umax, grad=True,
        action_gradient_mode="straight_through",
    )
    assert torch.equal(u_ag.detach(), torch.zeros_like(u_ag))
    loss_ag, _, _ = rollout_barrier_loss(x_ag[:, 1:, :], spec, kind="squared_hinge")
    loss_ag.backward()
    grad_ag = [(-p.grad.detach().clone()) for p in policy_ag.Xi]

    with torch.no_grad():
        x_sym, u_sym = predict_rollout(
            policy_sym, plant, x0, r, 3,
            umin=system.umin, umax=system.umax, grad=False,
        )
    sj = SymbolicSafetyJacobian(
        dyn, policy_sym, system.umin, system.umax,
        system=system, integration_method="rk4",
    )
    grad_exact, _ = sj.rollout_sensitivity_grad(
        x_sym, u_sym, r, spec, action_gradient_mode="exact"
    )
    grad_ste, _ = sj.rollout_sensitivity_grad(
        x_sym, u_sym, r, spec, action_gradient_mode="straight_through"
    )

    assert sum(float(g.norm()) for g in grad_exact) == 0.0
    assert sum(float(g.norm()) for g in grad_ste) > 0.0
    for ga, gs in zip(grad_ag, grad_ste):
        assert torch.allclose(gs, ga, atol=2e-5, rtol=2e-4)


def test_twotank_symbolic_leaky_straight_through_matches_autograd_at_saturation():
    system = make_system("twotank")
    dyn = system.perturbed_sindy_model({"perturbation": {"c1": 0.12, "c2": 0.02}})
    policy_ag = _model(system.policy_library_cfg(), n_out=system.nu, scale=0.0)
    policy_sym = _model(system.policy_library_cfg(), n_out=system.nu, scale=0.0)

    with torch.no_grad():
        for Xi_k, active in zip(policy_ag.Xi, policy_ag.active_idx):
            Xi_k.zero_()
            Xi_k[active.index(0)] = -0.25
    policy_sym.load_state_dict(policy_ag.state_dict())

    x0 = torch.tensor([[0.205, 0.205]])
    r = torch.full((1, 3, system.nx), 0.2)
    spec = SafetySpec(
        state_constraints=box_constraints(0.2, 0.8, idx=[0, 1], band=0.02)
    )

    def plant(x, u):
        return sindy_step(dyn, x, u, system=system, method="rk4")

    x_ag, _ = predict_rollout(
        policy_ag, plant, x0, r, 3,
        umin=system.umin, umax=system.umax, grad=True,
        action_gradient_mode="leaky_straight_through",
        action_gradient_band=0.1,
        action_gradient_leak=0.05,
    )
    loss_ag, _, _ = rollout_barrier_loss(
        x_ag[:, 1:, :], spec, kind="squared_hinge"
    )
    loss_ag.backward()
    grad_ag = [(-p.grad.detach().clone()) for p in policy_ag.Xi]

    with torch.no_grad():
        x_sym, u_sym = predict_rollout(
            policy_sym, plant, x0, r, 3,
            umin=system.umin, umax=system.umax, grad=False,
        )
    sj = SymbolicSafetyJacobian(
        dyn, policy_sym, system.umin, system.umax,
        system=system, integration_method="rk4",
    )
    grad_sym, _ = sj.rollout_sensitivity_grad(
        x_sym, u_sym, r, spec,
        action_gradient_mode="leaky_straight_through",
        action_gradient_band=0.1,
        action_gradient_leak=0.05,
    )

    assert sum(float(g.norm()) for g in grad_sym) > 0.0
    for ga, gs in zip(grad_ag, grad_sym):
        assert torch.allclose(gs, ga, atol=2e-5, rtol=2e-4)


def test_symbolic_reference_straight_through_recovers_saturated_update():
    dyn = _model(
        dict(n_features=1, n_control=1, include_bias=True, max_degree=1,
             add_sqrt=False, add_xu=False, add_u_prod=False, add_fourier=False,
             is_policy=False),
        n_out=1,
        scale=0.0,
    )
    policy = _model(
        dict(n_features=1, n_control=1, include_bias=True, max_degree=1,
             add_sqrt=False, add_xu=False, add_u_prod=False, add_fourier=False,
             is_policy=True),
        n_out=1,
        scale=0.0,
    )
    with torch.no_grad():
        dyn.Xi[0][dyn.active_idx[0].index(0)] = 0.1
        dyn.Xi[0][dyn.active_idx[0].index(2)] = 1.0
        policy.Xi[0][policy.active_idx[0].index(0)] = -0.2

    x = torch.tensor([[0.1]])
    r = torch.tensor([[0.5]])
    jac = SymbolicJacobian(dyn, policy, umin=0.0, umax=1.0)
    _, u_applied = jac._clamped_action(x, r)
    exact = compute_updates_symbolic(jac, x, r, action_gradient_mode="exact")
    ste = compute_updates_symbolic(
        jac, x, r, action_gradient_mode="straight_through"
    )
    leaky = compute_updates_symbolic(
        jac, x, r, action_gradient_mode="leaky_straight_through",
        action_gradient_band=0.1, action_gradient_leak=0.05,
    )
    exact_ag = compute_updates_discrete_ref(
        dyn, policy, x, r, umin=0.0, umax=1.0,
        action_gradient_mode="exact",
    )
    ste_ag = compute_updates_discrete_ref(
        dyn, policy, x, r, umin=0.0, umax=1.0,
        action_gradient_mode="straight_through",
    )
    leaky_ag = compute_updates_discrete_ref(
        dyn, policy, x, r, umin=0.0, umax=1.0,
        action_gradient_mode="leaky_straight_through",
        action_gradient_band=0.1, action_gradient_leak=0.05,
    )

    assert torch.equal(u_applied, torch.zeros_like(u_applied))
    assert sum(float(g.norm()) for g in exact) == 0.0
    assert sum(float(g.norm()) for g in ste) > 0.0
    assert sum(float(g.norm()) for g in exact_ag) == 0.0
    assert sum(float(g.norm()) for g in ste_ag) > 0.0
    assert 0.0 < sum(float(g.norm()) for g in leaky) < sum(float(g.norm()) for g in ste)
    assert 0.0 < sum(float(g.norm()) for g in leaky_ag) < sum(float(g.norm()) for g in ste_ag)
    for symbolic, autograd in zip(ste, ste_ag):
        assert torch.allclose(symbolic, autograd, atol=1e-6, rtol=1e-5)
    for symbolic, autograd in zip(leaky, leaky_ag):
        assert torch.allclose(symbolic, autograd, atol=1e-6, rtol=1e-5)


def test_run_safe_adaptation_symbolic_smoke():
    dyn = _model(
        dict(n_features=2, n_control=2, include_bias=True, max_degree=1,
             add_sqrt=False, add_xu=False, add_u_prod=False, add_fourier=False,
             is_policy=False),
        n_out=2,
        scale=0.05,
    )
    policy = _model(
        dict(n_features=2, n_control=2, include_bias=True, max_degree=1,
             add_sqrt=False, add_xu=False, add_u_prod=False, add_fourier=False,
             is_policy=True),
        n_out=2,
        scale=0.05,
    )
    plant = lambda x, u: dyn(x, u)
    data = {
        "xn": torch.tensor([[[0.1, -0.1]]]),
        "r": torch.zeros(1, 4, 2),
    }
    spec = SafetySpec(state_constraints=box_constraints(-1.0, 1.0, idx=[0, 1]))
    cfg = SafeAdaptationConfig(
        horizon=2,
        gamma_ref=0.01,
        gamma_safe=0.01,
        max_safety_iters=1,
        ref_backend="symbolic",
        safety_backend="symbolic",
    )
    prepared = prepare_safe_adaptation(
        policy, plant, cfg,
        umin=-10.0, umax=10.0,
        pred_plant=plant,
        derivative_model=dyn,
    )
    assert prepared.reference_jacobian is not None
    assert prepared.safety_jacobian is not None

    res = run_safe_adaptation(
        policy, plant, data, spec, cfg,
        umin=-10.0, umax=10.0,
        prepared=prepared,
    )
    assert res["x_traj"].shape == (1, 4, 2)
    assert res["u_traj"].shape == (1, 4, 2)
    assert len(res["logs"]) == 3
    assert res["timing"]["prediction_setup_s"] == 0.0
    assert res["timing"]["reference_jacobian_setup_s"] == 0.0
    assert res["timing"]["safety_jacobian_setup_s"] == 0.0


def test_preparation_is_backend_selective_for_reference_and_safety():
    dyn = _model(
        dict(n_features=2, n_control=2, include_bias=True, max_degree=1,
             add_sqrt=False, add_xu=False, add_u_prod=False, add_fourier=False,
             is_policy=False),
        n_out=2,
        scale=0.05,
    )
    policy = _model(
        dict(n_features=2, n_control=2, include_bias=True, max_degree=1,
             add_sqrt=False, add_xu=False, add_u_prod=False, add_fourier=False,
             is_policy=True),
        n_out=2,
        scale=0.05,
    )
    plant = lambda x, u: dyn(x, u)
    data = {
        "xn": torch.tensor([[[0.1, -0.1]]]),
        "r": torch.zeros(1, 3, 2),
    }

    ref_cfg = UnconstrainedAdaptationConfig(ref_backend="symbolic")
    ref_prepared = prepare_unconstrained_adaptation(
        policy, ref_cfg, umin=-10.0, umax=10.0, derivative_model=dyn,
    )
    assert ref_prepared.reference_jacobian is not None
    ref_res = run_unconstrained_adaptation(
        policy, plant, data, ref_cfg,
        umin=-10.0, umax=10.0, prepared=ref_prepared,
    )
    assert ref_res["x_traj"].shape == (1, 3, 2)
    assert ref_res["timing"]["prepared_setup_s"] == ref_prepared.setup_time_s

    autograd_cfg = SafeAdaptationConfig(
        ref_backend="autograd", safety_backend="autograd"
    )
    autograd_prepared = prepare_safe_adaptation(
        policy, plant, autograd_cfg, umin=-10.0, umax=10.0
    )
    assert autograd_prepared.reference_jacobian is None
    assert autograd_prepared.safety_jacobian is None


def test_perturbed_sindy_models_match_true_plant_steps():
    configs = {
        "twotank": {"perturbation": {"c1": 0.12, "c2": 0.02}},
        "vanderpol": {"perturbation": {"mu": 1.5}},
        "double_integrator": {
            "perturbation": {
                "A": [[1.0, 0.02], [0.02, 1.0]],
                "B": [[0.5, 0.0], [0.0, 0.5]],
            }
        },
    }
    for name, cfg in configs.items():
        system = make_system(name)
        exact = system.perturbed_sindy_model(cfg)
        exact_step = system.discrete_step(exact)
        true_step = system.perturbed_plant(cfg)
        x = torch.tensor([[0.6, 0.7], [0.3, 0.4]])
        if name == "vanderpol":
            x = torch.tensor([[0.6, -0.7], [0.3, 0.4]])
        u = torch.tensor([[0.2, 0.4], [0.6, -0.1]])
        if name == "vanderpol":
            u = torch.tensor([[0.2], [0.6]])
        if name == "twotank":
            u = torch.tensor([[0.2, 0.4], [0.6, 0.1]])
        assert torch.allclose(exact_step(x, u), true_step(x, u), atol=2e-6, rtol=1e-5), name


def test_box_relu_constraints_sensitivity_matches_autograd_euler_and_rk4():
    system = make_system("twotank")
    cfg = {"perturbation": {"c1": 0.12, "c2": 0.02}}
    dyn = system.perturbed_sindy_model(cfg)
    spec = SafetySpec(state_constraints=box_constraints(0.0, 1.0, idx=[0, 1], weight=1.0))

    for method in ("euler", "rk4"):
        policy_ag = _model(system.policy_library_cfg(), n_out=2, scale=0.15)
        policy_sym = _model(system.policy_library_cfg(), n_out=2, scale=0.15)
        policy_sym.load_state_dict(policy_ag.state_dict())

        x0 = torch.tensor([[1.2, 1.1]])
        r = torch.zeros(1, 5, 1)

        def plant(x, u, method=method):
            if method == "euler":
                return x + system.ts * dyn.ode_equations(x, u)
            k1 = dyn.ode_equations(x, u)
            k2 = dyn.ode_equations(x + 0.5 * system.ts * k1, u)
            k3 = dyn.ode_equations(x + 0.5 * system.ts * k2, u)
            k4 = dyn.ode_equations(x + system.ts * k3, u)
            return x + (system.ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        for p in policy_ag.Xi:
            p.grad = None
        x_roll, u_roll = predict_rollout(
            policy_ag, plant, x0, r, 5, umin=-10.0, umax=10.0, grad=True
        )
        loss, _ = rollout_box_relu_loss(x_roll, spec, include_x0=False)
        loss.backward()
        grad_ag = [(-p.grad.detach().clone()) for p in policy_ag.Xi]

        with torch.no_grad():
            x_roll2, u_roll2 = predict_rollout(
                policy_sym, plant, x0, r, 5, umin=-10.0, umax=10.0, grad=False
            )
        sj = SymbolicSafetyJacobian(
            dyn, policy_sym, umin=-10.0, umax=10.0,
            system=system, integration_method=method,
        )
        grad_sym, _ = sj.rollout_box_relu_sensitivity_grad(
            x_roll2, u_roll2, r, spec, normalize=False
        )

        for ga, gs in zip(grad_ag, grad_sym):
            assert torch.allclose(gs, ga, atol=5e-5, rtol=5e-4), method


def test_box_barrier_constraints_sensitivity_matches_autograd_euler_and_rk4():
    system = make_system("twotank")
    cfg = {"perturbation": {"c1": 0.12, "c2": 0.02}}
    dyn = system.perturbed_sindy_model(cfg)
    spec = SafetySpec(state_constraints=box_constraints(0.0, 1.0, idx=[0, 1], band=0.05, weight=1.0))

    for method in ("euler", "rk4"):
        policy_ag = _model(system.policy_library_cfg(), n_out=2, scale=0.15)
        policy_sym = _model(system.policy_library_cfg(), n_out=2, scale=0.15)
        policy_sym.load_state_dict(policy_ag.state_dict())

        x0 = torch.tensor([[1.2, 1.1]])
        r = torch.zeros(1, 5, 1)

        def plant(x, u, method=method):
            if method == "euler":
                return x + system.ts * dyn.ode_equations(x, u)
            k1 = dyn.ode_equations(x, u)
            k2 = dyn.ode_equations(x + 0.5 * system.ts * k1, u)
            k3 = dyn.ode_equations(x + 0.5 * system.ts * k2, u)
            k4 = dyn.ode_equations(x + system.ts * k3, u)
            return x + (system.ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        for p in policy_ag.Xi:
            p.grad = None
        x_roll, u_roll = predict_rollout(
            policy_ag, plant, x0, r, 5, umin=-10.0, umax=10.0, grad=True
        )
        loss, _ = rollout_box_barrier_loss(
            x_roll, spec, barrier_kind="squared_hinge", include_x0=False
        )
        loss.backward()
        grad_ag = [(-p.grad.detach().clone()) for p in policy_ag.Xi]

        with torch.no_grad():
            x_roll2, u_roll2 = predict_rollout(
                policy_sym, plant, x0, r, 5, umin=-10.0, umax=10.0, grad=False
            )
        sj = SymbolicSafetyJacobian(
            dyn, policy_sym, umin=-10.0, umax=10.0,
            system=system, integration_method=method,
        )
        grad_sym, _ = sj.rollout_sensitivity_grad(
            x_roll2, u_roll2, r, spec, kind="box_barrier_constraints"
        )

        for ga, gs in zip(grad_ag, grad_sym):
            assert torch.allclose(gs, ga, atol=5e-5, rtol=5e-4), method


def test_symbolic_modules_do_not_reference_autograd():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "src/sdpc/adaptation/analytic.py",
        "src/sdpc/adaptation/jacobian.py",
        "src/sdpc/adaptation/safety_jacobian.py",
    ]:
        tree = ast.parse((root / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "autograd"
