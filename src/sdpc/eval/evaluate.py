"""Evaluation orchestration: compare methods across seeds and save paper-ready tables.

Runs the comparisons required for Sec. 5:

* nominal sparse policy vs. model mismatch,
* unconstrained adaptation (autograd vs. symbolic Jacobian),
* safe adaptation (barrier: squared-hinge vs. relaxed-log),

over a set of random seeds, computing every metric in :mod:`sdpc.eval.metrics`, then writing
per-seed and aggregated CSV/JSON plus a LaTeX summary table. MPC (CasADi) is included when
available. The set of methods is data-driven via the eval config.
"""
from __future__ import annotations

import copy
import math
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..adaptation.safe import SafeAdaptationConfig, run_safe_adaptation
from ..adaptation.analytic import sindy_step
from ..adaptation.unconstrained import (
    UnconstrainedAdaptationConfig,
    run_unconstrained_adaptation,
)
from ..baselines import solve_mpc
from ..data.reference import build_reference
from ..io.run import save_json
from .metrics import compute_metrics, policy_forward_latency
from .rollout import rollout_closed_loop

__all__ = [
    "sample_scenario", "default_methods", "dpc_loss_reference", "evaluate", "aggregate"
]


def sample_scenario(system, cfg: Dict, seed: int, device) -> Dict[str, torch.Tensor]:
    """Draw one (initial state, reference) test scenario for ``seed``.

    Obstacle systems sample positions outside the keep-out ellipse; box systems sample a
    random start and either a constant reference target (``ref_target``, the default) or a
    longer, changing reference trajectory built from ``cfg['reference']``. Box systems
    may express starts and references as fractions of configured ``xmin``/``xmax`` limits.
    See
    :func:`sdpc.data.reference.build_reference` for the supported ``kind`` values
    (equally divided levels, a hand-written schedule, or a signal generator).
    """
    nsteps = cfg.get("eval_nsteps", cfg.get("nsteps", 100))
    ell = system.obstacle(cfg) if system.obstacle(cfg) else None
    if ell is not None:
        from ..plotting.trajectories import make_test_sample_discrete

        data = make_test_sample_discrete(
            nx=system.nx, nsteps=nsteps, device=device, seed=seed,
            ellipse={**ell, "theta": ell.get("theta", 0.0)},
            init_rect=_square(tuple(cfg["init_center"]), cfg["init_side"]),
            ref_rect=_square(tuple(cfg["ref_center"]), cfg["ref_side"]),
        )
        return data

    xmin = float(cfg.get("xmin", cfg.get("safe_xmin", system.xmin)))
    xmax = float(cfg.get("xmax", cfg.get("safe_xmax", system.xmax)))
    if not float(system.xmin) <= xmin < xmax <= float(system.xmax):
        raise ValueError(
            f"scenario bounds must satisfy {system.xmin} <= xmin < xmax <= {system.xmax}; "
            f"got xmin={xmin}, xmax={xmax}"
        )

    span = xmax - xmin
    initial_range = cfg.get("initial_fraction_range")
    if initial_range is not None:
        if len(initial_range) != 2:
            raise ValueError("initial_fraction_range must have two values")
        lo = xmin + span * float(initial_range[0])
        hi = xmin + span * float(initial_range[1])
    else:
        lo = float(cfg.get("x0_min", xmin))
        hi = float(cfg.get("x0_max", xmax))
    if not xmin <= lo < hi <= xmax:
        raise ValueError(f"initial-state range [{lo}, {hi}] must lie inside [{xmin}, {xmax}]")

    g = torch.Generator(device=device).manual_seed(seed)
    x0 = lo + (hi - lo) * torch.rand(system.nx, generator=g, device=device)

    box_band = float(cfg.get("bands", {}).get("box", 0.0))
    psf_cfg = cfg.get("psf", {})
    psf_buffer = float(
        psf_cfg.get("buffers", {}).get("box", psf_cfg.get("default_buffer", 0.0))
    )
    clearance = max(box_band, psf_buffer)

    ref_cfg = cfg.get("reference")
    if ref_cfg is not None:
        r_traj = build_reference(
            system.nx, nsteps, ref_cfg, seed=seed, device=device,
            bounds=(xmin, xmax), clearance=clearance,
        )
    else:
        ref = cfg.get("ref_target", [0.5 * (xmin + xmax)] * system.nx)
        r_traj = build_reference(
            system.nx, nsteps, {"kind": "constant", "level": ref}, device=device,
            bounds=(xmin, xmax), clearance=clearance,
        )

    return {
        "xn": x0.reshape(1, 1, system.nx),
        "r": r_traj,
        "state_bounds": torch.tensor([xmin, xmax], dtype=torch.float32, device=device),
    }


def dpc_loss_reference(system, cfg: Dict) -> Dict:
    """Return and validate the SD-DPC loss shared by SD-DPC, NN-DPC, and MPC."""
    spec = system.dpc_loss_spec(cfg)
    mpc_horizon = int(cfg.get("mpc", {}).get("horizon", spec["horizon"]))
    if mpc_horizon != int(spec["horizon"]):
        raise ValueError(
            f"MPC horizon {mpc_horizon} differs from SD-DPC horizon {spec['horizon']}"
        )
    return {**spec, "mpc_horizon": mpc_horizon}


def default_methods(
    system,
    sindy,
    policy,
    cfg: Dict,
    device,
    *,
    nn_policy=None,
) -> Dict[str, Callable]:
    """Return a mapping ``method_name -> fn(data) -> result-dict`` for the eval config."""
    nominal_plant = system.discrete_step(sindy)
    perturbed = system.perturbed_plant(cfg)
    exact_sindy = system.perturbed_sindy_model(cfg)
    evaluation_plant_name = str(cfg.get("evaluation_plant", "nominal")).lower()
    if evaluation_plant_name == "nominal":
        evaluation_plant = lambda x, u: system.true_step(x, u, system.nominal_params())
    elif evaluation_plant_name in {"perturbed", "deployment"}:
        evaluation_plant = perturbed
    else:
        raise ValueError("evaluation_plant must be 'nominal' or 'perturbed'")
    umin, umax = system.umin, system.umax
    scale = cfg.get("action_scale", 1.0)
    spec = system.safety_specs(cfg)

    unc = cfg.get("unconstrained", {})
    safe = cfg.get("safe", {})

    def _nominal(data):
        res = rollout_closed_loop(policy, nominal_plant, data, umin=umin, umax=umax, action_scale=scale)
        return {**res, "policy": policy, "spec": spec}

    def _fixed_controller(controller):
        def run(data):
            res = rollout_closed_loop(
                controller, evaluation_plant, data,
                umin=umin, umax=umax, action_scale=scale,
            )
            return {**res, "policy": controller, "spec": spec}
        return run

    def _mpc(data):
        mpc_cfg = {**cfg, **cfg.get("mpc", {})}
        if evaluation_plant_name == "nominal":
            mpc_cfg["perturbation"] = system.nominal_params()
        res = solve_mpc(system, data, mpc_cfg)
        return {**res, "policy": None, "spec": spec}

    def _mismatch(data):
        res = rollout_closed_loop(policy, perturbed, data, umin=umin, umax=umax, action_scale=scale)
        return {**res, "policy": policy, "spec": spec}

    def _unconstrained(backend):
        def run(data):
            pol = copy.deepcopy(policy)
            cfg_u = UnconstrainedAdaptationConfig(
                gamma_ref=unc.get("gamma_ref", 0.1), clip_update=unc.get("clip_update", 0.5),
                action_scale=scale, ref_backend=backend,
                action_gradient_mode=unc.get("action_gradient_mode", "straight_through"),
                action_gradient_band=unc.get("action_gradient_band", 0.1),
                action_gradient_leak=unc.get("action_gradient_leak", 0.05),
                integration_method=unc.get("integration_method", safe.get("integration_method", "rk4")),
                adaptive_gamma=unc.get("adaptive_gamma", False),
            )
            res = run_unconstrained_adaptation(
                pol, perturbed, data, cfg_u, umin=umin, umax=umax,
                derivative_model=exact_sindy, system=system,
            )
            return {**res, "r_traj": data["r"], "policy": pol, "spec": spec}
        return run

    def _safe(kind):
        def run(data):
            pol = copy.deepcopy(policy)
            cfg_s = SafeAdaptationConfig(
                horizon=safe.get("horizon", 20), gamma_ref=safe.get("gamma_ref", 0.05),
                gamma_safe=safe.get("gamma_safe", 0.05), clip_update=safe.get("clip_update", 0.5),
                max_safety_iters=safe.get("max_safety_iters", 30), barrier_kind=kind,
                action_scale=scale, ref_backend=safe.get("ref_backend", "autograd"),
                integration_method=safe.get("integration_method", "rk4"),
                action_gradient_mode=safe.get("action_gradient_mode", "straight_through"),
                action_gradient_band=safe.get("action_gradient_band", 0.1),
                action_gradient_leak=safe.get("action_gradient_leak", 0.05),
                adaptive_gamma=safe.get("adaptive_gamma", False),
            )
            exact_plant = lambda x, u: sindy_step(
                exact_sindy, x, u, system=system, method=cfg_s.integration_method
            )
            res = run_safe_adaptation(
                pol, perturbed, data, spec, cfg_s, umin=umin, umax=umax,
                pred_plant=exact_plant, derivative_model=exact_sindy, system=system,
            )
            return {**res, "r_traj": data["r"], "policy": pol, "spec": spec}
        return run

    methods: Dict[str, Callable] = {}
    for name in cfg.get("methods", ["nominal", "mismatch", "unconstrained_autograd", "safe_barrier"]):
        if name == "mpc":
            dpc_loss_reference(system, cfg)
            methods[name] = _mpc
        elif name == "sd_dpc":
            methods[name] = _fixed_controller(policy)
        elif name == "nn_dpc":
            if nn_policy is None:
                raise ValueError("method 'nn_dpc' requires evaluate(..., nn_policy=<NN-DPC policy>)")
            methods[name] = _fixed_controller(nn_policy)
        elif name == "nominal":
            methods[name] = _nominal
        elif name == "mismatch":
            methods[name] = _mismatch
        elif name == "unconstrained_autograd":
            methods[name] = _unconstrained("autograd")
        elif name == "unconstrained_symbolic":
            methods[name] = _unconstrained("symbolic")
        elif name == "safe_barrier":
            methods[name] = _safe("squared_hinge")
        elif name == "safe_relaxed_log":
            methods[name] = _safe("relaxed_log")
    return methods


def evaluate(
    system,
    sindy,
    policy,
    cfg: Dict,
    device,
    out_dir: Optional[Path] = None,
    *,
    nn_policy=None,
) -> Dict:
    """Run every configured method across seeds; return and (optionally) save metrics."""
    seeds = cfg.get("seeds", list(range(cfg.get("n_seeds", 10))))
    methods = default_methods(system, sindy, policy, cfg, device, nn_policy=nn_policy)

    per_seed: Dict[str, List[Dict]] = {m: [] for m in methods}
    fixed_latency: Dict[str, Dict[str, float]] = {}
    examples: Dict[str, Dict[str, torch.Tensor]] = {}
    example_scenario = None
    for seed_index, seed in enumerate(seeds):
        data = sample_scenario(system, cfg, seed, device)
        if seed_index == 0:
            example_scenario = data
        for name, fn in methods.items():
            res = fn(data)
            if seed_index == 0:
                examples[name] = {
                    "x_traj": res["x_traj"],
                    "u_traj": res["u_traj"],
                }
            metrics = compute_metrics(
                res["x_traj"], res["u_traj"], res["r_traj"],
                spec=res.get("spec"), policy=res.get("policy"), logs=res.get("logs"),
                reach_tolerance=float(cfg.get("reach_tolerance", 0.05)),
                reach_hold_steps=int(cfg.get("reach_hold_steps", 5)),
            )
            inference_cfg = cfg.get("inference_benchmark", {})
            if res.get("policy") is not None:
                if name in {"sd_dpc", "nn_dpc", "nominal", "mismatch"}:
                    if name not in fixed_latency:
                        fixed_latency[name] = policy_forward_latency(
                            res["policy"], res["x_traj"], res["r_traj"], **inference_cfg
                        )
                    metrics.update(fixed_latency[name])
                else:
                    metrics.update(policy_forward_latency(
                        res["policy"], res["x_traj"], res["r_traj"], **inference_cfg
                    ))
            if "solve_time_s" in res:
                metrics.update({
                    "mpc_solve_time_s": float(res["solve_time_s"]),
                    "mpc_solve_per_step_ms": 1.0e3 * float(res["per_step_s"]),
                    "mpc_setup_time_s": float(res.get("setup_time_s", 0.0)),
                    "mpc_iterations_per_solve": float(res.get("solver_iterations_mean", 0.0)),
                })
            elif "rollout_time_s" in res:
                metrics["closed_loop_rollout_time_s"] = float(res["rollout_time_s"])
            metrics["seed"] = seed
            metrics.pop("sparsity_per_output", None)
            per_seed[name].append(metrics)

    summary = aggregate(per_seed)
    if out_dir is not None:
        out_dir = Path(out_dir)
        save_json(out_dir / "metrics_per_seed.json", per_seed)
        save_json(out_dir / "metrics_summary.json", summary)
        _write_csv(out_dir / "metrics_summary.csv", summary)
        _write_latex(out_dir / "metrics_table.tex", summary)
    return {
        "per_seed": per_seed,
        "summary": summary,
        "dpc_loss_reference": dpc_loss_reference(system, cfg),
        "nn_checkpoint_loss_match": (
            getattr(nn_policy, "dpc_loss_reference_verified", None)
            if nn_policy is not None else None
        ),
        "example_scenario": example_scenario,
        "examples": examples,
    }


def aggregate(per_seed: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
    """Mean/std of each numeric metric per method across seeds."""
    out: Dict[str, Dict[str, float]] = {}
    for method, rows in per_seed.items():
        keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k != "seed"}
        stats: Dict[str, float] = {}
        for k in sorted(keys):
            vals = [
                float(r[k]) for r in rows
                if k in r and isinstance(r[k], (int, float)) and math.isfinite(float(r[k]))
            ]
            if vals:
                stats[f"{k}_mean"] = statistics.mean(vals)
                stats[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out[method] = stats
    return out


def _write_csv(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for s in summary.values() for k in s})
    lines = ["method," + ",".join(cols)]
    for method, s in summary.items():
        lines.append(method + "," + ",".join(f"{s.get(c, ''):.6g}" if c in s else "" for c in cols))
    path.write_text("\n".join(lines) + "\n")


def _write_latex(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_metrics = ["final_tracking_mse_mean", "steps_to_reach_mean",
                   "total_control_action_l1_mean", "control_smoothness_rms_mean",
                   "num_violations_mean",
                   "min_safety_margin_mean", "policy_forward_per_step_ms_mean",
                   "runtime_per_step_s_mean", "sparsity_total_terms_mean"]
    header = ["Method", "Final MSE", "Steps to reach", "Ctrl $\\Sigma\\|u\\|_1$",
              "RMS $\\|\\Delta u\\|$", "\\#Viol", "Min margin", "Policy ms/fwd",
              "Online s/step", "\\#Terms"]
    rows = [" & ".join(header) + " \\\\"]
    for method, s in summary.items():
        cells = [method.replace("_", "\\_")]
        for k in key_metrics:
            v = s.get(k)
            cells.append("--" if v is None else (f"{v:.3g}"))
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    path.write_text(
        "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}\n\\toprule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n"
    )


def _square(center, side):
    cx, cy = center
    half = side / 2.0
    return ((cx - half, cy - half), (cx + half, cy + half))
