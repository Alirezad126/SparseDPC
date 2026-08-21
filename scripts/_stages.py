"""Shared implementations behind the case-specific command-line scripts.

The public entry points live in ``scripts/<CaseStudy>/``. Each one fixes the system,
stage, and default YAML config; this module only avoids duplicating stage internals.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402

from sdpc.adaptation import (  # noqa: E402
    SafeAdaptationConfig,
    UnconstrainedAdaptationConfig,
    run_safe_adaptation,
    run_unconstrained_adaptation,
)
from sdpc.adaptation.analytic import sindy_step  # noqa: E402
from sdpc.config import load_config, seed_everything  # noqa: E402
from sdpc.eval import compute_metrics, evaluate, sample_scenario  # noqa: E402
from sdpc.io import (  # noqa: E402
    CustomLogger,
    find_dynamics_checkpoint,
    find_nn_checkpoint,
    find_policy_checkpoint,
    new_policy_run_dir,
    new_run_dir,
    save_json,
    snapshot_config,
)
from sdpc.plotting import format_metrics_table  # noqa: E402
from sdpc.registry import make_system  # noqa: E402
from sdpc.sindy import (  # noqa: E402
    CompiledFunctionLibrary,
    SINDyVectorized,
    load_model,
    save_model,
)
from sdpc.training import (  # noqa: E402
    build_nn_policy,
    load_nn_policy,
    train_policy,
    train_sysid,
)


def _context(
    system_name: str,
    case_dir: Path,
    config_name: str,
    description: str,
) -> Tuple[argparse.Namespace, Dict]:
    default_config = case_dir / "configs" / config_name
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"stage YAML (default: {default_config})",
    )
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    parser.add_argument("--seed", type=int, default=None, help="override the YAML seed")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    cfg = load_config(config_path)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    seed_everything(seed)
    device = torch.device(args.device)
    system = make_system(system_name, device=device)
    return args, {
        "cfg": cfg,
        "seed": seed,
        "device": device,
        "system": system,
        "case_dir": case_dir,
        "results_dir": case_dir / "results",
    }


def _load_sindy(results_dir: Path, cfg: Dict, device):
    return load_model(find_dynamics_checkpoint(results_dir, cfg), device=device)


def _load_policy(results_dir: Path, cfg: Dict, device):
    return load_model(find_policy_checkpoint(results_dir, cfg), device=device)


def run_system_id(system_name: str, case_dir: Path) -> None:
    _, ctx = _context(system_name, case_dir, "sysid.yaml", "SINDy system identification")
    cfg, system, device = ctx["cfg"], ctx["system"], ctx["device"]

    library = CompiledFunctionLibrary(**system.sindy_library_cfg())
    sindy = SINDyVectorized(library=library, n_out=system.nx, device=device)
    train_loader, dev_loader, _ = system.make_sysid_data(cfg, device)

    run_dir = new_run_dir(ctx["results_dir"] / "models" / "dynamics")
    logger = CustomLogger(
        args=None,
        savedir=str(run_dir / "logs"),
        verbosity=cfg.get("verbosity", 100),
        stdout=["train_loss", "dev_loss"],
    )
    train_sysid(system, sindy, train_loader, dev_loader, cfg, device, logger=logger)

    model_path = run_dir / "saved_models" / "sindy.pt"
    save_model(sindy, model_path)
    snapshot_config(run_dir, cfg, ctx["seed"])
    print("\nIdentified dynamics:")
    sindy.pretty_print()
    print(f"\nSaved SINDy model to {model_path}")


def run_policy_training(system_name: str, case_dir: Path, policy_type: str) -> None:
    if policy_type not in {"sparse", "nn"}:
        raise ValueError("policy_type must be 'sparse' or 'nn'")
    config_name = "policy.yaml" if policy_type == "sparse" else "policy_nn.yaml"
    label = "SD-DPC" if policy_type == "sparse" else "NN-DPC"
    _, ctx = _context(system_name, case_dir, config_name, f"{label} policy training")
    cfg, system, device = ctx["cfg"], ctx["system"], ctx["device"]
    sindy = _load_sindy(ctx["results_dir"], cfg, device)

    if policy_type == "sparse":
        library = CompiledFunctionLibrary(**system.policy_library_cfg())
        policy = SINDyVectorized(
            library=library,
            n_out=system.nu,
            policy_name=[f"u{i}" for i in range(system.nu)],
            device=device,
            seed=cfg.get("policy_seed", 0),
        )
        init_scale = cfg.get("init_scale")
        if init_scale is not None:
            for coefficients in policy.Xi:
                coefficients.data.mul_(init_scale)
    else:
        policy = build_nn_policy(system, cfg).to(device)

    train_loader, dev_loader = system.make_policy_data(cfg, device)
    run_dir = new_policy_run_dir(ctx["results_dir"], policy_type)
    logger = CustomLogger(
        args=None,
        savedir=str(run_dir / "logs"),
        verbosity=cfg.get("verbosity", 100),
        stdout=cfg.get("logger_metrics", ["train_loss", "dev_loss"]),
    )
    train_policy(
        system,
        sindy,
        policy,
        train_loader,
        dev_loader,
        cfg,
        device,
        logger=logger,
        action_scale=cfg.get("action_scale", 1.0),
    )

    if policy_type == "sparse":
        model_path = run_dir / "saved_models" / "policy_sparse.pt"
        save_model(policy, model_path)
        print("\nLearned sparse policy:")
        policy.pretty_print()
    else:
        model_path = run_dir / "saved_models" / "policy_nn.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), model_path)
    snapshot_config(run_dir, cfg, ctx["seed"])
    print(f"\nSaved {label} policy to {model_path}")


def run_unconstrained(system_name: str, case_dir: Path) -> None:
    _, ctx = _context(
        system_name,
        case_dir,
        "unconstrained.yaml",
        "Unconstrained online adaptation",
    )
    cfg, system, device = ctx["cfg"], ctx["system"], ctx["device"]
    policy = _load_policy(ctx["results_dir"], cfg, device)
    exact_sindy = system.perturbed_sindy_model(cfg)
    plant = system.perturbed_plant(cfg)
    data = sample_scenario(system, cfg, ctx["seed"], device)

    adaptation_cfg = dict(cfg.get("unconstrained", {}))
    adaptation_cfg.setdefault("action_scale", cfg.get("action_scale", 1.0))
    result = run_unconstrained_adaptation(
        policy,
        plant,
        data,
        UnconstrainedAdaptationConfig(**adaptation_cfg),
        umin=system.umin,
        umax=system.umax,
        derivative_model=exact_sindy,
        system=system,
    )

    spec = system.safety_specs(cfg)
    metrics = compute_metrics(
        result["x_traj"],
        result["u_traj"],
        data["r"],
        spec=spec,
        policy=policy,
        logs=result["logs"],
    )
    run_dir = new_run_dir(ctx["results_dir"] / "adaptation" / "unconstrained")
    torch.save(
        {"x_traj": result["x_traj"], "u_traj": result["u_traj"], "r": data["r"]},
        run_dir / "trajectories.pt",
    )
    save_json(run_dir / "logs.json", result["logs"])
    save_json(run_dir / "metrics.json", metrics)
    snapshot_config(run_dir, cfg, ctx["seed"])
    _print_metrics("Unconstrained adaptation", metrics, run_dir)


def run_safe(system_name: str, case_dir: Path) -> None:
    _, ctx = _context(system_name, case_dir, "safe.yaml", "Safe online adaptation")
    cfg, system, device = ctx["cfg"], ctx["system"], ctx["device"]
    policy = _load_policy(ctx["results_dir"], cfg, device)
    exact_sindy = system.perturbed_sindy_model(cfg)
    plant = system.perturbed_plant(cfg)
    data = sample_scenario(system, cfg, ctx["seed"], device)
    spec = system.safety_specs(cfg)

    adaptation_cfg = dict(cfg.get("safe", {}))
    adaptation_cfg.setdefault("action_scale", cfg.get("action_scale", 1.0))
    safe_cfg = SafeAdaptationConfig(**adaptation_cfg)
    prediction_plant = lambda x, u: sindy_step(
        exact_sindy,
        x,
        u,
        system=system,
        method=safe_cfg.integration_method,
    )
    result = run_safe_adaptation(
        policy,
        plant,
        data,
        spec,
        safe_cfg,
        umin=system.umin,
        umax=system.umax,
        pred_plant=prediction_plant,
        derivative_model=exact_sindy,
        system=system,
    )

    metrics = compute_metrics(
        result["x_traj"],
        result["u_traj"],
        data["r"],
        spec=spec,
        policy=policy,
        logs=result["logs"],
    )
    run_dir = new_run_dir(ctx["results_dir"] / "adaptation" / "safe")
    torch.save(
        {"x_traj": result["x_traj"], "u_traj": result["u_traj"], "r": data["r"]},
        run_dir / "trajectories.pt",
    )
    save_json(run_dir / "safety_logs.json", result["logs"])
    save_json(run_dir / "metrics.json", metrics)
    snapshot_config(run_dir, cfg, ctx["seed"])
    _print_metrics("Safe adaptation", metrics, run_dir)


def run_evaluation(system_name: str, case_dir: Path) -> None:
    _, ctx = _context(system_name, case_dir, "eval.yaml", "Controller evaluation")
    cfg, system, device = ctx["cfg"], ctx["system"], ctx["device"]
    sindy = _load_sindy(ctx["results_dir"], cfg, device)
    policy = _load_policy(ctx["results_dir"], cfg, device)
    nn_policy = (
        load_nn_policy(
            system,
            find_nn_checkpoint(ctx["results_dir"], cfg),
            cfg,
            device=device,
        )
        if "nn_dpc" in cfg.get("methods", [])
        else None
    )

    run_dir = new_run_dir(ctx["results_dir"] / "eval")
    output = evaluate(
        system,
        sindy,
        policy,
        cfg,
        device,
        out_dir=run_dir,
        nn_policy=nn_policy,
    )
    snapshot_config(run_dir, cfg, ctx["seed"])
    print("\n" + format_metrics_table(output["summary"]))
    print(f"\nSaved evaluation to {run_dir}")


def _print_metrics(label: str, metrics: Dict, run_dir: Path) -> None:
    print(f"{label} metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nSaved to {run_dir}")
