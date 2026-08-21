import torch
import numpy as np

from sdpc.baselines.mpc import _held_reference
from sdpc.eval import sample_scenario
from sdpc.eval.twotank import make_twotank_scenarios, violation_summary
from sdpc.registry import make_system


def test_mpc_holds_only_the_current_reference_across_its_horizon():
    reference = np.asarray([[0.2, 0.2], [0.8, 0.8], [0.4, 0.4]])

    before_switch = _held_reference(reference, step=0, horizon=4)
    after_switch = _held_reference(reference, step=1, horizon=4)

    assert before_switch.shape == (5, 2)
    assert np.all(before_switch == reference[0])
    assert np.all(after_switch == reference[1])


def test_nominal_twotank_scenarios_have_equal_channels_and_four_segments():
    system = make_system("twotank", device=torch.device("cpu"))
    cfg = {
        "seed": 4,
        "n_trajectories": 5,
        "nsteps": 12,
        "n_references": 4,
        "x0_min": 0.1,
        "x0_max": 0.9,
        "reference": {"mode": "random", "min": 0.2, "max": 0.8},
    }
    data = make_twotank_scenarios(system, cfg)

    assert data["xn"].shape == (5, 1, 2)
    assert data["r"].shape == (5, 13, 2)
    assert data["reference_levels"].shape == (5, 4)
    assert torch.all(data["r"][:, :, 0] == data["r"][:, :, 1])
    assert not torch.all(data["reference_levels"][0] == data["reference_levels"][1])
    assert torch.all(data["r"][:, :3, 0] == data["reference_levels"][:, [0]])
    assert torch.all(data["r"][:, 3:6, 0] == data["reference_levels"][:, [1]])
    assert torch.all(data["r"][:, -1, 0] == data["reference_levels"][:, -1])


def test_safety_scenarios_include_both_near_boundary_references():
    system = make_system("twotank", device=torch.device("cpu"))
    cfg = {
        "seed": 5,
        "n_trajectories": 6,
        "nsteps": 20,
        "n_references": 4,
        "reference": {
            "mode": "safety_edges",
            "edge_levels": [0.05, 0.95],
            "interior_min": 0.2,
            "interior_max": 0.8,
            "shuffle": True,
        },
    }
    levels = make_twotank_scenarios(system, cfg)["reference_levels"]

    assert torch.all((levels == 0.05).any(dim=1))
    assert torch.all((levels == 0.95).any(dim=1))


def test_safety_scenarios_scale_fractions_to_configured_bounds():
    system = make_system("twotank", device=torch.device("cpu"))
    cfg = {
        "seed": 5,
        "n_trajectories": 6,
        "nsteps": 20,
        "n_references": 4,
        "xmin": 0.1,
        "xmax": 0.9,
        "initial_fraction_range": [0.2, 0.7],
        "bands": {"box": 0.05},
        "psf": {"buffers": {"box": 0.05}},
        "reference": {
            "mode": "safety_edges",
            "edge_fractions": [0.1, 0.9],
            "interior_fraction_range": [0.25, 0.75],
            "shuffle": True,
        },
    }
    data = make_twotank_scenarios(system, cfg)
    levels = data["reference_levels"]

    assert torch.allclose(data["state_bounds"], torch.tensor([0.1, 0.9]))
    assert torch.all((torch.isclose(levels, torch.tensor(0.18))).any(dim=1))
    assert torch.all((torch.isclose(levels, torch.tensor(0.82))).any(dim=1))
    assert float(data["xn"].min()) >= 0.26
    assert float(data["xn"].max()) <= 0.66


def test_safety_scenarios_reject_overlapping_clearance():
    system = make_system("twotank", device=torch.device("cpu"))
    cfg = {
        "n_trajectories": 1,
        "nsteps": 4,
        "n_references": 1,
        "bands": {"box": 0.5},
        "reference": {"mode": "random", "fraction_range": [0.4, 0.6]},
    }
    try:
        make_twotank_scenarios(system, cfg)
    except ValueError as exc:
        assert "overlaps" in str(exc)
    else:
        raise AssertionError("overlapping box clearance was accepted")


def test_safety_scenarios_reject_reference_inside_clearance():
    system = make_system("twotank", device=torch.device("cpu"))
    cfg = {
        "n_trajectories": 1,
        "nsteps": 4,
        "n_references": 2,
        "bands": {"box": 0.1},
        "reference": {
            "mode": "safety_edges",
            "edge_fractions": [0.05, 0.95],
            "interior_fraction_range": [0.2, 0.8],
        },
    }
    try:
        make_twotank_scenarios(system, cfg)
    except ValueError as exc:
        assert "conflict" in str(exc)
    else:
        raise AssertionError("reference inside the box clearance was accepted")


def test_safe_adaptation_scenario_uses_relative_bounds_and_references():
    system = make_system("twotank", device=torch.device("cpu"))
    cfg = {
        "seed": 3,
        "nsteps": 20,
        "xmin": 0.2,
        "xmax": 0.8,
        "initial_fraction_range": [0.2, 0.7],
        "bands": {"box": 0.05},
        "psf": {"buffers": {"box": 0.05}},
        "reference": {
            "kind": "equal",
            "relative_to_bounds": True,
            "levels": [[0.15, 0.15], [0.9, 0.9]],
        },
    }
    data = sample_scenario(system, cfg, cfg["seed"], torch.device("cpu"))

    assert torch.allclose(data["state_bounds"], torch.tensor([0.2, 0.8]))
    assert 0.32 <= float(data["xn"].min()) <= float(data["xn"].max()) <= 0.62
    assert torch.allclose(data["r"][0, 0], torch.tensor([0.29, 0.29]))
    assert torch.allclose(data["r"][0, -1], torch.tensor([0.74, 0.74]))


def test_violation_summary_counts_bad_trajectories_separately():
    summary = violation_summary({
        "method": [
            {"num_violations": 0, "num_state_violations": 0},
            {"num_violations": 3, "num_state_violations": 2,
             "num_control_rate_violations": 1},
            {"num_violations": 1, "num_state_violations": 1,
             "num_control_rate_violations": 0},
        ]
    })["method"]

    assert summary["trajectories_with_violations"] == 2
    assert summary["trajectory_violation_fraction"] == 2 / 3
    assert summary["total_violations"] == 4
    assert summary["max_violations_per_trajectory"] == 3
