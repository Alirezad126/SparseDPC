import torch

from sdpc.registry import make_system


def test_double_integrator_builds_multiple_named_obstacles():
    system = make_system("double_integrator", device=torch.device("cpu"))
    cfg = {
        "obstacles": [
            {"name": "left", "p": 1.0, "b": 2.0, "c": -1.0, "d": 0.0},
            {"name": "right", "p": 1.0, "b": 2.0, "c": 1.0, "d": 0.0},
        ],
        "bands": {"obstacle": 0.2, "box": 0.1},
        "weights": {"Q_con_obs": 3.0, "Q_con_x": 1.0},
    }
    spec = system.safety_specs(cfg)
    obstacles = [con for con in spec.state_constraints if con.meta.get("kind") == "rotated_ellipse"]
    assert [con.name for con in obstacles] == ["left", "right"]
    assert all(con.band == 0.2 for con in obstacles)
    assert all(con.weight == 3.0 for con in obstacles)
    assert obstacles[0].margin(torch.tensor([[-1.0, 0.0]])).item() < 0.0
    assert obstacles[1].margin(torch.tensor([[1.0, 0.0]])).item() < 0.0


def test_double_integrator_rejects_duplicate_obstacle_names():
    system = make_system("double_integrator", device=torch.device("cpu"))
    cfg = {
        "obstacles": [
            {"name": "same", "p": 1.0, "b": 1.0, "c": -1.0, "d": 0.0},
            {"name": "same", "p": 1.0, "b": 1.0, "c": 1.0, "d": 0.0},
        ]
    }
    try:
        system.safety_specs(cfg)
    except ValueError as exc:
        assert "duplicate obstacle name" in str(exc)
    else:
        raise AssertionError("duplicate obstacle names must be rejected")
