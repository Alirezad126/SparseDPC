import torch

from sdpc.eval import CollapseCriteria, diagnose_tracking_collapse


def test_monotone_trajectory_reaches_without_collapse():
    x = torch.linspace(0.0, 1.0, 21).view(1, -1, 1)
    r = torch.ones_like(x)
    criteria = CollapseCriteria(reach_tolerance=0.1, reach_hold_steps=2)
    result = diagnose_tracking_collapse(x, r, criteria)
    assert result["reached_reference"]
    assert not result["back_and_forth"]
    assert not result["collapsed"]


def test_oscillatory_nonconvergent_trajectory_is_collapse():
    values = [0.0] + [0.25, 0.55] * 12
    x = torch.tensor(values).view(1, -1, 1)
    r = torch.ones_like(x)
    criteria = CollapseCriteria(
        reach_tolerance=0.1,
        reach_hold_steps=3,
        analysis_start_fraction=0.0,
        min_direction_reversals=4,
        min_backtrack_ratio=0.1,
    )
    result = diagnose_tracking_collapse(x, r, criteria)
    assert result["not_reached"]
    assert result["back_and_forth"]
    assert result["collapsed"]


def test_stalled_nonconvergent_trajectory_is_not_oscillatory_collapse():
    x = torch.full((1, 20, 1), 0.3)
    r = torch.ones_like(x)
    result = diagnose_tracking_collapse(x, r)
    assert result["not_reached"]
    assert not result["back_and_forth"]
    assert not result["collapsed"]


def test_trajectory_that_leaves_reference_is_not_settled():
    x = torch.tensor([0.0, 0.8, 0.95, 1.0, 0.7, 0.5]).view(1, -1, 1)
    r = torch.ones_like(x)
    criteria = CollapseCriteria(reach_tolerance=0.1, reach_hold_steps=2)
    result = diagnose_tracking_collapse(x, r, criteria)
    assert result["not_reached"]
    assert not result["reached_reference"]
