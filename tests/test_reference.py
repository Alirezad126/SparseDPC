import torch

from sdpc.data.reference import build_reference, make_equal_piecewise_reference


def test_equal_reference_divides_control_steps_and_keeps_initial_sample():
    ref = make_equal_piecewise_reference(
        2, 10, [[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]], device=torch.device("cpu")
    )

    assert ref.shape == (1, 11, 2)
    assert torch.allclose(ref[0, :5], torch.tensor([0.2, 0.3]).expand(5, 2))
    assert torch.allclose(ref[0, 5:8], torch.tensor([0.5, 0.6]).expand(3, 2))
    assert torch.allclose(ref[0, 8:], torch.tensor([0.8, 0.9]).expand(3, 2))


def test_equal_reference_config_supports_any_feasible_number_of_levels():
    cfg = {"kind": "equal", "levels": [[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]]}
    ref = build_reference(2, 1500, cfg)

    assert ref.shape == (1, 1501, 2)
    assert torch.all(ref[0, :376] == torch.tensor([0.2, 0.2]))
    assert torch.all(ref[0, 376:751] == torch.tensor([0.4, 0.4]))
    assert torch.all(ref[0, 751:1126] == torch.tensor([0.6, 0.6]))
    assert torch.all(ref[0, 1126:] == torch.tensor([0.8, 0.8]))


def test_equal_reference_rejects_empty_levels():
    try:
        make_equal_piecewise_reference(2, 10, [])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty equal-reference schedules must be rejected")


def test_relative_reference_scales_to_bounds_and_respects_clearance():
    cfg = {
        "kind": "equal",
        "relative_to_bounds": True,
        "levels": [[0.15, 0.15], [0.9, 0.9]],
    }
    ref = build_reference(2, 10, cfg, bounds=(0.2, 0.8), clearance=0.05)

    assert torch.allclose(ref[0, 0], torch.tensor([0.29, 0.29]))
    assert torch.allclose(ref[0, -1], torch.tensor([0.74, 0.74]))


def test_relative_reference_rejects_unreachable_level():
    cfg = {
        "kind": "constant",
        "relative_to_bounds": True,
        "level": [0.95, 0.95],
    }
    try:
        build_reference(2, 10, cfg, bounds=(0.2, 0.8), clearance=0.05)
    except ValueError as exc:
        assert "conflict" in str(exc)
    else:
        raise AssertionError("reference inside the boundary clearance was accepted")


def test_relative_reference_rejects_overlapping_clearance():
    cfg = {
        "kind": "constant",
        "relative_to_bounds": True,
        "level": [0.5, 0.5],
    }
    try:
        build_reference(2, 10, cfg, bounds=(0.2, 0.8), clearance=0.3)
    except ValueError as exc:
        assert "overlaps" in str(exc)
    else:
        raise AssertionError("overlapping boundary clearance was accepted")
