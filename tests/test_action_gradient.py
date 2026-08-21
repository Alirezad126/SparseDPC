import pytest
import torch

from sdpc.actions import bounded_action
from sdpc.adaptation.rollout import clamp_action


def test_straight_through_clamp_keeps_forward_value_and_uses_raw_gradient():
    raw = torch.tensor([[-0.25, 1.25]], requires_grad=True)
    applied = clamp_action(
        raw, 0.0, 1.0, gradient_mode="straight_through"
    )

    assert torch.equal(applied.detach(), torch.tensor([[0.0, 1.0]]))
    applied.sum().backward()
    assert torch.equal(raw.grad, torch.ones_like(raw))


def test_exact_clamp_retains_zero_gradient_outside_bounds():
    raw = torch.tensor([[-0.25, 1.25]], requires_grad=True)
    applied = clamp_action(raw, 0.0, 1.0, gradient_mode="exact")

    applied.sum().backward()
    assert torch.equal(raw.grad, torch.zeros_like(raw))


def test_unknown_clamp_gradient_mode_is_rejected():
    with pytest.raises(ValueError, match="gradient_mode"):
        clamp_action(torch.zeros(1, 1), 0.0, 1.0, gradient_mode="unknown")


def test_action_scale_is_applied_before_clamp_and_preserved_in_ste_gradient():
    raw = torch.tensor([[0.4, 0.8]], requires_grad=True)
    applied = bounded_action(
        raw, 0.0, 1.0, action_scale=2.0, gradient_mode="straight_through"
    )

    assert torch.equal(applied.detach(), torch.tensor([[0.8, 1.0]]))
    applied.sum().backward()
    assert torch.equal(raw.grad, torch.full_like(raw, 2.0))


def test_leaky_straight_through_uses_full_gradient_near_bounds_and_leak_far_away():
    raw = torch.tensor([[-0.2, -0.05, 0.5, 1.05, 1.2]], requires_grad=True)
    applied = bounded_action(
        raw,
        0.0,
        1.0,
        gradient_mode="leaky_straight_through",
        gradient_band=0.1,
        gradient_leak=0.05,
    )

    assert torch.equal(
        applied.detach(), torch.tensor([[0.0, 0.0, 0.5, 1.0, 1.0]])
    )
    applied.sum().backward()
    assert torch.allclose(
        raw.grad, torch.tensor([[0.05, 1.0, 1.0, 1.0, 0.05]])
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gradient_band": -0.1}, "gradient_band"),
        ({"gradient_leak": 1.1}, "gradient_leak"),
    ],
)
def test_leaky_straight_through_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bounded_action(
            torch.zeros(1, 1),
            0.0,
            1.0,
            gradient_mode="leaky_straight_through",
            **kwargs,
        )
