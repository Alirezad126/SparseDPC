import torch

from sdpc.adaptation.rollout import (
    current_reference,
    hold_current_reference,
    predict_rollout,
)
from sdpc.eval.rollout import rollout_closed_loop


def test_predictive_horizon_does_not_see_upcoming_reference_switch():
    r_data = torch.tensor([[[0.2, 0.2], [0.2, 0.2], [0.8, 0.8], [0.8, 0.8]]])

    before_switch = hold_current_reference(r_data, 1, horizon=5)
    at_switch = hold_current_reference(r_data, 2, horizon=5)

    assert before_switch.shape == (1, 5, 2)
    assert torch.all(before_switch == torch.tensor([0.2, 0.2]))
    assert torch.all(at_switch == torch.tensor([0.8, 0.8]))


def test_current_reference_changes_only_at_current_data_index():
    r_data = torch.tensor([[[0.1], [0.1], [0.6], [0.6]]])

    assert torch.equal(current_reference(r_data, 1), torch.tensor([[0.1]]))
    assert torch.equal(current_reference(r_data, 2), torch.tensor([[0.6]]))


def test_policy_inference_receives_only_reference_at_each_execution_index():
    seen = []

    def policy(x, r):
        seen.append(r.clone())
        return torch.zeros(x.shape[0], 1)

    def plant(x, u):
        return x

    r_data = torch.tensor([[[0.1], [0.1], [0.7], [0.7]]])
    predict_rollout(policy, plant, torch.zeros(1, 1), r_data, n_steps=4)

    assert torch.allclose(torch.stack(seen).flatten(), torch.tensor([0.1, 0.1, 0.7, 0.7]))


def test_evaluation_rollout_does_not_preview_reference_schedule():
    seen = []

    def policy(x, r):
        seen.append(r.clone())
        return torch.zeros(x.shape[0], 1)

    data = {
        "xn": torch.zeros(1, 1, 1),
        "r": torch.tensor([[[0.2], [0.2], [0.8], [0.8]]]),
    }
    rollout_closed_loop(policy, lambda x, u: x, data)

    assert torch.allclose(torch.stack(seen).flatten(), torch.tensor([0.2, 0.2, 0.8]))
