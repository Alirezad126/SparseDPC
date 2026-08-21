import pytest
import torch

from sdpc.training.trainer import SparseTrainer


def test_step_lr_uses_global_epochs_across_new_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    trainer = SparseTrainer.__new__(SparseTrainer)
    trainer.optimizers = torch.optim.AdamW([parameter], lr=5.0e-3)
    trainer.lr_scheduler_enabled = True
    trainer.lr_decay_step = 25
    trainer.lr_decay_gamma = 0.5

    trainer.lr = 5.0e-3
    trainer._validate_lr_schedule()
    assert trainer._set_scheduled_learning_rate(25) == pytest.approx(2.5e-3)

    new_parameter = torch.nn.Parameter(torch.tensor(0.0))
    trainer.optimizers = torch.optim.AdamW([new_parameter], lr=2.5e-3)

    assert trainer._set_scheduled_learning_rate(49) == pytest.approx(2.5e-3)
    assert trainer._set_scheduled_learning_rate(50) == pytest.approx(1.25e-3)


@pytest.mark.parametrize(
    ("step", "gamma", "message"),
    [(0, 0.5, "lr_decay_step"), (25, 0.0, "lr_decay_gamma")],
)
def test_step_lr_rejects_invalid_config(step, gamma, message):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    trainer = SparseTrainer.__new__(SparseTrainer)
    trainer.optimizers = torch.optim.AdamW([parameter], lr=5.0e-3)
    trainer.lr_scheduler_enabled = True
    trainer.lr = 5.0e-3
    trainer.lr_decay_step = step
    trainer.lr_decay_gamma = gamma

    with pytest.raises(ValueError, match=message):
        trainer._validate_lr_schedule()


def test_proximal_l1_uses_task_gradient_then_creates_exact_zeros():
    coefficients = torch.nn.Parameter(torch.tensor([[0.0005], [0.0030]]))
    trainer = SparseTrainer.__new__(SparseTrainer)
    trainer.sindy = type("SparseModel", (), {})()
    trainer.sindy.Xi = torch.nn.ParameterList([coefficients])
    trainer.proximal_l1 = 0.1
    trainer.optimizers = torch.optim.SGD([coefficients], lr=0.01)

    task_loss = ((coefficients - 1.0) ** 2).sum()
    reported_loss = task_loss + trainer.proximal_l1 * coefficients.abs().sum()
    trainer._training_objective(reported_loss).backward()

    assert torch.allclose(
        coefficients.grad,
        2.0 * (coefficients.detach() - 1.0),
    )

    coefficients.grad.zero_()
    trainer._apply_proximal_l1()

    assert coefficients[0].item() == 0.0
    assert torch.allclose(coefficients[1], torch.tensor([0.0020]))


def test_negative_proximal_l1_is_rejected():
    with pytest.raises(ValueError, match="proximal_l1"):
        SparseTrainer(
            problem=torch.nn.Linear(1, 1),
            sindy=torch.nn.Linear(1, 1),
            lr=0.01,
            optimizers=torch.optim.SGD(
                torch.nn.Linear(1, 1).parameters(), lr=0.01
            ),
            proximal_l1=-0.1,
        )


def test_total_control_action_l1_is_mean_trajectory_sum():
    u = torch.tensor(
        [
            [[1.0, -2.0], [3.0, -4.0]],
            [[0.5, -0.5], [1.0, -1.0]],
        ]
    )

    value = SparseTrainer._total_control_action_l1(u)

    assert value.item() == pytest.approx((10.0 + 3.0) / 2.0)
