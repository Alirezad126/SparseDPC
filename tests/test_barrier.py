import torch

from sdpc.adaptation.barrier import barrier_derivative, relaxed_log_barrier


def test_relaxed_log_rejects_nonpositive_band():
    h = torch.tensor([-1.0, 1.0])
    for fn in (
        lambda: relaxed_log_barrier(h, delta=0.0, band=0.0),
        lambda: barrier_derivative(h, delta=0.0, band=0.0, kind="relaxed_log"),
    ):
        try:
            fn()
        except ValueError as exc:
            assert "band > 0" in str(exc)
        else:
            raise AssertionError("zero-band relaxed_log must be rejected")


def test_relaxed_log_positive_band_is_finite_and_inactive_above_band():
    h = torch.tensor([-1.0, 0.5, 2.0])
    value = relaxed_log_barrier(h, delta=0.0, band=1.0)
    derivative = barrier_derivative(h, delta=0.0, band=1.0, kind="relaxed_log")
    assert torch.isfinite(value).all()
    assert torch.isfinite(derivative).all()
    assert value[-1].item() == 0.0
    assert derivative[-1].item() == 0.0
