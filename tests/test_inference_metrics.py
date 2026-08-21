import torch

from sdpc.eval.metrics import policy_forward_latency


class _Policy:
    def __init__(self):
        self.calls = 0
        self.shapes = []

    def __call__(self, x, r):
        self.calls += 1
        self.shapes.append((tuple(x.shape), tuple(r.shape)))
        return x + r


def test_policy_forward_latency_times_only_requested_forward_calls():
    policy = _Policy()
    x = torch.zeros(1, 12, 2)
    r = torch.ones(1, 12, 2)

    metrics = policy_forward_latency(
        policy, x, r, warmup=3, repeats=4, iters=5, batch_size=1, seed=7
    )

    assert metrics["policy_forward_per_step_s"] >= 0.0
    assert metrics["policy_forward_per_step_ms"] == 1.0e3 * metrics["policy_forward_per_step_s"]
    assert metrics["policy_forward_per_step_us"] == 1.0e6 * metrics["policy_forward_per_step_s"]
    assert policy.calls == 4 * (3 + 5)
    assert set(policy.shapes) == {((1, 2), (1, 2))}
