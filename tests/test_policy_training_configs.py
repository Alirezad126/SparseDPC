from pathlib import Path

from sdpc.config import load_config


ROOT = Path(__file__).resolve().parents[1]
POLICY_CONFIGS = [
    ROOT / "Relative_Degree_One" / "DoubleIntegrator" / "configs" / "policy.yaml",
    ROOT / "Relative_Degree_One" / "TwoTank" / "configs" / "policy.yaml",
    ROOT / "Relative_Degree_One" / "VanDerPol" / "configs" / "policy.yaml",
    ROOT / "Relative_Degree_Two" / "VanDerPol" / "configs" / "policy.yaml",
]
COMMON_TRAINING_KEYS = {
    "action_gradient_mode",
    "action_gradient_band",
    "action_gradient_leak",
    "grad_clip",
    "proximal_l1",
    "lr_decay_enabled",
    "lr_decay_gamma",
    "lr_decay_step",
    "logger_metrics",
    "eval_seed",
    "eval_margin",
}


def test_sparse_policy_configs_share_training_control_schema():
    for path in POLICY_CONFIGS:
        cfg = load_config(path)
        assert not COMMON_TRAINING_KEYS.difference(cfg), path
        assert cfg["action_gradient_mode"] == "leaky_straight_through"
        assert isinstance(cfg["proximal_l1"], bool)
        assert cfg["lr_decay_enabled"] is True
