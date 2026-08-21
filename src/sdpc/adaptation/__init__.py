"""Online adaptation: reference tracking, symbolic Jacobian, barriers, and safe adaptation.

Layout mirrors Sec. 3.4 of the paper:
* ``reference``       — unconstrained reference-tracking update (Eq. 12).
* ``jacobian``        — analytic closed-loop sensitivity (Eq. 11) for the reference step.
* ``safety_jacobian`` — analytic forward-sensitivity recursion for the barrier-gradient
  safety step (symbolic-Jacobian generalization of the box/ReLU sensitivity approach).
* ``barrier``         — barrier functions and the predictive barrier loss (Eq. 18).
* ``rollout``         — differentiable finite-horizon closed-loop rollout.
* ``unconstrained``   — reference-tracking runner (Sec. 3.4.1).
* ``safe``            — barrier-augmented safe adaptation runner (Algorithm 2).
* ``psf``             — predictive safety-filter benchmark (Algorithm 3, CasADi).
"""
from .reference import (
    compute_updates_discrete_ref,
    apply_policy_updates,
    adaptive_gamma_from_error,
)
from .jacobian import SymbolicJacobian, build_symbolic_jacobian, compute_updates_symbolic
from .safety_jacobian import SymbolicSafetyJacobian
from .barrier import (
    squared_hinge,
    relaxed_log_barrier,
    barrier_value,
    barrier_derivative,
    rollout_barrier_loss,
    rollout_box_relu_loss,
    rollout_box_barrier_loss,
)
from .rollout import predict_rollout, clamp_action, current_reference, hold_current_reference
from .unconstrained import (
    UnconstrainedAdaptationConfig,
    PreparedUnconstrainedAdaptation,
    prepare_unconstrained_adaptation,
    run_unconstrained_adaptation,
)
from .safe import (
    SafeAdaptationConfig,
    PreparedSafeAdaptation,
    prepare_safe_adaptation,
    run_safe_adaptation,
)
from .psf import PSFConfig, run_psf_adaptation, casadi_available

__all__ = [
    "compute_updates_discrete_ref",
    "apply_policy_updates",
    "adaptive_gamma_from_error",
    "SymbolicJacobian",
    "build_symbolic_jacobian",
    "compute_updates_symbolic",
    "SymbolicSafetyJacobian",
    "squared_hinge",
    "relaxed_log_barrier",
    "barrier_value",
    "barrier_derivative",
    "rollout_barrier_loss",
    "rollout_box_relu_loss",
    "rollout_box_barrier_loss",
    "predict_rollout",
    "clamp_action",
    "current_reference",
    "hold_current_reference",
    "UnconstrainedAdaptationConfig",
    "PreparedUnconstrainedAdaptation",
    "prepare_unconstrained_adaptation",
    "run_unconstrained_adaptation",
    "SafeAdaptationConfig",
    "PreparedSafeAdaptation",
    "prepare_safe_adaptation",
    "run_safe_adaptation",
    "PSFConfig",
    "run_psf_adaptation",
    "casadi_available",
]
