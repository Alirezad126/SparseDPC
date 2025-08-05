import torch
from neuromancer.dynamics.ode import ODESystem
from SparseDPC.sindy.library import FunctionLibrary


class SINDy(ODESystem):
    """
    Sparse Identification of Nonlinear Dynamics or Control Policy (SINDy)

    This model can represent either:
    - Dynamics:    dx/dt = Θ(x,u)·coef
    - Policy:      u    = Θ(x,r)·coef
    """

    def __init__(
        self,
        library: FunctionLibrary,
        n_out: int | None = None,
        main_idx: int = 0,
        policy_name: str | None = None,
        seed: int | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        assert isinstance(library, FunctionLibrary), "`library` must be FunctionLibrary"

        self.library = library
        self.main_idx = main_idx
        self.policy_name = policy_name
        self.n_out = n_out or library.n_features

        super().__init__(library.shape[1], self.n_out)

        gen = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        init_coef = 0.5 * (2 * torch.rand(
            (library.shape[0], self.n_out),
            generator=gen,
            device=device
        ) - 1)

        self.coef = torch.nn.Parameter(init_coef, requires_grad=True)
        self.float()

    def ode_equations(
        self,
        x: torch.Tensor,
        u: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Evaluate model output as Θ(x,u)·coef.

        Interpreted as dx/dt or u depending on context.
        """
        x = x.to(self.coef.device)
        if u is not None:
            u = u.to(self.coef.device)

        lib_eval = self.library.evaluate(x, u)
        return lib_eval @ self.coef

    def set_parameters(self, new_params: torch.nn.Parameter):
        """
        Replace trainable coefficients.
        """
        assert isinstance(new_params, torch.nn.Parameter), "Expected nn.Parameter"
        assert new_params.requires_grad, "Parameter must require gradients"
        assert new_params.shape == self.coef.shape, "Shape mismatch"

        self.coef = new_params

    def __str__(self) -> str:
        """
        Pretty-print symbolic representation of model.

        Format:
        - If policy_name: policy_name = ...
        - Else: dx/dt = ...
        """
        names = self.library.function_names or [f"f{i}" for i in range(self.library.shape[0])]
        out = ""

        for i in range(self.nx):
            lhs = f"{self.policy_name}" if self.policy_name else f"dx{self.main_idx}/dt"
            terms = [f"{self.coef[j, i]:.3f}*{names[j]}" for j in range(len(names))]
            out += f"{lhs} = {' + '.join(terms)}\n"

        return out
