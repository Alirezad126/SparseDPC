import torch
from neuromancer.dynamics.ode import ODESystem
from SparseDPC.sindy.library import FunctionLibrary

class SINDy(ODESystem):
    """
    Sparse Identification of Non-Linear Dynamics
    """

    def __init__(
        self,
        library: FunctionLibrary,
        n_out: int | None = None,
        main_idx: int     = 0,
        policy_name: str | None = None,
        seed: int | None  = None,
        device: torch.device = torch.device("cpu"),
    ):
        assert isinstance(library, FunctionLibrary), "`library` must be FunctionLibrary"

        self.n_out = n_out or library.n_features
        super().__init__(library.shape[1], self.n_out)

        self.library     = library
        self.main_idx    = main_idx
        self.policy_name = policy_name

        # ---------------------------------------------------------------
        # reproducible initialisation
        # ---------------------------------------------------------------
        if seed is None:
            gen = None                       # use global RNG
        else:
            gen = torch.Generator(device=device).manual_seed(seed)

        init_coef = 0.5 * (2 * torch.rand(
            (self.library.shape[0], self.n_out), generator=gen, device=device
        ) - 1)

        self.coef = torch.nn.Parameter(init_coef, requires_grad=True)
        self.float()
        self.main_idx = main_idx
        self.policy_name = policy_name

    def ode_equations(self, x, u=None):
        """
        Compute the time derivative of state variables using SINDy equations.

        :param x: (torch.tensor) Current state values
        :param u: (torch.tensor) Control input (optional)
        """
        # Move tensors to the same device
        device = self.coef.device  # Get device of trainable parameters
        x = x.to(device)
        if u is not None:
            u = u.to(device)

        if u is None:
            lib_eval = self.library.evaluate(x)
        else:
            lib_eval = self.library.evaluate(x, u)

        # Compute dx/dt
        output = torch.matmul(lib_eval, self.coef)
        return output


    def __str__(self):
        """
        return: (str) a list of the linear combinations of candidate functions for each state variable
        """
        f_names = self.library.__str__()
        f_names = f_names.split(", ")
        return_str = ""

        for i in range(self.nx):
            if self.policy_name is None:
                return_str += f"dx{self.main_idx}/dt = "
            else:
                return_str += f"{self.policy_name} = "
            for j in range(len(f_names)):
                coef = self.coef[j, i]
                func = f_names[j]
                return_str += f"{coef:.3f}*{func} + "
            return_str = return_str[:-2]
            return_str += "\n"

        return return_str

    def set_parameters(self, new_params):
        assert self.coef.shape == new_params.shape, "New parameters must have same shape"
        assert isinstance(new_params, torch.nn.Parameter), "Must be torch.nn.Parameter"
        assert new_params.requires_grad, "Must require gradients"

        self.coef = new_params
