import torch
from neuromancer.dynamics.ode import ODESystem
from SparseDPC.src.sindy.library import FunctionLibrary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SINDy(ODESystem):
    """
    Sparse Identification of Nonlinear Dynamics
    Reference: https://www.pnas.org/doi/10.1073/pnas.1517384113
    """


    def __init__(
        self,
        library,
        n_out=None,
        main_idx=0,
    ):
        """
        :param library: (FunctionLibrary) the library of candidate functions
        :param threshold: (float) all functions with coefficients lower than this are omitted
        """
        assert isinstance(library, FunctionLibrary), "Must be valid library"

        self.n_out = n_out
        if n_out is None:
            self.n_out = library.n_features

        super().__init__(library.shape[1], self.n_out)


        self.library = library
        init_coef = torch.rand((self.library.shape[0], self.n_out))
        self.coef = torch.nn.Parameter(init_coef, requires_grad=True).to(device)
        self.float()
        self.main_idx = main_idx

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
            return_str += f"dx{self.main_idx}/dt = "
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
