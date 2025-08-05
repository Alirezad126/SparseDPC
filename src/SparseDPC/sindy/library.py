import itertools
import torch
from typing import List, Callable, Optional, Tuple, Union


class FunctionLibrary:
    def __init__(
        self,
        functions: List[Callable],
        n_features: int,
        n_control: int = 0,
        function_names: Optional[List[str]] = None
    ):
        """
        Generic symbolic function library.

        Args:
            functions: List of callables(X, u) -> tensor
            n_features: Number of state variables
            n_control: Number of control inputs
            function_names: Optional list of names for functions
        """
        assert isinstance(functions, list), "Functions must be provided as a list."
        if function_names is not None:
            assert len(functions) == len(function_names), "Mismatch in function count and names."

        self.library = functions
        self.n_features = n_features
        self.n_control = n_control
        self.shape = (len(functions), n_features + n_control)
        self.function_names = function_names or self.__str__()

    def evaluate(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate each function in the library on inputs x and optionally u.

        Returns:
            A tensor of shape (batch, num_functions)
        """
        device = x.device
        output = torch.zeros((x.shape[0], self.shape[0]), device=device)

        for i in range(self.shape[0]):
            try:
                output[:, i] = self.library[i](x, u).to(device)
            except Exception:
                output[:, i] = self.library[i](x).to(device)

        return output

    def __str__(self) -> str:
        return ", ".join(self.function_names or [f"f{i}" for i in range(self.shape[0])])


class PolynomialLibrary(FunctionLibrary):
    def __init__(
        self,
        n_features: int,
        n_control: int = 0,
        max_degree: int = 2,
        interaction: bool = True
    ):
        """
        Polynomial basis library for SINDy.

        Args:
            n_features: Number of state variables
            n_control: Number of control inputs
            max_degree: Maximum polynomial degree
            interaction: Include interaction terms
        """
        self.max_degree = max_degree
        self.interaction = interaction
        lib, function_names = self.__create_library(n_features, n_control)
        super().__init__(lib, n_features, n_control, function_names)

    def __create_library(
        self, n_features: int, n_control: int
    ) -> Tuple[List[Callable], List[str]]:
        funs_list = [
            (lambda i=i: (lambda X, *_: X[:, i]))() for i in range(n_features)
        ] + [
            (lambda i=i: (lambda X, *args: args[0][:, i]))() for i in range(n_control)
        ]

        vars_list = [f"x{i}" for i in range(n_features)] + [f"u{i}" for i in range(n_control)]

        all_funs: List[Tuple[Callable]] = [(lambda X, *args: torch.ones(X.shape[0], device=X.device),)]
        all_names = []

        for i in range(1, self.max_degree + 1):
            if self.interaction:
                combos = list(itertools.combinations_with_replacement(funs_list, i))
                names = list(itertools.combinations_with_replacement(vars_list, i))
            else:
                combos, names = [], []
                for j in range(len(funs_list)):
                    new_combo = list(itertools.combinations_with_replacement([funs_list[j]], i))
                    new_names = list(itertools.combinations_with_replacement([vars_list[j]], i))
                    combos += new_combo
                    names += new_names

            all_funs.extend(combos)
            all_names.extend(names)

        return all_funs, self.__convert(all_names)

    def __convert(self, names: List[Tuple[str]]) -> List[str]:
        """
        Converts raw tuple terms into readable expressions like x^2*u.
        """
        return_names = ["1"]
        for group in names:
            count = {}
            for term in group:
                count[term] = count.get(term, 0) + 1
            term_str = "*".join(f"{k}" if v == 1 else f"{k}^{v}" for k, v in count.items())
            return_names.append(term_str)
        return return_names

    def evaluate(self, X: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate polynomial basis functions.
        """
        output = torch.ones((X.shape[0], self.shape[0]), device=X.device)
        for i, combo in enumerate(self.library):
            for func in combo:
                output[:, i] *= func(X, u)
        return output


class FourierLibrary(FunctionLibrary):
    def __init__(
        self,
        n_features: int,
        n_control: int = 0,
        max_freq: int = 2,
        include_sin: bool = True,
        include_cos: bool = True
    ):
        """
        Fourier basis library.

        Args:
            n_features: Number of state variables
            n_control: Number of control inputs
            max_freq: Maximum frequency multiplier
            include_sin: Include sine terms
            include_cos: Include cosine terms
        """
        self.max_freq = max_freq
        self.include_sin = include_sin
        self.include_cos = include_cos
        lib, names = self.__create_library(n_features, n_control)
        super().__init__(lib, n_features, n_control, names)

    def __create_library(
        self, n_features: int, n_control: int
    ) -> Tuple[List[Callable], List[str]]:
        functions: List[Callable] = []
        names: List[str] = []

        def sinusoid(fun, base, label):
            return [
                (lambda z=z, k=k: (lambda X, *args: fun(k * (X[:, z] if base == 'x' else args[0][:, z]))))()
                for z in range(n_features if base == 'x' else n_control)
                for k in range(1, self.max_freq + 1)
            ], [
                f"{label}({k}*{base}{z})"
                for z in range(n_features if base == 'x' else n_control)
                for k in range(1, self.max_freq + 1)
            ]

        if self.include_sin:
            fns, nms = sinusoid(torch.sin, 'x', 'sin')
            functions += fns; names += nms
            fns, nms = sinusoid(torch.sin, 'u', 'sin')
            functions += fns; names += nms

        if self.include_cos:
            fns, nms = sinusoid(torch.cos, 'x', 'cos')
            functions += fns; names += nms
            fns, nms = sinusoid(torch.cos, 'u', 'cos')
            functions += fns; names += nms

        return functions, names


class CombinedLibrary(FunctionLibrary):
    def __init__(self, libraries: List[FunctionLibrary]):
        """
        Combine multiple FunctionLibrary instances.

        Args:
            libraries: List of libraries to merge
        """
        assert all(isinstance(lib, FunctionLibrary) for lib in libraries)
        base_features = libraries[0].n_features
        base_controls = libraries[0].n_control

        for lib in libraries:
            assert lib.n_features == base_features
            assert lib.n_control == base_controls

        self.n_features = base_features
        self.n_control = base_controls
        self.libraries = libraries
        self.function_names = sum([lib.function_names for lib in libraries], [])
        self.shape = (sum([lib.shape[0] for lib in libraries]), base_features + base_controls)

    def evaluate(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Concatenate outputs from all sub-libraries.
        """
        outputs = [lib.evaluate(x, u).T for lib in self.libraries]
        return torch.cat(outputs).T
