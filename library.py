import numpy as np
import itertools
import torch


class FunctionLibrary:

    def __init__(
        self,
        functions,
        n_features,
        n_control=0,
        function_names=None
    ):
        """
        :param functions: (list) a one-dimensional list of callables functions.
                                 All callables must be able to operate on the
                                 columns of a two-dimensional torch.Tensor of
                                 floating point numbers.
        :param n_features: (int) number of columns of data matrix
        :param n_control: (int) number of control inputs
        :param function_names: (list) name for each function, must either be None
                                      or a list of the same length as the functions
        """
        if function_names is not None:
            assert len(functions) == len(function_names), "Must have one name for each function"

        assert isinstance(functions, list), "Functions must be passed as list"

        self.library = functions

        self.shape = (len(self.library), n_features + n_control)
        self.n_features = n_features
        self.n_control = n_control
        self.function_names = function_names

        if function_names is None:
            self.function_names = self.__str__()

    def evaluate(self, x, u=None):
        """
        :param x: (torch.Tensor) an array of values to evaluate the functions at
        :param u: (torch.Tensor) a two-dimensional matrix of control inputs
        :return: (torch.Tensor, shape=[self.shape + x.shape + u.shape])
        """
        device = x.device
        output = torch.zeros(size=(x.shape[0], self.shape[0])).to(device)
        for i in range(self.shape[0]):
            try:
                output[:, i] = self.library[i](x, u).to(device)
            except:
                output[:, i] = self.library[i](x).to(device)

        return output

    def __str__(self):
        """
        :return: (str) all of the names provided to the library, or generic function names
        """
        out_str = ""
        if self.function_names is not None:
            for name in self.function_names:
                out_str += f"{name}, "

        else:

            for i in range(self.shape[0]):
                out_str += f"f{i}, "
        return out_str[:-2]


class PolynomialLibrary(FunctionLibrary):
    def __init__(
        self,
        n_features,
        n_control=0,
        max_degree=2,
        interaction=True
        ):
        """
        :param n_features: (int) the number of features in the dataset
        :param n_control: (int) the number of control inputs in the dataset
        :param max_degree: (int) the maximum total degree of any polynomial
        :param interaction: (bool) whether or not to include interaction terms such as x_0^2*x_1
        """
        self.max_degree = max_degree
        self.interaction = interaction
        lib, function_names = self.__create_library(n_features, n_control)
        super().__init__(lib, n_features, n_control, function_names)

    def __create_library(self, n_features, n_control):
        """
        :param n_features: (int) the number of features in the dataset
        :param n_control: (int) the number of control inputs in the dataset
        :return: (list) a row vector of all of the polynomial functions
        :return: (list) a list of all of the names of the functions
        """
        funs_list = []
        all_combos = []
        funs_list = [(lambda y: lambda X, *args: X[:, y])(i) for i in range(n_features)]
        funs_list += [(lambda y: lambda X, *args: args[0][:, y])(i) for i in range(n_control)]

        vars_list = [f"x{i}" for i in range(n_features)]
        vars_list += [f"u{i}" for i in range(n_control)]

        all_combos = [(lambda X, *args: torch.ones(size=(X.shape[0],)),)]
        all_names = []

        for i in range(1, self.max_degree+1):
            if self.interaction:
                combos = list(itertools.combinations_with_replacement(funs_list, i))
                names = list(itertools.combinations_with_replacement(vars_list, i))

            else:
                combos = []
                names = []
                for j in range(n_features+n_control):
                    new_funs = list(itertools.combinations_with_replacement([funs_list[j]], i))
                    new_names = list(itertools.combinations_with_replacement([vars_list[j]], i))
                    combos += new_funs
                    names += new_names

            for j in range(len(combos)):
                all_combos.append(combos[j])

            for j in range(len(names)):
                all_names.append(names[j])

        library = all_combos
        names = self.__convert(all_names)
        return library, names

    def __convert(self, names):
        """
        :param names: (list) a list of tuples of all of the names of terms combined
        :return: (list) the names converted into a more intrepreted form (i.e. x*x*x -> x^3)
        """
        return_names = ["1"]
        for func in names:
            name = ""
            for term in func:
                if term not in name:
                    name += f"{term}*"
                else:
                    if f"{term}^" in name:
                        degree = int(name[name.find(f"^") + 1])
                        name = name.replace(f"{term}^{degree}", f"{term}^{degree+1}")
                    else:
                        name = name.replace(term, f"{term}^2")
            return_names.append(name[:-1])
        return return_names

    def evaluate(self, X, u=None):
        """
        :param X: (torch.Tensor) the two-dimensional dataset to put through the library
        :return: (torch.Tensor, [# of rows of X, number of functions]) the library evaluated at X
        """
        device = X.device
        output = torch.ones((X.shape[0], self.shape[0])).to(device)
        for i in range(self.shape[0]):
            for func in self.library[i]:
                output[:, i] *= func(X, u).to(device)

        return output


class FourierLibrary(FunctionLibrary):
    def __init__(
        self,
        n_features,
        n_control=0,
        max_freq=2,
        include_sin=True,
        include_cos=True
    ):
        """
        :param n_features: (int) the number of features of the dataset
        :param n_control: (int) the number of control inputs in the dataset
        :param max_freq: (int) the max multiplier for the frequency
        :param include_sin: (bool) include the sine function
        :param include_cos: (bool) include the cosine function
        """
        self.max_freq = max_freq
        self.include_sin = include_sin
        self.include_cos = include_cos
        lib, function_names = self.__create_library(n_features, n_control)
        super().__init__(lib, n_features, n_control, function_names)

    def __create_library(self, n_features, n_control):
        """
        :param n_features: (int) the number of input features of the dataset
        :param n_control: (int) the number of control inputs in the dataset
        :return: (list) an array of functions of sines and cosines
        :return: (list) a list of the names of each function (i.e. sin(2*x0))
        """
        functions = []
        names = []

        if self.include_sin:
            functions += [(lambda z: [(lambda y: lambda X, *args: torch.sin(y * X[:,z]))(i) \
                    for i in range(1,self.max_freq+1)]) (j) for j in \
                    range(n_features)]

            functions += [(lambda z: [(lambda y: lambda X, *args: torch.sin(y * args[0][:,z]))(i) \
                    for i in range(1,self.max_freq+1)]) (j) for j in \
                    range(n_control)]

            names += [[f"sin({j}*x{i})"  for j in \
                range(1, self.max_freq+1)] for i in range(n_features)]

            names += [[f"sin({j}*u{i})"  for j in \
                    range(1, self.max_freq+1)] for i in range(n_control)]

        if self.include_cos:
            functions += [(lambda z: [(lambda y: lambda X, *args: torch.cos(y * X[:,z]))(i) \
                    for i in range(1,self.max_freq+1)]) (j) for j in \
                    range(n_features)]

            functions += [(lambda z: [(lambda y: lambda X, *args: torch.cos(y * args[0][:,z]))(i) \
                    for i in range(1,self.max_freq+1)]) (j) for j in \
                    range(n_control)]

            names += [[f"cos({j}*x{i})"  for j in \
                range(1, self.max_freq+1)] for i in range(n_features)]

            names += [[f"cos({j}*u{i})"  for j in \
                    range(1, self.max_freq+1)] for i in range(n_control)]

        return sum(functions, []), sum(names, [])


class CombinedLibrary(FunctionLibrary):
    def __init__(
        self,
        libraries
        ):
        """
        :param libraries: (list) a list of libraries
        """
        self.n_features = libraries[0].n_features
        self.n_control = libraries[0].n_control
        self.libraries = libraries
        self.function_names = []
        length = 0

        for library in libraries:
            assert library.n_features == self.n_features, "All libraries must have same number of features"
            assert library.n_control == self.n_control, "All libraries must have same number of control inputs"
            length += library.shape[0]
            self.function_names += library.function_names
        self.shape = (length, self.n_features + self.n_control)

    def evaluate(self, x, u=None):
        """
        :param x: (torch.Tensor) the input data to evaluate the library at
        :param u: (torch.Tensor) a two-dimensional matrix of control inputs
        :return: (torch.Tensor, shape=[self.shape + x.shape + u.shape])
        """
        outputs = []

        for library in self.libraries:
            outputs.append(library.evaluate(x, u).T)

        return torch.cat(outputs).T
