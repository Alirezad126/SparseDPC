import torch, sympy as sp
from typing import Sequence, Callable, List, Tuple

def _normalize(text: str) -> str:
    for old, new in (("x0","x_0"),("x1","x_1"),
                     ("u0","u_0"),("u1","u_1"),
                     ("r0","r_0")):
        text = text.replace(old, new)
    return text.replace("^", "**")

def _sympy_to_torch(expr: sp.Expr, symbols: List[sp.Symbol]) -> Callable:
    code = str(expr)
    idx = {str(s): i for i, s in enumerate(symbols)}
    for s, i in idx.items():
        code = code.replace(s, f"inp[{i}]")
    def _safe_sqrt(z: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(z, min=1e-12))
    env = {"torch": torch, "sin": torch.sin, "cos": torch.cos, "sqrt": _safe_sqrt}
    raw = eval(f"lambda inp: {code}", env)
    def fn(inp: torch.Tensor):
        out = raw(inp)
        return out if torch.is_tensor(out) else torch.tensor(out, device=inp.device, dtype=inp.dtype)
    return fn

def build_symbolic_jacobian(
    f_dyn_rows: Sequence,     # SINDy rows for f(x,u): each has .library.function_names and .coef (Tensor)
    policies:   Sequence,     # policies: each has .library.function_names and .coef (Tensor)
    umin: float, umax: float, eps: float = 0.0,
):
    n_state  = len(f_dyn_rows)
    n_policy = len(policies)

    x_syms  = [sp.symbols(f"x_{i}") for i in range(n_state)]
    r_sym   = sp.symbols("r_0")
    u_syms  = [sp.symbols(f"u_{k}") for k in range(n_policy)]
    xr_syms = x_syms + [r_sym]        # ψ_k(x,r)
    xu_syms = x_syms + u_syms         # f_i(x,u)

    # ψ libraries per policy
    psi_exprs, psi_funcs, psi_sizes = [], [], []
    for pol in policies:
        terms = [sp.sympify(_normalize(t)) for t in pol.library.function_names]
        psi_exprs.append(terms)
        psi_funcs.append([_sympy_to_torch(e, xr_syms) for e in terms])
        psi_sizes.append(len(terms))

    # h_{i,k}(x,u) = df_i/du_k
    h_funcs: List[List[Callable]] = [[None]*n_policy for _ in range(n_state)]
    h_exprs: List[List[sp.Expr]]  = [[None]*n_policy for _ in range(n_state)]

    for i, f_row in enumerate(f_dyn_rows):
        dphi_num = [[] for _ in range(n_policy)]
        dphi_sym = [[] for _ in range(n_policy)]
        a_list = f_row.coef.detach().view(-1).cpu().tolist()
        for term_str, a_val in zip(f_row.library.function_names, a_list):
            expr = sp.sympify(_normalize(term_str))
            for k, uk in enumerate(u_syms):
                deriv = sp.diff(expr, uk)
                dphi_num[k].append(_sympy_to_torch(deriv, xu_syms))
                dphi_sym[k].append(sp.simplify(deriv * float(a_val)))
        for k in range(n_policy):
            h_exprs[i][k] = sp.simplify(sum(dphi_sym[k]))
            def _bind(fr=f_row, dlist=dphi_num[k]):
                def h_num(xu: torch.Tensor):
                    coeffs = fr.coef.view(-1)
                    vals   = torch.stack([fn(xu) for fn in dlist])
                    return torch.dot(coeffs, vals)
                return h_num
            h_funcs[i][k] = _bind()

    # symbolic matrix (optional debug)
    n_a_total = sum(psi_sizes)
    expr_table = [[None]*n_a_total for _ in range(n_state)]
    col = 0
    for k, (expr_list, n_k) in enumerate(zip(psi_exprs, psi_sizes)):
        for j in range(n_k):
            for i in range(n_state):
                expr_table[i][col + j] = sp.simplify(h_exprs[i][k] * expr_list[j])
        col += n_k
    expr_matrix = sp.Matrix(expr_table)

    class FastJac:
        def __init__(self):
            self.expr        = expr_matrix
            self.psi_sizes   = psi_sizes
            self.n_state     = n_state
            self.n_policy    = n_policy
            self.n_a_total   = n_a_total
            self.umin        = float(umin)
            self.umax        = float(umax)
            self.eps         = float(eps)
            self._psi_funcs  = psi_funcs
            self._h_funcs    = h_funcs
            self._pols       = policies

        @torch.no_grad()
        def __call__(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            # x: (1,n_state), r: (1,1) or (1, n_state); use r_0
            device, dtype = x.device, x.dtype
            x_vals = x.squeeze(0).tolist()
            r_val  = float(r.view(-1)[0].item())
            xr = torch.tensor(x_vals + [r_val], device=device, dtype=dtype)

            # per-policy u_raw, dua_k with own clamp mask
            u_raw_list, dua_cols = [], []
            for pol, psi_k in zip(self._pols, self._psi_funcs):
                a_vec = pol.coef.view(-1)
                psi   = torch.stack([fn(xr) for fn in psi_k])              # (#terms,)
                u_raw = torch.dot(a_vec, psi)
                u_raw_list.append(u_raw)
                mask_k = ((u_raw > self.umin + self.eps) &
                          (u_raw < self.umax - self.eps)).to(psi.dtype)
                dua_cols.append(mask_k * psi)                               # (#terms,)

            # clamped u for df/du eval
            u_clamped = [torch.clamp(u, min=self.umin, max=self.umax) for u in u_raw_list]
            xu = torch.tensor(x_vals + [float(u.item()) for u in u_clamped],
                              device=device, dtype=dtype)

            # h_mat: (n_state, n_policy)
            h_mat = torch.stack([
                        torch.stack([self._h_funcs[i][k](xu) for k in range(self.n_policy)])
                        for i in range(self.n_state)
                    ])

            # assemble J
            J = torch.zeros(self.n_state, self.n_a_total, device=device, dtype=dtype)
            col = 0
            for k, (dua_k, n_k) in enumerate(zip(dua_cols, self.psi_sizes)):
                J[:, col:col+n_k] = h_mat[:, [k]] * dua_k
                col += n_k
            return J

    return FastJac()
