import torch, sympy as sp
from typing import Sequence, Callable, List

# ---------------- helpers ----------------

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

# --------------- builder -----------------

def build_symbolic_jacobian(
    f_dyn_rows: Sequence,     # rows of f(x,u); each has .library.function_names and .coef (Tensor)
    policies:   Sequence,     # each has .library.function_names and .coef (Tensor)
    umin: float | Sequence | None,
    umax: float | Sequence | None,
    eps: float = 0.0,         # kept (unused in masking below; mask uses <= / >=)
):
    """
    Returns FastJac with:
      - __call__(x, r, u) -> (n_state, n_a_total) numeric Jacobian
        *uses the PROVIDED u everywhere*; if u_k <= umin_k or u_k >= umax_k,
        gradients for the columns of policy k are zeroed (fast mask).
      - Van-der-Pol–style coupling for rows with no explicit u-dependence:
        J_i := sum_j (∂f_i/∂x_j) * J_j over rows j that depend on u.

    Exposes:
      .expr_base, .expr_coupled, .expr, .psi_sizes
    """
    n_state  = len(f_dyn_rows)
    n_policy = len(policies)

    # symbols
    x_syms  = [sp.symbols(f"x_{i}") for i in range(n_state)]
    r_sym   = sp.symbols("r_0")
    u_syms  = [sp.symbols(f"u_{k}") for k in range(n_policy)]
    xr_syms = x_syms + [r_sym]          # for ψ(x,r)
    xu_syms = x_syms + u_syms           # for f(x,u) and its partials

    # ψ libraries per policy
    psi_exprs, psi_funcs, psi_sizes = [], [], []
    for pol in policies:
        terms = [sp.sympify(_normalize(t)) for t in pol.library.function_names]
        psi_exprs.append(terms)
        psi_funcs.append([_sympy_to_torch(e, xr_syms) for e in terms])
        psi_sizes.append(len(terms))

    # containers
    h_funcs: List[List[Callable]] = [[None]*n_policy for _ in range(n_state)]  # h[i][k] = ∂f_i/∂u_k
    h_exprs: List[List[sp.Expr]]  = [[None]*n_policy for _ in range(n_state)]
    f_exprs: List[sp.Expr]        = [None]*n_state
    g_funcs: List[List[Callable]] = [[None]*n_state for _ in range(n_state)]   # g[i][j] = ∂f_i/∂x_j
    G_exprs: List[List[sp.Expr]]  = [[None]*n_state for _ in range(n_state)]
    row_has_u: List[bool]         = [False]*n_state

    # build f_i, ∂f_i/∂u_k, and ∂f_i/∂x_j
    for i, f_row in enumerate(f_dyn_rows):
        dphi_num = [[] for _ in range(n_policy)]
        dphi_sym = [[] for _ in range(n_policy)]
        a_list = f_row.coef.detach().view(-1).cpu().tolist()

        f_i_expr = 0
        for term_str, a_val in zip(f_row.library.function_names, a_list):
            expr = sp.sympify(_normalize(term_str))
            f_i_expr += sp.simplify(float(a_val) * expr)
            for k, uk in enumerate(u_syms):
                deriv = sp.diff(expr, uk)
                dphi_num[k].append(_sympy_to_torch(deriv, xu_syms))
                dphi_sym[k].append(sp.simplify(deriv * float(a_val)))

        for k in range(n_policy):
            h_exprs[i][k] = sp.simplify(sum(dphi_sym[k]))
            def _bind(fr=f_row, dlist=dphi_num[k]):
                def h_num(xu: torch.Tensor):
                    coeffs = fr.coef.view(-1)
                    vals   = torch.stack([fn(xu) for fn in dlist]) if len(dlist) else torch.zeros_like(coeffs)
                    return torch.dot(coeffs, vals)
                return h_num
            h_funcs[i][k] = _bind()

        f_exprs[i] = sp.simplify(f_i_expr)
        row_has_u[i] = any(not sp.simplify(h_exprs[i][k]).equals(0) for k in range(n_policy))

        for j, xj in enumerate(x_syms):
            gij_expr = sp.simplify(sp.diff(f_exprs[i], xj))
            G_exprs[i][j] = gij_expr
            g_funcs[i][j] = _sympy_to_torch(gij_expr, xu_syms)

    # ----- symbolic matrices -----

    # base expr: J_base = (∂f/∂u)·(∂u/∂a)
    n_a_total = sum(psi_sizes)
    expr_table = [[None]*n_a_total for _ in range(n_state)]
    col = 0
    for k, (expr_list, n_k) in enumerate(zip(psi_exprs, psi_sizes)):
        for j in range(n_k):
            for i in range(n_state):
                expr_table[i][col + j] = sp.simplify(h_exprs[i][k] * expr_list[j])
        col += n_k
    expr_matrix_base = sp.Matrix(expr_table)

    # coupled expr: replace rows with no-u by Σ_j (∂f_i/∂x_j) * J_base_row_j
    rows_with_u = [i for i in range(n_state) if row_has_u[i]]
    expr_matrix_coupled = sp.MutableDenseMatrix(n_state, n_a_total, [0]*(n_state*n_a_total))
    for i in range(n_state):
        for c in range(n_a_total):
            if row_has_u[i]:
                expr_matrix_coupled[i, c] = expr_matrix_base[i, c]
            else:
                s = 0
                for j in rows_with_u:
                    s += sp.simplify(G_exprs[i][j] * expr_matrix_base[j, c])
                expr_matrix_coupled[i, c] = sp.simplify(s)

    # ----- runtime class -----

    class FastJac:
        def __init__(self):
            self.expr_base     = expr_matrix_base
            self.expr_coupled  = expr_matrix_coupled
            self.expr          = self.expr_coupled
            self.psi_sizes     = psi_sizes
            self.n_state       = n_state
            self.n_policy      = n_policy
            self.n_a_total     = n_a_total

            # bounds (allow scalar or per-channel)
            def _to_bound(v):
                if v is None:
                    return None
                if torch.is_tensor(v):
                    t = v.clone().detach().float()
                else:
                    t = torch.tensor(v, dtype=torch.float32)
                if t.ndim == 0:
                    t = t.expand(n_policy)
                return t
            self.umin = _to_bound(umin)
            self.umax = _to_bound(umax)

            # compiled functions
            self._psi_funcs = psi_funcs
            self._h_funcs   = h_funcs
            self._g_funcs   = g_funcs
            self._pols      = policies
            self._row_has_u = row_has_u

        @torch.no_grad()
        def __call__(self, x: torch.Tensor, r: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            """
            x: (B, n_state)
            r: (B, 1) or (B, n_state)  (only r[:,0] used)
            u: (B, n_policy)  -- PROVIDED (already clamped if you want)
            Returns: (B, n_state, n_a_total) numeric Jacobian with:
                     - zero columns for channels k where u_k <= umin_k or u_k >= umax_k
                     - coupling applied for rows with no explicit u-dependence
            Note: evaluation under the hood is per-sample (SymPy lambdas are scalar),
                  but the API is batched; results are stacked across batch.
            """
            if x.dim() != 2 or x.size(1) != self.n_state:
                raise ValueError(f"x must be (B, {self.n_state})")
            if u.dim() != 2 or u.size(1) != self.n_policy:
                raise ValueError(f"u must be (B, {self.n_policy})")
            if r.dim() != 2 or r.size(0) != x.size(0):
                raise ValueError("r must be (B, 1) or (B, n_state)")

            B = x.size(0)
            J_out = x.new_zeros((B, self.n_state, self.n_a_total))

            # vectorized over batch by small loop (SymPy eval is scalar)
            for b in range(B):
                device, dtype = x.device, x.dtype
                x_vals = x[b].tolist()
                r_val  = float(r[b, 0].item())
                # inputs for ψ(x,r): [x..., r0]
                xr = torch.tensor(x_vals + [r_val], device=device, dtype=dtype)

                # evaluate ∂u/∂a columns (psi) per policy
                dua_cols = []
                for psi_k in self._psi_funcs:
                    psi_vals = torch.stack([fn(xr) for fn in psi_k])  # (#terms,)
                    dua_cols.append(psi_vals)

                # mask for saturation (<= umin or >= umax) using PROVIDED u
                # mask_k = 1 inside bounds, 0 when saturated
                if (self.umin is None) and (self.umax is None):
                    mask = torch.ones(self.n_policy, dtype=dtype, device=device)
                else:
                    mask = torch.ones(self.n_policy, dtype=dtype, device=device)
                    if self.umin is not None:
                        mask = mask * (u[b] > self.umin.to(device=device, dtype=dtype)).to(dtype)
                    if self.umax is not None:
                        mask = mask * (u[b] < self.umax.to(device=device, dtype=dtype)).to(dtype)

                # inputs for h(x,u) and g(x,u): [x..., u...,] (r not used in f; add if needed)
                xu = torch.tensor(x_vals + u[b].tolist(), device=device, dtype=dtype)

                # h_mat: (n_state, n_policy) with h[i,k] = ∂f_i/∂u_k(x,u)
                h_mat = torch.stack([
                            torch.stack([self._h_funcs[i][k](xu) for k in range(self.n_policy)])
                            for i in range(self.n_state)
                        ])  # (n_state, n_policy)

                # base J = (∂f/∂u) · (∂u/∂a) with saturation mask per channel
                J = x.new_zeros((self.n_state, self.n_a_total))
                col = 0
                for k, (dua_k, n_k) in enumerate(zip(dua_cols, self.psi_sizes)):
                    J[:, col:col+n_k] = h_mat[:, [k]] * (mask[k] * dua_k)
                    col += n_k

                # coupling for rows with no explicit u-dependence
                rows_with_u = [j for j in range(self.n_state) if self._row_has_u[j]]
                if rows_with_u:
                    # G[i,j] = ∂f_i/∂x_j(x,u)
                    G = torch.stack([
                            torch.stack([self._g_funcs[i][j](xu) for j in range(self.n_state)])
                            for i in range(self.n_state)
                        ])  # (n_state, n_state)
                    for i in range(self.n_state):
                        if not self._row_has_u[i]:
                            Ji = x.new_zeros(self.n_a_total)
                            for j in rows_with_u:
                                Ji = Ji + G[i, j] * J[j, :]
                            J[i, :] = Ji

                J_out[b] = J

            return J_out

    return FastJac()
