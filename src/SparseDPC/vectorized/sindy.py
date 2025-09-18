import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict
import torch
import torch.nn as nn
from neuromancer.dynamics.ode import ODESystem
from typing import Optional, Sequence, List, Dict


@dataclass
class LibraryPlan:
    subset_global: torch.Tensor
    order: torch.Tensor
    take_bias: bool
    x_idx: Optional[torch.Tensor]
    u_idx: Optional[torch.Tensor]
    sqrt_x_idx: Optional[torch.Tensor]
    poly_sel: Dict[int, torch.Tensor]
    uu_pairs_sel: Optional[torch.Tensor]
    xu_sel: Optional[torch.Tensor]
    fourier_sin_sel: Optional[torch.Tensor]
    fourier_cos_sel: Optional[torch.Tensor]
    global2local: torch.Tensor


class CompiledFunctionLibrary:
    """
    Θ(X,U) with a compiled layout. If `is_policy=True`, control-like inputs
    are *named* 'r' (reference) instead of 'u' in `function_names`.
    Computation is unchanged: the second argument is still `U` (shape (B, n_control)).
    """
    def __init__(self,
                 n_features: int,
                 n_control: int,
                 *,
                 include_bias: bool = True,
                 max_degree: int = 2,
                 add_sqrt: bool = False,
                 add_xu: bool = True,
                 add_u_prod: bool = True,
                 add_fourier: bool = False,
                 max_freq: int = 1,
                 is_policy: bool = False):                     # NEW
        self.n_features = n_features
        self.n_control = n_control
        self.include_bias = include_bias
        self.max_degree = max_degree
        self.add_sqrt = add_sqrt
        self.add_xu = add_xu
        self.add_u_prod = add_u_prod
        self.add_fourier = add_fourier
        self.max_freq = max_freq
        self.is_policy = is_policy                            # NEW

        # ----- build static layout (offsets) once -----
        self.offsets: Dict[str, Tuple[int, int]] = {}
        self._poly_combos: Dict[int, torch.Tensor] = {}
        self._poly_deg_offsets: Dict[int, Tuple[int,int]] = {}
        self._uu_pairs = None
        self._freqs = None

        start = 0
        if include_bias:
            self.offsets["bias"] = (start, start+1); start += 1
        self.offsets["x"] = (start, start + n_features); start += n_features
        if n_control > 0:
            self.offsets["u"] = (start, start + n_control); start += n_control
        if add_sqrt:
            self.offsets["sqrt_x"] = (start, start + n_features); start += n_features

        if max_degree >= 2:
            for d in range(2, max_degree+1):
                combos = list(itertools.combinations_with_replacement(range(n_features), d))
                if combos:
                    self._poly_combos[d] = torch.tensor(combos, dtype=torch.long)
                    m_d = len(combos)
                    self._poly_deg_offsets[d] = (start, start + m_d); start += m_d

        if add_u_prod and n_control >= 2:
            p_idx, q_idx = torch.triu_indices(n_control, n_control, offset=1)
            self._uu_pairs = torch.stack([p_idx, q_idx], dim=0)  # (2, n_pairs)
            self.offsets["uu"] = (start, start + self._uu_pairs.shape[1]); start += self._uu_pairs.shape[1]

        if add_xu and n_control > 0:
            self.offsets["xu"] = (start, start + n_features*n_control); start += n_features*n_control

        if add_fourier and max_freq >= 1:
            self._freqs = torch.arange(1, max_freq+1).long()
            self.offsets["fourier_sin"] = (start, start + n_features*max_freq); start += n_features*max_freq
            self.offsets["fourier_cos"] = (start, start + n_features*max_freq); start += n_features*max_freq

        # Total shape (n_terms, nx+nu)
        self.shape = (start, n_features + n_control)

        # Build names once (using 'u' or 'r' according to is_policy)
        self.function_names: List[str] = self._build_names()

    # ---------- helper to (re)build names ----------
    def _build_names(self) -> List[str]:
        ctrl = "r" if self.is_policy else "u"               # NEW
        names: List[str] = []
        s = 0

        def add_range(label: str, count: int, fmt):
            nonlocal s
            a, b = s, s + count
            self.offsets[label] = (a, b)
            s = b
            names.extend(fmt(i) for i in range(count))

        # Respect existing offsets; only generate textual names
        if "bias" in self.offsets:
            names.append("1")

        # linear x
        nx = self.n_features
        names.extend([f"x{i}" for i in range(nx)])

        # linear controls
        if "u" in self.offsets:
            nu = self.n_control
            names.extend([f"{ctrl}{j}" for j in range(nu)])  # NEW label

        # sqrt(x)
        if "sqrt_x" in self.offsets:
            names.extend([f"sqrt(x{i})" for i in range(nx)])

        # poly(x)
        for d in sorted(self._poly_combos.keys()):
            combos = self._poly_combos[d].tolist()
            for combo in combos:
                counts = {}
                for idx in combo: counts[idx] = counts.get(idx, 0) + 1
                names.append("*".join([f"x{k}" if p==1 else f"x{k}^{p}" for k,p in sorted(counts.items())]))

        # uu
        if self._uu_pairs is not None:
            for p, q in zip(self._uu_pairs[0].tolist(), self._uu_pairs[1].tolist()):
                names.append(f"{ctrl}{p}*{ctrl}{q}")         # NEW label

        # xu
        if "xu" in self.offsets:
            for i in range(nx):
                for j in range(self.n_control):
                    names.append(f"x{i}*{ctrl}{j}")          # NEW label

        # fourier
        if "fourier_sin" in self.offsets:
            K = (self.offsets["fourier_sin"][1] - self.offsets["fourier_sin"][0]) // nx
            for i in range(nx):
                for k in range(1, K+1):
                    names.append(f"sin({k}·x{i})")
        if "fourier_cos" in self.offsets:
            K = (self.offsets["fourier_cos"][1] - self.offsets["fourier_cos"][0]) // nx
            for i in range(nx):
                for k in range(1, K+1):
                    names.append(f"cos({k}·x{i})")

        assert len(names) == self.shape[0]
        return names

    # Public toggle if you ever want to switch after creation
    def set_policy_names(self, is_policy: bool = True):
        self.is_policy = bool(is_policy)
        self.function_names = self._build_names()

    # ---- compile & evaluate (unchanged logic) ----
    def compile(self, subset_global: Optional[Sequence[int]] = None, device: Optional[torch.device] = None) -> LibraryPlan:
        n_terms = self.shape[0]
        if subset_global is None:
            subset = torch.arange(n_terms, dtype=torch.long, device=device)
        else:
            subset = torch.as_tensor(subset_global, dtype=torch.long, device=device)
        subset, _ = torch.sort(subset)

        def in_range(r):
            s,e = r
            mask = (subset >= s) & (subset < e)
            idx_local = subset[mask] - s
            return mask, idx_local

        take_bias = False
        x_idx = u_idx = sqrt_x_idx = None
        poly_sel: Dict[int, torch.Tensor] = {}
        uu_pairs_sel = xu_sel = fourier_sin_sel = fourier_cos_sel = None

        if "bias" in self.offsets:
            mask_bias, _ = in_range(self.offsets["bias"])
            take_bias = bool(mask_bias.any().item())

        mask_x, x_idx = in_range(self.offsets["x"])
        if not mask_x.any(): x_idx = None

        if "u" in self.offsets:
            mask_u, u_idx = in_range(self.offsets["u"])
            if not mask_u.any(): u_idx = None

        if "sqrt_x" in self.offsets:
            mask_sx, sqrt_x_idx = in_range(self.offsets["sqrt_x"])
            if not mask_sx.any(): sqrt_x_idx = None

        for d, combos in self._poly_combos.items():
            s,e = self._poly_deg_offsets[d]
            mask_pd, idx_pd = in_range((s,e))
            if mask_pd.any():
                poly_sel[d] = idx_pd

        if "uu" in self.offsets and self._uu_pairs is not None:
            s,e = self.offsets["uu"]
            mask_uu, idx_uu = in_range((s,e))
            if mask_uu.any():
                uu_pairs_sel = self._uu_pairs[:, idx_uu]

        if "xu" in self.offsets:
            s,e = self.offsets["xu"]
            mask_xu, idx_xu = in_range((s,e))
            if mask_xu.any():
                nx, nu = self.n_features, self.n_control
                i_idx = idx_xu // nu
                j_idx = idx_xu %  nu
                xu_sel = torch.stack([i_idx, j_idx], dim=0)

        if "fourier_sin" in self.offsets:
            s,e = self.offsets["fourier_sin"]
            mask_fs, idx_fs = in_range((s,e))
            if mask_fs.any():
                nx, K = self.n_features, (e - s) // self.n_features
                i_idx = idx_fs // K
                k_idx0 = idx_fs %  K
                fourier_sin_sel = torch.stack([i_idx, k_idx0], dim=0)

        if "fourier_cos" in self.offsets:
            s,e = self.offsets["fourier_cos"]
            mask_fc, idx_fc = in_range((s,e))
            if mask_fc.any():
                nx, K = self.n_features, (e - s) // self.n_features
                i_idx = idx_fc // K
                k_idx0 = idx_fc %  K
                fourier_cos_sel = torch.stack([i_idx, k_idx0], dim=0)

        g2l = torch.full((self.shape[0],), -1, dtype=torch.long, device=subset.device)
        g2l[subset] = torch.arange(subset.numel(), device=subset.device)

        return LibraryPlan(
            subset_global=subset,
            order=subset.clone(),
            take_bias=take_bias,
            x_idx=x_idx,
            u_idx=u_idx,
            sqrt_x_idx=sqrt_x_idx,
            poly_sel=poly_sel,
            uu_pairs_sel=uu_pairs_sel,
            xu_sel=xu_sel,
            fourier_sin_sel=fourier_sin_sel,
            fourier_cos_sel=fourier_cos_sel,
            global2local=g2l
        )

    def evaluate_plan(self, x: torch.Tensor, u: Optional[torch.Tensor], plan: LibraryPlan) -> torch.Tensor:
        B = x.shape[0]; device = x.device
        outs = []
        if plan.take_bias: outs.append(torch.ones(B, 1, device=device))
        if plan.x_idx is not None: outs.append(x[:, plan.x_idx])
        if plan.u_idx is not None: outs.append(u[:, plan.u_idx])
        if plan.sqrt_x_idx is not None: outs.append(torch.sqrt(torch.clamp(x[:, plan.sqrt_x_idx], min=1e-12)))
        for d, rows in plan.poly_sel.items():
            combos = self._poly_combos[d].to(device)[rows]
            Xsel = x.index_select(1, combos.reshape(-1)).reshape(B, combos.shape[0], d)
            outs.append(Xsel.prod(dim=2))
        if plan.uu_pairs_sel is not None:
            p_idx = plan.uu_pairs_sel[0].to(device); q_idx = plan.uu_pairs_sel[1].to(device)
            outs.append(u[:, p_idx] * u[:, q_idx])
        if plan.xu_sel is not None:
            i_idx = plan.xu_sel[0].to(device); j_idx = plan.xu_sel[1].to(device)
            outs.append(x[:, i_idx] * u[:, j_idx])
        if plan.fourier_sin_sel is not None:
            i_idx = plan.fourier_sin_sel[0].to(device); k_idx0 = plan.fourier_sin_sel[1].to(device)
            freqs = (k_idx0 + 1).to(x.dtype).view(1, -1)
            outs.append(torch.sin(x[:, i_idx] * freqs))
        if plan.fourier_cos_sel is not None:
            i_idx = plan.fourier_cos_sel[0].to(device); k_idx0 = plan.fourier_cos_sel[1].to(device)
            freqs = (k_idx0 + 1).to(x.dtype).view(1, -1)
            outs.append(torch.cos(x[:, i_idx] * freqs))
        return torch.cat(outs, dim=1) if outs else torch.empty(B, 0, device=device)

    def evaluate(self, x: torch.Tensor, u: Optional[torch.Tensor]) -> torch.Tensor:
        plan = self.compile(None, device=x.device)
        return self.evaluate_plan(x, u, plan)



class SINDyVectorized(ODESystem):
    """
    Multi-output SINDy with per-state coefficient vectors (variable length).
    Uses a CompiledFunctionLibrary and recompiles a fast plan for the
    union of active columns across states (so forward computes only needed Θ cols).
    """
    def __init__(self,
                 library: CompiledFunctionLibrary,
                 n_out: Optional[int] = None,
                 policy_name: Optional[list] = None,
                 seed: Optional[int] = None,
                 device: torch.device = torch.device("cpu")):
        assert isinstance(library, CompiledFunctionLibrary), "`library` invalid"
        self.library = library
        self.n_out = n_out or library.n_features
        self.policy_name = policy_name

        super().__init__(library.shape[1], self.n_out)

        n_terms = library.shape[0]
        gen = None if seed is None else torch.Generator(device=device).manual_seed(seed)

        # per-state active global columns + coef vectors
        self.active_idx: List[List[int]] = [list(range(n_terms)) for _ in range(self.n_out)]
        self.Xi = nn.ParameterList([
            nn.Parameter(torch.empty(n_terms, 1, device=device).uniform_(-0.05, 0.05, generator=gen))
            for _ in range(self.n_out)
        ])

        self.function_names = list(library.function_names)
        self._plan: LibraryPlan = self._recompile_fast_path(device=device)

        self.float()

    # compile fast plan for the union of active terms (called at init and after pruning)
    def _recompile_fast_path(self, device=None) -> LibraryPlan:
        union = sorted(set(idx for s in self.active_idx for idx in s))
        plan = self.library.compile(union, device=device)
        # Build per-state mapping from global -> local-in-plan (LongTensor)
        self._state_local_idx: List[torch.Tensor] = []
        for s in self.active_idx:
            g = torch.tensor(s, dtype=torch.long, device=plan.global2local.device)
            l = plan.global2local[g]
            assert (l >= 0).all(), "Plan missing some active columns."
            self._state_local_idx.append(l)
        return plan

    # read-only dense view
    @property
    def coef(self) -> torch.Tensor:
        device = self.Xi[0].device
        n_terms = len(self.function_names)
        C = torch.zeros(n_terms, self.n_out, device=device)
        for i in range(self.n_out):
            idx = torch.tensor(self.active_idx[i], dtype=torch.long, device=device)
            C[idx, i] = self.Xi[i].squeeze(1)
        return C

    def ode_equations(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.to(self.Xi[0].device)
        if u is not None: u = u.to(self.Xi[0].device)
        Theta_sub = self.library.evaluate_plan(x, u, self._plan)  # (B, M_union)

        rhs = []
        for i in range(self.n_out):
            Theta_i = Theta_sub.index_select(1, self._state_local_idx[i])  # (B, n_i)
            rhs_i = Theta_i @ self.Xi[i]                                   # (B,1)
            rhs.append(rhs_i)
        return torch.cat(rhs, dim=1)                                       # (B, n_out)

    @torch.no_grad()
    def prune_(self,
               thresholds: float | Sequence[float] = 1e-3,
               *,
               noise_after: float = 0.0,
               nonzero_only: bool = True,
               eps: float = 0.0) -> Dict:
        """Hard magnitude pruning per state; shrinks Xi[i] and active_idx[i], then recompiles plan."""
        if isinstance(thresholds, (int, float)):
            thr = [float(thresholds)] * self.n_out
        else:
            thr = list(thresholds); assert len(thr) == self.n_out

        changed_states = []
        kept_counts, removed_counts = [], []

        for i in range(self.n_out):
            coef_i = self.Xi[i].squeeze(1)                         # (n_i,)
            keep_mask = (coef_i.abs() >= thr[i])
            if keep_mask.sum().item() == 0:
                keep_mask[torch.argmax(coef_i.abs())] = True

            if keep_mask.sum().item() < coef_i.numel():
                new_loc = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).tolist()
                self.active_idx[i] = [self.active_idx[i][k] for k in new_loc]

                kept_counts.append(len(self.active_idx[i]))
                removed_counts.append(int(coef_i.numel() - len(self.active_idx[i])))

                new_coef = self.Xi[i][keep_mask, :].detach().clone()
                if noise_after and noise_after > 0.0:
                    if nonzero_only:
                        nz = (new_coef.abs() > eps)
                        new_coef[nz] += noise_after * torch.randn_like(new_coef[nz])
                    else:
                        new_coef += noise_after * torch.randn_like(new_coef)

                self.Xi[i] = nn.Parameter(new_coef, requires_grad=True)
                changed_states.append(i)
            else:
                kept_counts.append(int(coef_i.numel()))
                removed_counts.append(0)

        # recompile fast plan for the union; refresh per-state local indices
        self._plan = self._recompile_fast_path(device=self.Xi[0].device)

        return {
            "changed": len(changed_states) > 0,
            "changed_states": changed_states,
            "kept_counts": kept_counts,
            "removed_counts": removed_counts,
        }

    @torch.no_grad()
    def pretty_print(self, labels: Optional[List[str]] = None, coef_tol: float = 1e-12):
        if labels is None:
            labels = [f"x{i}" for i in range(self.n_out)]
        for i in range(self.n_out):
            names_i = [self.function_names[j] for j in self.active_idx[i]]
            coefs_i = self.Xi[i].squeeze(1).detach().cpu().tolist()
            terms = [f"{c:+.4e}·{n}" for n, c in zip(names_i, coefs_i) if abs(c) > coef_tol]
            rhs = " + ".join(terms) if terms else "0"
            lhs = self.policy_name[i] if self.policy_name else f"d{labels[i]}/dt"
            print(f"{lhs} = {rhs}")

    @torch.no_grad()
    def add_terms_(self, state: int, global_indices: Sequence[int], init_scale: float = 1e-3):
        device = self.Xi[state].device
        cur = set(self.active_idx[state])
        to_add = [g for g in global_indices if g not in cur]
        if not to_add: return
        self.active_idx[state].extend(to_add)
        add_coef = init_scale * torch.randn(len(to_add), 1, device=device)
        self.Xi[state] = nn.Parameter(torch.cat([self.Xi[state].data, add_coef], dim=0), requires_grad=True)
        self._plan = self._recompile_fast_path(device=device)
