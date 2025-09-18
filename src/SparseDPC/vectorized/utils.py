import numpy as np
import time
import os
import dill
import torch.nn as nn
from neuromancer.dynamics.ode import ODESystem
from neuromancer.psl.signals import step, sines, periodic, noise, walk
from neuromancer.psl.base import ODE_NonAutonomous as ODE
from neuromancer.psl.base import cast_backend
import casadi as ca
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class DoubleIntegrator2D(ODE):
    """
    2D mass–spring–damper (double integrator with optional k, c).
    State:  x = [x, xdot, y, ydot]
    Input:  u = [u_x, u_y]
    ODEs:
        xdot   = v_x
        vxdot  = -(c_x/m_x) v_x - (k_x/m_x) x + (1/m_x) u_x
        ydot   = v_y
        vydot  = -(c_y/m_y) v_y - (k_y/m_y) y + (1/m_y) u_y
    """

    @property
    def params(self):
        # variables: initial state
        variables = {'x0': [0., 0., 0., 0.]}
        # constants:
        constants = {'ts': 0.1}

        # default parameters
        parameters = {
            # masses
            'mx': 1.0,    # mass on x-axis
            'my': 1.0,    # mass on y-axis
            # damping
            'cx': 0.2,    # damping on x-axis
            'cy': 0.2,    # damping on y-axis
            # stiffness
            'kx': 0.0,    # stiffness on x-axis
            'ky': 0.0,    # stiffness on y-axis
        }
        meta = {}
        return variables, constants, parameters, meta

    @property
    def umin(self):
        # symmetric input box (force limits), adjust as you like
        return np.array([-1.0, -1.0], dtype=np.float32)

    @property
    def umax(self):
        return np.array([+1.0, +1.0], dtype=np.float32)

    @cast_backend
    def get_x0(self):
        # random small displacement, zero-ish velocities
        pos = self.rng.uniform(low=-1.0, high=1.0, size=2)
        vel = self.rng.uniform(low=-0.1, high=0.1, size=2)
        return np.concatenate([pos[0:1], vel[0:1], pos[1:2], vel[1:2]], axis=0)

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        # piecewise-constant random steps in both axes (ZOH)
        u = step(
            nsim=nsim, d=2,
            min= -1.0, max=1.0,
            randsteps=int(np.ceil(nsim / 20)),
            rng=self.rng
        )
        return u

    @cast_backend
    def equations(self, t, x, u):
        """
        x: [x, xdot, y, ydot]
        u: [ux, uy]
        returns: [xdot, vxdot, ydot, vydot]
        """
        # unpack state
        px  = x[0]
        vx  = x[1]
        py  = x[2]
        vy  = x[3]

        # clip inputs to bounds (backend-friendly)
        ux = self.B.core.clip(u[0], self.umin[0], self.umax[0])
        uy = self.B.core.clip(u[1], self.umin[1], self.umax[1])

        # parameters (assumed exposed as attributes by ODE base, like TwoTank did)
        mx, my = self.mx, self.my
        cx, cy = self.cx, self.cy
        kx, ky = self.kx, self.ky

        # ODEs
        dpx = vx
        dvx = -(cx / mx) * vx - (kx / mx) * px + (1.0 / mx) * ux

        dpy = vy
        dvy = -(cy / my) * vy - (ky / my) * py + (1.0 / my) * uy

        return [dpx, dvx, dpy, dvy]

class DoubleIntegrator2DParam(ODESystem):
    def __init__(self, insize=6, outsize=4):
        """
        2D mass–spring–damper (double integrator with optional damping/stiffness).

        insize = 6  -> concatenation of state and input: [x, xdot, y, ydot, ux, uy]
        outsize = 4 -> state derivatives:                [xdot, xddot, ydot, yddot]
        """
        super().__init__(insize=insize, outsize=outsize)
        # learnable parameters (per-axis)
        self.mx = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.my = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.cx = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.cy = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.kx = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.ky = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def ode_equations(self, x, u):
        """
        x: (..., 4) = [x, xdot, y, ydot]
        u: (..., 2) = [ux, uy]
        returns (..., 4): [xdot, xddot, ydot, yddot]
        """
        # states
        px = x[:, [0]]
        vx = x[:, [1]]
        py = x[:, [2]]
        vy = x[:, [3]]

        # inputs (clip similar to sample style; adjust bounds as needed)
        ux = torch.clip(u[:, [0]], min=-1.0, max=1.0)
        uy = torch.clip(u[:, [1]], min=-1.0, max=1.0)

        # dynamics
        dpx = vx
        dvx = -(self.cx / (self.mx + 1e-15)) * vx - (self.kx / (self.mx + 1e-15)) * px + (1.0 / (self.mx + 1e-15)) * ux

        dpy = vy
        dvy = -(self.cy / (self.my + 1e-15)) * vy - (self.ky / (self.my + 1e-15)) * py + (1.0 / (self.my + 1e-15)) * uy

        return torch.cat([dpx, dvx, dpy, dvy], dim=-1)

class CustomLogger:
    def __init__(self, args=None, savedir='test', verbosity=10,
                 stdout=('nstep_dev_loss', 'loop_dev_loss', 'best_loop_dev_loss',
                         'nstep_dev_ref_loss', 'loop_dev_ref_loss')):
        """
        :param args: (Namespace) returned by argparse.ArgumentParser.parse_args()
        :param savedir: (str) Folder to write results to.
        :param verbosity: (int) Print to stdout every verbosity epochs
        :param stdout: (list of str) Metrics to print to stdout. These should correspond to keys in the output dictionary of the Problem
        """
        os.makedirs(savedir, exist_ok=True)
        self.stdout = stdout
        self.savedir = savedir
        self.verbosity = verbosity
        self.start_time = time.time()
        self.step = 0
        self.args = args
        self.log_parameters()

    def log_parameters(self):
        """
        Print experiment parameters to stdout

        :param args: (Namespace) returned by argparse.ArgumentParser.parse_args()
        """
        print(self.args)

    def log_weights(self, model):
        """

        :param model: (nn.Module)
        :return: (int) The number of learnable parameters in the model
        """
        nweights = sum([i.numel() for i in list(model.parameters()) if i.requires_grad])
        print(f'Number of parameters: {nweights}')
        return nweights

    def log_metrics(self, output, step=None):
        """
        Print metrics to stdout.

        :param output: (dict {str: tensor}) Will only record 0d tensors (scalars)
        :param step: (int) Epoch of training
        """
        if step is None:
            step = self.step
        else:
            self.step = step
        if step % self.verbosity == 0:
            elapsed_time = time.time() - self.start_time
            entries = [f'epoch: {step}']
            for k, v in output.items():
                try:
                    if k in self.stdout:
                        entries.append(f'{k}: {v.item():.3e}')
                except (ValueError, AttributeError) as e:
                    pass
            entries.append(f'eltime: {elapsed_time: .5f}')
            print('\t'.join([e for e in entries if 'reg_error' not in e]))

    def log_artifacts(self, artifacts):
        """
        Stores artifacts created in training to disc.

        :param artifacts: (dict {str: Object})
        """
        for k, v in artifacts.items():
            savepath = os.path.join(self.savedir, k)
            torch.save(v, savepath, pickle_module=dill)

    def clean_up(self):
        pass


def solve_and_plot_ipopt(
    sample: dict,
    pol_configs: dict,
    *,
    ts: float,
    ellipse_cfg: dict | None = None,
    bounds: dict | None = None,
    # dynamics options
    use_msd: bool = False,     # True → mass–spring–damper; False → double integrator
    m: float = 1.0,
    k: float = 0.0,
    c: float = 0.0,
    # objective shaping
    track_whole_horizon: bool = True,   # True → track ref along horizon; False → terminal-only
    refstep: int = 1,
    add_state_smoothing: bool = True,
    add_control_smoothing: bool = True,
    title_xy: str = "IPOPT trajectory with keep-out ellipse",
):
    """
    Solve min_u J s.t. X_{k+1} = X_k + ts * f(X_k, U_k), and plot results.

    States: [x, xdot, y, ydot]
    Inputs: [ux, uy]
    sample['xn']: (1,1,4)
    sample['r'] : (1,N+1,2)  (# positions only, repeated across horizon in your data)
    """
    # -------------------- unpack data --------------------
    assert 'xn' in sample and 'r' in sample, "sample must have 'xn' and 'r'"
    x0 = sample['xn'][0, 0, :].detach().cpu().numpy()        # (4,)
    R  = sample['r'][0].detach().cpu().numpy()               # (N+1,2)
    N  = R.shape[0] - 1                                      # horizon length

    if ellipse_cfg is None:
        ellipse_cfg = dict(p=4.0, b=1.0, c=0.0, d=0.0)
    p = float(ellipse_cfg['p'])
    b = float(ellipse_cfg['b'])
    cC = float(ellipse_cfg['c'])
    dC = float(ellipse_cfg['d'])

    if bounds is None:
        bounds = dict(xmin=-2.0, xmax=2.0, umin=-1.0, umax=1.0)

    xmin = bounds['xmin']; xmax = bounds['xmax']
    umin = bounds['umin']; umax = bounds['umax']

    # weights
    Q_con_obs = float(pol_configs.get("Q_con_obs", 100.0))
    Q_con_x   = float(pol_configs.get("Q_con_x",   10.0))
    Q_u       = float(pol_configs.get("Q_u",        0.1))
    Q_r       = float(pol_configs.get("Q_r",        2.0))
    Q_dx      = float(pol_configs.get("Q_dx",       0.1))
    Q_du      = float(pol_configs.get("Q_du",       0.1))

    # -------------------- CasADi variables --------------------
    opti = ca.Opti()
    X = opti.variable(4, N+1)     # [x, xdot, y, ydot] over horizon
    U = opti.variable(2, N)       # [ux, uy] over horizon

    # -------------------- dynamics --------------------
    def f_cont(x, u):
        x_pos, x_dot, y_pos, y_dot = x[0], x[1], x[2], x[3]
        ux, uy = u[0], u[1]
        if not use_msd:
            ddx = ux / m
            ddy = uy / m
        else:
            ddx = (ux - c*x_dot - k*x_pos) / m
            ddy = (uy - c*y_dot - k*y_pos) / m
        return ca.vertcat(x_dot, ddx, y_dot, ddy)

    # Forward Euler discretization
    for k_step in range(N):
        xk = X[:, k_step]
        uk = U[:, k_step]
        xk_next = X[:, k_step+1]
        opti.subject_to(xk_next == xk + ts * f_cont(xk, uk))

    # Initial condition
    opti.subject_to(X[:, 0] == x0)

    # -------------------- bounds --------------------
    def to_arr(val, shape_len):
        if np.isscalar(val):
            return np.full(shape_len, float(val))
        arr = np.asarray(val, dtype=float)
        assert arr.shape == (shape_len,)
        return arr

    xmin_arr = to_arr(xmin, 4); xmax_arr = to_arr(xmax, 4)
    for i in range(4):
        opti.subject_to(opti.bounded(xmin_arr[i], X[i, :], xmax_arr[i]))

    umin_arr = to_arr(umin, 2); umax_arr = to_arr(umax, 2)
    for i in range(2):
        opti.subject_to(opti.bounded(umin_arr[i], U[i, :], umax_arr[i]))


    # -------------------- objective --------------------
    obj = 0

    # 1) tracking: positions only
    if track_whole_horizon:
        for k_step in range(N+1):
            rx, ry = R[k_step, 0], R[k_step, 1]
            obj += Q_r * ca.sumsqr(ca.vertcat(X[0, k_step]-rx, X[2, k_step]-ry))
    else:
        rxN, ryN = R[-1, 0], R[-1, 1]
        obj += Q_r * ca.sumsqr(ca.vertcat(X[0, -refstep:]-rxN, X[2, -refstep:]-ryN))

    obj += Q_r * ca.sumsqr(X[1, -1:])  # final xdot = 0
    obj += Q_r * ca.sumsqr(X[3, -1:])   # final ydot = 0
    # 2) effort
    obj += Q_u * ca.sumsqr(U)

    # 3) smoothing
    if add_state_smoothing and N >= 1:
        obj += Q_dx * ca.sumsqr(X[[0,2], 1:] - X[[0,2], :-1])
    if add_control_smoothing and N >= 2:
        obj += Q_du * ca.sumsqr(U[:, 1:] - U[:, :-1])

    # 4) obstacle keep-out (hinge^2): active only if INSIDE ellipse
    r2_boundary = (p*0.5)**2
    for k_step in range(N+1):
        xk, yk = X[0, k_step], X[2, k_step]
        r2 = b*(xk - cC)**2 + (yk - dC)**2
        violation = r2_boundary - r2                 # > 0 only when inside
        hinge = 0.5*(violation + ca.fabs(violation)) # ReLU
        obj += Q_con_obs * ca.sumsqr(hinge)

    # optional soft state box penalties (if desired)
    if Q_con_x > 0:
        obj += Q_con_x * ca.sumsqr(ca.fmax(0, (xmin_arr.reshape(-1,1) - X)))
        obj += Q_con_x * ca.sumsqr(ca.fmax(0, (X - xmax_arr.reshape(-1,1))))

    opti.minimize(obj)

    # -------------------- solver --------------------
    p_opts = {"print_time": False}
    s_opts = {"max_iter": 2000, "tol": 1e-6, "print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)

    # Initial guess: straight line in position, zero inputs
    Xg = np.tile(x0.reshape(-1,1), (1, N+1))
    rxN, ryN = R[-1, 0], R[-1, 1]
    Xg[0, :] = np.linspace(x0[0], rxN, N+1)
    Xg[2, :] = np.linspace(x0[2], ryN, N+1)
    opti.set_initial(X, Xg)
    opti.set_initial(U, 0.0)

    sol = opti.solve()
    Xsol = np.array(sol.value(X))  # (4, N+1)
    Usol = np.array(sol.value(U))  # (2, N)
    J    = float(sol.value(obj))

    # -------------------- plotting --------------------
    # 1) XY with ellipse
    fig1, ax1 = plt.subplots(figsize=(6,6))
    # ellipse semi-axes from b*(x-c)^2 + (y-d)^2 = (p/2)^2
    a_x = (p*0.5)/np.sqrt(b)
    a_y = (p*0.5)
    ell = Ellipse((cC, dC), width=2*a_x, height=2*a_y, angle=0.0,
                  facecolor="tab:orange", edgecolor="tab:orange", alpha=0.25)
    ax1.add_patch(ell)
    ax1.plot(Xsol[0, :], Xsol[2, :], marker="o", markersize=2, linewidth=1.5, label="trajectory")
    ax1.scatter([Xsol[0, 0]], [Xsol[2, 0]], c="green", marker="x", s=80, label="start")
    ax1.scatter([R[-1,0]], [R[-1,1]], c="red", marker="*", s=120, label="ref (terminal)")
    ax1.set_title(title_xy)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("equal"); ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xlim(min(xmin if np.isscalar(xmin) else min(xmin), Xsol[0,:].min(), cC-1.5*a_x),
                 max(xmax if np.isscalar(xmax) else max(xmax), Xsol[0,:].max(), cC+1.5*a_x))
    ax1.set_ylim(min(xmin if np.isscalar(xmin) else min(xmin), Xsol[2,:].min(), dC-1.5*a_y),
                 max(xmax if np.isscalar(xmax) else max(xmax), Xsol[2,:].max(), dC+1.5*a_y))
    ax1.legend(loc="best")

    # 2) time-series of states and inputs
    t = np.arange(N+1)*ts
    fig2, axs = plt.subplots(3, 1, figsize=(8,7), sharex=True)
    axs[0].plot(t, Xsol[0, :], label="x"); axs[0].plot(t, Xsol[2, :], label="y")
    axs[0].plot(t, np.repeat(R[:,0], 1), "--", alpha=0.5, label="x_ref")
    axs[0].plot(t, np.repeat(R[:,1], 1), "--", alpha=0.5, label="y_ref")
    axs[0].set_ylabel("pos"); axs[0].grid(True, ls="--", alpha=0.5); axs[0].legend(loc="best")

    axs[1].plot(t, Xsol[1, :], label="xdot"); axs[1].plot(t, Xsol[3, :], label="ydot")
    axs[1].set_ylabel("vel"); axs[1].grid(True, ls="--", alpha=0.5); axs[1].legend(loc="best")

    tt = np.arange(N)*ts
    axs[2].step(tt, Usol[0, :], where="post", label="ux")
    axs[2].step(tt, Usol[1, :], where="post", label="uy")
    axs[2].set_ylabel("u"); axs[2].set_xlabel("time [s]")
    axs[2].grid(True, ls="--", alpha=0.5); axs[2].legend(loc="best")

    plt.tight_layout()
    return Xsol, Usol, J, (fig1, ax1), (fig2, axs)
