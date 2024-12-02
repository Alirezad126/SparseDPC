from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import time
from neuromancer.psl.nonautonomous import TwoTank
import neuromancer.psl as psl

# seed for consistency in initial conditions
sys = TwoTank(seed=9)
umin = 0.
umax = 1.
xmin = 0.
xmax = 1.

c1 = sys.c1
c2 = sys.c2

p = 1e-5
# add for numerical issues with sqrt

def ode_equations(x, u):
    # clipping
    h1 = fmax(xmin, fmin(x[0], xmax))
    h2 = fmax(xmin, fmin(x[1], xmax))
    pump = fmax(umin, fmin(u[0], umax))
    valve = fmax(umin, fmin(u[1], umax))

    # ode equations
    dhdt1 = c1 * (1.0 - valve) * pump - c2 * sqrt(h1+p)
    dhdt2 = c1 * valve * pump + c2 * sqrt(h1+p) - c2 * sqrt(h2+p)

    # replace bounds logic
    dhdt1 = if_else(logic_and(h1 >= 1.0, dhdt1 > 0.0), 0, dhdt1)
    dhdt2 = if_else(logic_and(h2 >= 1.0, dhdt2 > 0.0), 0, dhdt2)
    dhdt = casadi.vertcat(dhdt1, dhdt2)
    return dhdt


# instantiate casadi optimizaiton problem class
opti = casadi.Opti()
nsim = 300
N = 2
dt = sys.ts
X = opti.variable(sys.nx, N+1)  # state trajectory
U = opti.variable(sys.nu, N)    # control trajectory
x0_param = opti.parameter(sys.nx)
ref_param = opti.parameter(sys.nx, N+1)
# sample initial conditions

# seed for consistency in reference trajectories

opti.subject_to(opti.bounded(umin, U, umax))
opti.subject_to(opti.bounded(xmin, X, xmax))


opti.subject_to(X[:, 0] == x0_param)
cost = 0
for i in range(N):
    cost = cost + sumsqr(X[:, i] - ref_param[:,i])
    opti.subject_to(X[:, i+1] == X[:, i] + ode_equations(X[:, i], U[:, i])*dt)
cost = cost + sumsqr(X - ref_param[:,N])

opti.minimize(cost)

times = []
Us = []
np_refs = psl.signals.step(nsim+1+N, 1, min=xmin, max=xmax, randsteps=3,
        rng=np.random.default_rng(20))
x0 = sys.get_x0()
x0_param.value = x0
Xs = [x0]
np_refs = np.hstack([np_refs, np_refs])
for k in range(nsim):
    opti.set_value(x0_param, x0)
    opti.set_value(ref_param, np_refs[k:k+N+1,:].T)
    start_time = time.time()
    opti.solver('ipopt')
    sol = opti.solve()
    sol_time = time.time() - start_time
    u_val = sol.value(U)[:,0]
    x0 += (ode_equations(x0, u_val)*dt)
    Xs.append(x0.full().flatten())
    Us.append(u_val)
    times.append(sol_time)
print(f'mean solution time: {np.mean(times)}')
print(f'max solution time: {np.max(times)}')


Xs = np.asarray(Xs)
Us = np.asarray(Us)
xup = xmax*np.ones([nsim+1, 1])
xlow = xmin*np.ones([nsim+1, 1])
u_min = umin*np.ones(nsim+1)
u_max = umax*np.ones(nsim+1)
fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].plot(Xs, label=f'y{i}', linewidth=2)
ax[0].plot(xup, 'k--', label='r', linewidth=2)
ax[0].plot(xlow, 'k--', label='r', linewidth=2)
ax[0].plot(np_refs[:nsim+1, :], 'k--', color="red")
ax[0].set(ylabel='$x$')
ax[0].set(xlabel='time')
ax[0].grid()
ax[0].set_xlim(0, nsim)

ax[1].plot(Us, label='u', drawstyle='steps', linewidth=2)
ax[1].plot(u_min, 'k--', label='r', linewidth=2)
ax[1].plot(u_max, 'k--', label='r', linewidth=2)
ax[1].set(ylabel='$u$')
ax[1].set(xlabel='time')
ax[1].grid()
ax[1].set_xlim(0, nsim)
#plt.tight_layout()
plt.savefig("tt-mpc.png")
plt.show()

traj = dict()
traj['xn'] = Xs
traj['u'] = Us
traj['refs'] = np_refs
traj['xmin'] = xup
traj['xmax'] = xlow
traj['umin'] = u_min
traj["umax"] = u_max
traj['times'] = times
import pickle
with open('tt-mpc.pyc', 'ab') as outfile:
    pickle.dump(traj, outfile)
