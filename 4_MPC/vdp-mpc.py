from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import time
from neuromancer.psl.nonautonomous import VanDerPolControl

nsim = 100

# seed for initial conditions
sys = VanDerPolControl()
mu = sys.mu
umin = -5.
umax = 5.
xmin = -4.
xmax = 4.

def ode_equations(x, u):
    dx1 = x[1]
    dx2 = mu*(1 - x[0]**2)*x[1] - x[0] + u
    return casadi.vertcat(dx1, dx2)

# instantiate casadi optimizaiton problem class
opti = casadi.Opti()

N = 50
dt = 0.1
X = opti.variable(sys.nx, N+1)  # state trajectory
U = opti.variable(N)    # control trajectory
x_param = opti.parameter(sys.nx)

# sample initial conditions

opti.subject_to(opti.bounded(umin, U, umax))
opti.subject_to(opti.bounded(xmin, X, xmax))
opti.subject_to(X[:, 0] == x_param)

cost = 0
for i in range(N):
    opti.subject_to(X[:, i+1] == X[:, i] + ode_equations(X[:, i], U[i])*dt)
    cost = cost + sumsqr(X[:, i])
opti.minimize(cost)
opti.solver('ipopt')
# select IPOPT solver and solve the NLP

times = []
Us = []
import torch
torch.manual_seed(22)


x0 = torch.randn(sys.nx, dtype=torch.float32).numpy()
Xs = [x0]
for k in range(nsim):
    opti.set_value(x_param, x0)
    start_time = time.time()
    sol = opti.solve()
    sol_time = time.time() - start_time
    times.append(sol_time)
    x0 += ode_equations(x0, sol.value(U)[0])*dt
    Xs.append(x0.full().flatten())
    Us.append(sol.value(U)[0])

Xs = np.asarray(Xs)
Us = np.asarray(Us)

print(f'mean solution time: {np.mean(times)}')
print(f'max solution time: {np.max(times)}')

fig, ax = plt.subplots(1, 2, figsize=(9,4))
ax[0].plot(Xs)
ax[0].set_ylabel("$x$")
ax[0].set_xlabel("time")
ax[1].set_ylabel("$u$")
ax[1].set_xlabel("time")
ax[1].plot(Us)
plt.savefig("vdp-mpc.png")
plt.show()

import pickle

traj = dict()
traj['xn'] = Xs
traj['u'] = Us
traj['times'] = times
with open('vdp-mpc.pyc', 'ab') as outfile:
    pickle.dump(traj, outfile)
