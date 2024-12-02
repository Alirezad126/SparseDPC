"""
linear-quadratic MPC
example from: https://osqp.org/docs/examples/mpc.html
"""

import cvxpy as cp
from cvxpy import *
import numpy as np
import scipy as sp

from scipy import sparse
from pylab import *
import time
import neuromancer.psl as psl

# get ICs from nonlinear version of the system - linear version doesn't
# provide realistic initial conditions

# seed for consistency in initial conditions
nonlin = psl.systems["Reno_ROM40"](seed=59)

nsim = 1000
system_name = 'LinearReno_ROM40'
sys = psl.systems[system_name]()

umin = sys.umin
umax = sys.umax
nx = sys.nx
nu = sys.nu
nd = sys.nD
ny = sys.ny

# Discrete time model of a quadcopter
A = sparse.csc_matrix(sys.A)
B = sparse.csc_matrix(sys.Beta)
C = sparse.csc_matrix(sys.C)
E = sparse.csc_matrix(sys.E)
F = sys.F.flatten()
G = sys.G.flatten()
y_ss = sys.y_ss.flatten()

def state_model(x, u, d):
    x = A @ x + B @ u + E @ d + G
    return x

def output_model(x):
    y = C @ x + F - y_ss
    return y

# Constraints
# Objective function
N = 20
x0 = nonlin.get_x0()
dist = nonlin.get_D(nsim+N+1)
y0 = output_model(x0)

Q = 50.*sparse.eye(ny)
QN = Q
R_weight = 0
R = R_weight * sparse.eye(nu)



# Initial and reference states


# Prediction horizon

# Define problem
u = Variable((nu, N))
x = Variable((nx, N + 1))
y = Variable((ny, N+1))


d = Parameter((nd, N))
slack_lower = Variable((ny))
slack_upper = Variable((ny))
#slack_u = Variable((nu))
x_init = Parameter(nx)
y_init = Parameter(ny)
ymin = Parameter((ny, N))
ymax = Parameter((ny, N))
objective = 0
constraints = [x[:, 0] == x_init]
constraints += [y[:, 0] == y_init]
constraints += [slack_lower >= 0]
constraints += [slack_upper >= 0]
for k in range(N):
    constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:,k] + E @ d[:,k] + G]
    constraints += [y[:,k] == C @ x[:, k] + F - y_ss]
    constraints += [y[:, k] >= ymin[:,k] - slack_lower]
    constraints += [y[:, k] <= ymax[:,k] + slack_upper]
    constraints += [u[:,k] <= umax]
    constraints += [u[:,k] >= umin]
    objective += quad_form(u[:,k],R)
objective += quad_form(slack_upper, QN) + quad_form(slack_lower, QN)

prob = Problem(Minimize(objective), constraints)

# Simulate in closed loop
Y = [y0]
U = []
D = []
times = []
ymin_vals = psl.signals.step(nsim+N+1, ny, min=18., max=22., randsteps=3,
        rng=np.random.default_rng(seed=0))
for i in range(nsim):
    print(f'Step: {i}')
    x_init.value = x0
    y_init.value = y0
    end = i + N

    d.value = dist[i:end, :].T
    ymin.value = ymin_vals[i:end, :].T
    ymax.value = ymin.value+2.
    start_time = time.time()
    prob.solve(solver=SCS, warm_start=True, verbose=False,)
    sol_time = time.time() - start_time
    times.append(sol_time)
    x0 = A @ x0 + B @ u[:,0].value + E @ dist[i,:] + G
    y0 = C @ x0 + F - y_ss
    U.append(u[:,0].value)
    D.append(dist[i,:])
    Y.append(y0)

Ynp = np.asarray(Y)
Unp = np.asarray(U)
mean_sol_time = np.mean(times)
max_sol_time = np.max(times)
print(f'mean sol time {mean_sol_time}')
print(f'max sol time {max_sol_time}')

u_min = umin*np.ones([nsim+1, umin.shape[0]])
u_max = umax*np.ones([nsim+1, umax.shape[0]])

traj = dict()
traj["ymin"] = ymin_vals
traj["ymax"] = ymin_vals+2.0
traj["umin"] = u_min
traj["umax"] = u_max
traj['y'] = Ynp
traj['u'] = Unp
traj['times'] = times

import pickle
with open(f"reno-mpc.pyc", "wb") as file:
    pickle.dump(traj, file)


