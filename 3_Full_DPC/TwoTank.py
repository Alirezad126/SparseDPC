#!/usr/bin/env python
# coding: utf-8

# ## Install (Colab only)
# Skip this step when running locally.

# In[1]:


#!pip install neuromancer


# *Note: When running on Colab, one might encounter a pip dependency error with Lida 0.0.10. This can be ignored*

# In[3]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import library
import sindy

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import ode, integrators
from neuromancer.plot import pltCL, pltPhase
torch.manual_seed(0)


# # Learning to Control an Energy Storage System in NeuroMANCER
#
# This tutorial demonstrates the use of [Differentiable predictive control (DPC)](https://www.sciencedirect.com/science/article/pii/S0959152422000981) method to learn constrained neural policy for controlling [pumped-storage hydroelectricity](https://en.wikipedia.org/wiki/Pumped-storage_hydroelectricity) system described by [nonlinear ordinary differential equations (ODE)](https://en.wikipedia.org/wiki/Ordinary_differential_equation).
#
#
# ## Pumped-storage Hyrdoelectricity System
#
# Lets consider a [pumped-storage hydroelectricity](https://en.wikipedia.org/wiki/Pumped-storage_hydroelectricity) (PSH) system which is a type of hydroelectric energy storage used by electric power systems for load balancing.
# [Load balancing](https://en.wikipedia.org/wiki/Load_balancing_(electrical_power)) or daily peak demand reserve refers to the use of various techniques by electrical power stations to store excess electrical power during low demand periods for release as demand rises. These techniques are an important part of modern power system that help to balance the time-varying load with the generation.
# As of 2020, the largest form of [grid energy storage]((https://en.wikipedia.org/wiki/Grid_energy_storage)) is dammed hydroelectricity, with both conventional hydroelectric generation as well as pumped-storage hydroelectricity.
#
# <img src="./figs/PSH.PNG" width="600">
#
# image adopted from: https://www.upsbatterycenter.com/blog/pumped-storage-hydroelectricity/
#
#
# **System schematics**:
# <img src="../figs/two_tank_level.png" width="250">
#
# **System model**:
# A simplified system dynamics of PSH system is defined by following nonlinear ordinary differential equations (ODEs):
# $$
#  \frac{dx_1}{dt} = c_1 (1.0 - v)  p - c_2  \sqrt{x_1}
#  $$
#
#  $$
#  \frac{dx_2}{dt}  = c_1 v p + c_2  \sqrt{x_1} - c_2 \sqrt{x_2}
# $$
# With system states $x_1$, and $x_2$ representing liquid levels in tank 1 and 2, respectively. Control actions are pump modulation $p$, and valve opening $v$. The ODE system is parametrized by inlet and outlet valve coefficients $c_1$ and $c_2$, respectively.
#
# System model and image adopted from: https://apmonitor.com/do/index.php/Main/LevelControlex.php/Main/LevelControl
#
#
# Main/LevelControl

# In[4]:


# ground truth system model
gt_model = psl.nonautonomous.TwoTank()
# sampling rate
ts = gt_model.params[1]['ts']
# problem dimensions
nx = gt_model.nx    # number of states
nu = gt_model.nu    # number of control inputs
nref = nx           # number of references
# constraints bounds
umin = 0.
umax = 1.
xmin = 0.
xmax = 1


# ## SINDy System Identification
#
# First, we will construct the SINDy system identification problem to learn our ODE. For a more in depth example of using SINDy for system identification in Neuromancer, go [here](https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/ODEs/Part_9_SINDy.ipynb).

# In[5]:


def get_data(sys, nsim, nsteps, ts, bs):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size
    :param bs: (int) batch size

    """
    train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=ts) for i in range(3)]
    nx = sys.nx
    nu = sys.nu
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    trainX = train_sim['X'][:length].reshape(nbatch, nsteps, nx)
    trainU = train_sim['U'][:length].reshape(nbatch, nsteps, nu)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainU = torch.tensor(trainU, dtype=torch.float32)
    train_data = DictDataset({'r': trainX, 'xn': trainX[:, 0:1, :], 'u': trainU}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)

    devX = dev_sim['X'][:length].reshape(nbatch, nsteps, nx)
    devU = dev_sim['U'][:length].reshape(nbatch, nsteps, nu)
    devX = torch.tensor(devX, dtype=torch.float32)
    devU = torch.tensor(devU, dtype=torch.float32)
    dev_data = DictDataset({'r': devX, 'xn': devX[:, 0:1, :], 'u': devU}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=True)

    testX = test_sim['X'][:length].reshape(1, nsim, nx)
    testU = test_sim['U'][:length].reshape(1, nsim, nu)
    testX = torch.tensor(testX, dtype=torch.float32)
    testU = torch.tensor(testU, dtype=torch.float32)
    test_data = {'r': testX, 'xn': testX[:, 0:1, :], 'u': testU}

    return train_loader, dev_loader, test_data


# In[6]:


nsim = 1000   # number of simulation steps in the dataset
nsteps = 10   # number of prediction horizon steps in the loss function
bs = 50     # minibatching batch size
train_loader, dev_loader, test_data = get_data(gt_model, nsim, nsteps, ts, bs)


# ## Construct SINDy Model
#
# Here we construct a SINDy model $\dot{x_k} = \theta(x_k, u_k)\xi_k$, which uses aims to fit the correct coefficient values $\xi_k$ to properly predict the change in $x_k$ given the state and the input. In the construction of our model, we add bounds on the states, which is based on prior knowledge we have about the two tank system.

# In[7]:


theta_1 = library.PolynomialLibrary(nx, nu, 2)

theta_2_funs = [lambda X, u: torch.sqrt(X[:,0]), lambda X, u: torch.sqrt(X[:,1])]
theta_2_names = ["sqrt(x_0)", "sqrt(x_1)"]
theta_2 = library.FunctionLibrary(theta_2_funs, nx, nu, theta_2_names)


fx = sindy.SINDy(library.CombinedLibrary([theta_1, theta_2]))


# In[8]:


integrator = integrators.Euler(fx, h=ts)
integrator_node = Node(integrator, ['xn', 'u'], ['x'])
x_bound = Node(lambda x: torch.clamp(x, xmin, xmax), ['x'], ['xn'])
dynamics_model = System([integrator_node, x_bound], nsteps=nsteps, name="dynamics_model")


# ## Define system identification loss function terms
#

# In[9]:


# %% Constraints + losses:
x = variable("r")
xhat = variable('xn')[:, :-1, :]

# one step loss
onestep_loss = 1.*(xhat[:, 1, :] == x[:, 1, :])^2
onestep_loss.name = "onestep_loss"

# reference tracking loss
reference_loss = ((xhat == x)^2)
reference_loss.name = "ref_loss"

#sparsity
l1 = variable([x], lambda x: torch.norm(list(fx.parameters())[0]))
loss_l1 = 0.000001*(l1 == 0)


# ## Construct System ID learning problem

# In[10]:


# aggregate list of objective terms and constraints
objectives = [reference_loss, onestep_loss, loss_l1]

# create constrained optimization loss
loss = PenaltyLoss(objectives, [])
# construct constrained optimization problem
problem = Problem([dynamics_model], loss)


# In[11]:


optimizer = torch.optim.AdamW(problem.parameters(),
                             lr=0.01)
# trainer
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_data,
    optimizer,
    patience=200,
    epochs=1000,
    epoch_verbose=200,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
)


# In[12]:


# %% train
best_model = trainer.train()
problem.load_state_dict(best_model)


# In[26]:


def get_policy_data(nsteps, n_samples):
    #  sampled references for training the policy
    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([n_samples, nsteps+1, nref])
    # Training dataset
    train_data = DictDataset({'xn': torch.rand(n_samples, 1, nx),   # sampled initial conditions of states
                              'r': batched_ref}, name='train')

    # sampled references for development set
    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([n_samples, nsteps+1, nref])
    # Development dataset
    dev_data = DictDataset({'xn': torch.rand(n_samples, 1, nx),    # sampled initial conditions of states
                            'r': batched_ref}, name='dev')

    # torch dataloaders
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=train_data.collate_fn,
                                               shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                             collate_fn=dev_data.collate_fn,
                                             shuffle=False)
    return train_loader, dev_loader

nsteps = 50  # prediction horizon
n_samples = 3000    # number of sampled scenarios

train_loader, dev_loader = get_policy_data(nsteps, n_samples)


# In[21]:


# symbolic system model
model = Node(integrator, ['xn', 'u'], ['x'], name='model')

# neural net control policy with hard control action bounds
net = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                    nonlin=activations['gelu'], min=umin, max=umax)
policy = Node(net, ['xn', 'r'], ['u'], name='policy')

# closed-loop system model
cl_system = System([policy, model, x_bound], nsteps=nsteps)
# cl_system.show()


# In[18]:


# variables
x = variable('xn')
ref = variable('r')
# objectives
regulation_loss = 5. * ((x == ref) ^ 2)  # target posistion
# constraints
state_lower_bound_penalty = 10.*(x > xmin)
state_upper_bound_penalty = 10.*(x < xmax)
terminal_lower_bound_penalty = 10.*(x[:, [-1], :] > ref-0.01)
terminal_upper_bound_penalty = 10.*(x[:, [-1], :] < ref+0.01)
# objectives and constraints names for nicer plot
regulation_loss.name = 'state_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'
terminal_lower_bound_penalty.name = 'y_N_min'
terminal_upper_bound_penalty.name = 'y_N_max'
# list of constraints and objectives
objectives = [regulation_loss]
constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty,
    terminal_lower_bound_penalty,
    terminal_upper_bound_penalty,
]


# In[17]:


# data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
components = [cl_system]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(components, loss)
# plot computational graph
# problem.show()


# In[18]:


optimizer = torch.optim.AdamW(policy.parameters(), lr=0.002)
#  Neuromancer trainer
trainer = Trainer(
    problem,
    train_loader, dev_loader,
    optimizer=optimizer,
    epochs=100,
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=50,
)
# Train control policy
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)


# In[28]:


torch.manual_seed(0)
nsteps_init = 10
n_iter = 1
max_freq = 5
max_degree = 2
nsteps = nsteps_init  # prediction horizon
n_samples = 2000    # number of sampled scenarios

theta_1 = library.FourierLibrary(nx, nref, max_freq=max_freq)
theta_2 = library.PolynomialLibrary(nx, nref, max_degree=max_degree)
policy_sindy = sindy.SINDy(library.CombinedLibrary([theta_1, theta_2]))
policy_node = Node(policy_sindy,  ['xn', 'r'], ['u_unbound'])
integrator_node = Node(integrator, ['xn', 'u'], ['x'])
u_bound = Node(lambda u: torch.sigmoid(u) * (umax-umin) + umin, ['u_unbound'], ['u'])

sindy_system = System([policy_node, u_bound, model, x_bound], nsteps=nsteps)


# In[22]:


# variables
x = variable('xn')
ref = variable('r')
u = variable('u')

# objectives
regulation_loss = 15.*((x == ref)^2)  # target posistion

# list of constraints and objectives
objectives = [regulation_loss]
constraints = [terminal_lower_bound_penalty, terminal_upper_bound_penalty]

# In[23]:


# data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
components = [sindy_system]
# create constrained optimization loss
loss = PenaltyLoss(objectives, [])
# construct constrained optimization problem
problem = Problem(components, loss)
# plot computational graph
# problem.show()


# In[24]:


optimizer = torch.optim.AdamW(policy_node.parameters(), lr=0.01)
#  Neuromancer trainer
trainer = Trainer(
    problem,
    None,
    None,
    optimizer=optimizer,
    epochs=250,
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=200,
)


# In[29]:


for _ in range(n_iter):
    train_loader, dev_loader = get_policy_data(nsteps, n_samples)
    trainer.train_data, trainer.dev_data = train_loader, dev_loader
    trainer.problem = problem

    # Train control policy
    best_model = trainer.train()

    # load best trained model
    trainer.model.load_state_dict(best_model)

    nsteps *= 2
    sindy_system.nsteps = nsteps
    trainer.badcount = 0


# In[ ]:


nsteps = 300
step_length = 100

# generate reference
np_refs = psl.signals.step(nsteps+1, 1, min=xmin, max=xmax, randsteps=3, rng=np.random.default_rng(seed=20))
R = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps+1, 1)
torch_ref = torch.cat([R, R], dim=-1)

# generate initial data for closed loop simulation
data = {'xn': torch.rand(1, 1, nx, dtype=torch.float32),
        'r': torch_ref}
cl_system.nsteps = nsteps
sindy_system.nsteps = nsteps
print(data['r'].shape)

nn_times = []
sindy_times = []
trajectories_nn = None
trajectories_sindy = None
import time
for i in range(50):
    start = time.time()
    # perform closed-loop simulation
    trajectories_nn = cl_system(data)
    end = time.time() - start
    nn_times.append(end)

    start_s = time.time()
    trajectories_sindy = sindy_system(data)
    end_s = time.time() - start
    sindy_times.append(end_s)

import pickle
traj = dict()
traj['nn'] = trajectories_nn
traj['nn']['times'] = nn_times
traj['sindy'] = trajectories_sindy
traj['sindy']['times'] = sindy_times

with open("tt-bb.pyc", 'wb') as outfile:
    pickle.dump(traj, outfile)
