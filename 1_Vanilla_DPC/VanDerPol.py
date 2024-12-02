#!/usr/bin/env python
# coding: utf-8

# ## Install (Colab only)
# Skip this step when running locally.

# In[1]:


#!pip install neuromancer


# *Note: When running on Colab, one might encounter a pip dependency error with Lida 0.0.10. This can be ignored*

# In[2]:


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
torch.manual_seed(0)


# In[3]:


# ground truth system model
gt_model = psl.nonautonomous.VanDerPolControl()
# sampling rate
ts = gt_model.params[1]['ts']
# problem dimensions
nx = gt_model.nx    # number of states
nu = gt_model.nu    # number of control inputs
nref = nx           # number of references
# constraints bounds
umin = -5.
umax = 5.
xmin = -4.
xmax = 4.

# white-box ODE model with no-plant model mismatch
van_der_pol = ode.VanDerPolControl()                   # ODE system equations implemented in PyTorch
van_der_pol.mu = nn.Parameter(torch.tensor(gt_model.mu), requires_grad=False)

# integrate continuous time ODE
integrator = integrators.RK4(van_der_pol, h=torch.tensor(ts))   # using 4th order runge kutta integrator
# symbolic system model
integrator_node = Node(integrator, ['xn', 'u'], ['xn'], name='model')




# # Differentiable Predictive Control
#
# Next we show how to solve the corresponding parametric optimal control using the [DPC method](https://www.sciencedirect.com/science/article/pii/S0959152422000981) implemented in Neuromancer.
#
# **Schematics of the Differentiable Predictive Control method**:
# <img src="./figs/DPC_simple_method.png" width="600">
#
# **Neural control policy**:
# The objective of this tutorial is to learn neural control policy $u_k = \pi(x_k, R)$ to control the tank levels by modulating the pump and valve control actions $u_k = [p_k, v_k]$. The policy takes in the measurements of system states $x_k$ at thime $k$, prediciton of desired references $R = [r_k, ..., r_{k+N}]$ over pre-defined horizon $N$.
#
# **Differentiable system model**:
# The DPC is a model-based policy optimization algorithm, that exploits the differentiability of a wide class of model representations for dynamical systems, including differential equations, state-space models, or various neural network architectures. In this example, we compactly represent the system model by ODE equations  $\text{ODESolve}(\theta(x^i_k, u^i_k)\xi_k)$  describing the governing dynamics of the controlled system.
#
# **Differentiable predictive control problem formulation**:
# We learn the explicit neural control policy by solving the following parametric optimal control problem:
# $$
# \begin{align}
# &\underset{\theta}{\text{minimize}}     && \sum_{i=1}^m  \Big( \sum_{k=1}^{N-1} Q_x||x^i_k - r^i_k||_2^2  + Q_N||x^i_N - r^i_N||_2^2 \Big) \\
# &\text{subject to}    && x^i_{k+1} =  \text{ODESolve}(\theta(x^i_k, u^i_k)\xi_k) \\
# &                     && u^i_k = \pi_{\theta}(x^i_k, R^i) \\
# &                     && 0 \le x^i_k \le 1 \\
# &                     && 0 \le u^i_k \le 1 \\
# &                     && x^i_0 \sim \mathcal{P}_{x_0} \\
# &                     && R^i \sim  \mathcal{P}_R
# \end{align}
# $$
# The objective function is to minimize the reference tracking error $||x^i_k - r^i_k||_2^2$ over pre-defined prediction horizon $N$ weighted by a scalar $Q_x$, including terminal penalty weighted by $Q_N$.  The parametric neural control policy is given by $\pi_{\theta}(x^i_k, R^i)$. The neural control policy is optimized over a problem parameters sampled from the distributions $\mathcal{P}_{x_0}$, and $\mathcal{P}_R$, for state initial conditions, and references, respectively. The parameters $\theta$ are optimized with stochastic gradient descent.

# ## Training dataset generation
#
# For a training dataset we randomly sample initial conditions of states and sequence of admissible reference trajectories over predefined prediction horizon from given distributions $\mathcal{P}_{x_0}$, and $\mathcal{P}_R$, respectively.

# In[14]:


def get_policy_data(nsteps, n_samples):
    # Training dataset generation
    train_data = DictDataset({'xn': torch.randn(n_samples, 1, nx),
                          'r': torch.zeros(n_samples, nsteps+1, nx)}, name='train')
    # Development dataset generation
    dev_data = DictDataset({'xn': torch.randn(n_samples, 1, nx),
                        'r': torch.zeros(n_samples, nsteps+1, nx)}, name='dev')
    # torch dataloaders
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                         collate_fn=dev_data.collate_fn, shuffle=False)
    return train_loader, dev_loader

nsteps = 50  # prediction horizon
n_samples = 2000    # number of sampled scenarios

train_loader, dev_loader = get_policy_data(nsteps, n_samples)


# ## System model and Control policy in Neuromancer
#
# Here we construct a closed-loop system as differentiable computational graph by coinnecting the system dynamics model  $x_{k+1} = \text{ODESolve}(\theta(x_k, u_k)\xi_k)$ with neural control policy $u_k = \pi_{\theta}(x_k, R)$. Hence we obtain a trainable system architecture:
# $x_{k+1} = \text{ODESolve}(\theta(x_k, \pi_{\theta}(x_k, R))\xi_k)$. In this example our system dynamics model is our learned SINDy model.

# In[15]:


# symbolic system model
model = Node(integrator, ['xn', 'u'], ['x'], name='model')

# neural net control policy with hard control action bounds
net = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                    nonlin=activations['gelu'], min=umin, max=umax)
policy = Node(net, ['xn', 'r'], ['u'], name='policy')

# closed-loop system model
cl_system = System([policy, integrator_node], nsteps=nsteps)
# cl_system.show()


# ## Differentiable Predictive Control objectives and constraints
#
# Here we take advantage of Neuromancer's high level symbolic language to define objective and constraint terms of our optimal control problem.

# In[16]:


# variables
x = variable('xn')
ref = variable('r')
# objectives
regulation_loss = 5. * ((x == ref) ^ 2)  # target posistion
# constraints
state_lower_bound_penalty = 10.*(x > xmin)
state_upper_bound_penalty = 10.*(x < xmax)

# objectives and constraints names for nicer plot
regulation_loss.name = 'state_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'

# list of constraints and objectives
objectives = [regulation_loss]
constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty,
]


# ## Differentiable optimal control problem
#
# Here we put things together to construct a differentibale optimal control problem.

# In[17]:


# data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
components = [cl_system]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(components, loss)
# plot computational graph
# problem.show()


# ## Solve the problem
#
# We solve the problem using stochastic gradient descent over pre-defined training data of sampled parameters.

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


# # Evaluate best model on a system rollout
#
# Here we generate 5 different reference system states over 750 timesteps to see how our control policy will act to stabilize the system to these reference states.

# In[20]:


nsteps_test = 100
torch.manual_seed(22)
data = {'xn': torch.randn(1, 1, nx, dtype=torch.float32),
        'r': torch.zeros(1, nsteps_test+1, nx, dtype=torch.float32)}
cl_system.nsteps = nsteps_test

import time
nn_times = []
trajectories = None
for i in range(50):
    start = time.time()
    # perform closed-loop simulation
    trajectories = cl_system(data)
    total = time.time() - start
    nn_times.append(total)

# ## Dictionary Policy
#
# Additionally, we can learn a sparse dictionary policy using the SINDy object. So now our neural control policy $u_k = \pi_{\theta}(x_k, R)$ can be represented by a sparse dictionary regression model instead of a neural network. As such we have $u_k = \theta(x_k, R)\xi_k$, where $\theta$ is our library and $\xi_k$ is our learned coefficient. Remember, we have the constraint for control inputs: $0 \le u^i_k \le 1$. So, to add this constraint, we use the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to ensure that all control inputs generated by our policy meet this constraint.

# In[21]:


nsteps_init = 50
n_iter = 1
max_degree = 2
max_freq = 2
lambda_l = 1e-7
p = 1

nsteps = nsteps_init  # prediction horizon
n_samples = 2000    # number of sampled scenarios

theta_1 = library.PolynomialLibrary(nx, nref, max_degree=max_degree)
theta_2 = library.FourierLibrary(nx, nref, max_freq=max_freq)

policy_sindy = sindy.SINDy(library.CombinedLibrary([theta_1, theta_2]), n_out=nu)
policy_node = Node(policy_sindy,  ['xn', 'r'], ['u_unbound'], name="policy")
u_bound = Node(lambda u: torch.clamp(u, umin, umax), ['u_unbound'], ['u'])

sindy_system = System([policy_node, u_bound, integrator_node], nsteps=nsteps)

hyperparameters = [nsteps, p]
initial_values = [10, 1]
stopping_conditions = [lambda nsteps: nsteps >= 15, lambda p: p > 1]
update_rules = [lambda nsteps: nsteps * 2, lambda p: p + 1]


# In[22]:


# variables
x = variable('xn')
ref = variable('r')
u = variable('u')
l1 = variable([x], lambda x: torch.norm(list(policy_sindy.parameters())[0], 1))

loss_l1 = lambda_l*(l1 == 0)

# objectives
regulation_loss = 5.*((x == ref)^2)  # target posistion
state_lower_bound_penalty = 2.5*(x > xmin)
state_upper_bound_penalty = 2.5*(x < xmax)
terminal_lower_bound_penalty = 2.5*(x[:, [-1], :] > ref-0.01)
terminal_upper_bound_penalty = 2.5*(x[:, [-1], :] < ref+0.01)

# objectives and constraints names for nicer plot
regulation_loss.name = 'state_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'
terminal_lower_bound_penalty.name = 'y_N_min'
terminal_upper_bound_penalty.name = 'y_N_max'
# list of constraints and objectives
objectives = [regulation_loss, loss_l1]
constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty,
]


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


# ## Model Training
#
# Here, we draw on the idea of [cirriculum learning](https://en.wikipedia.org/wiki/Curriculum_learning) to gradually increase the number of rollout steps of our model. So, we start with by learning the sparse dictionary policy for how to control the system for the next two timesteps. As we progress through training, we increase the number of timesteps we learn for which we learn the policy. This progressive type of training allows us to learn the policy while gradually making the learning task more complex.

# In[25]:


train_loader, dev_loader = get_policy_data(nsteps, n_samples)
trainer.train_data, trainer.dev_data = train_loader, dev_loader

for hyper, stop, update, value in zip(hyperparameters, stopping_conditions, update_rules, initial_values):
    hyperparameter = hyper
    stopping_condition = stop
    update_rule = update


    while not stopping_condition(hyperparameter):
        print(f"Next Iteration: {hyperparameter}")
        trainer.problem = problem

        # Train control policy
        best_model = trainer.train()

        # load best trained model
        trainer.model.load_state_dict(best_model)

        hyperparameter = update_rule(hyperparameter)
        sindy_system.nsteps = nsteps
        trainer.badcount = 0

    hyperparameter = value


# ## Compare Dictionary Policy to Neural Policy
# Again, we rollout both of our models to 5 desired reference states over 750 timestep to compare the two policies. As we can see, our dictionary policy does a slightly better job of controlling our system than the neural policy.

# In[26]:


cl_system.nsteps = nsteps_test
sindy_system.nsteps = nsteps_test
# perform closed-loop simulation
trajectories_nn = cl_system(data)

sindy_times = []
for i in range(50):
    start = time.time()
    trajectories_sindy = sindy_system(data)
    end = time.time() - start
    sindy_times.append(end)


# In[27]:


nn_loss = torch.nn.functional.mse_loss(trajectories_nn['xn'], data['r'])
sindy_loss = torch.nn.functional.mse_loss(trajectories_sindy['xn'], data['r'])


# In[ ]:


import pickle


traj = dict()
traj['nn'] = trajectories_nn
traj['nn']['time'] = nn_times
traj['sindy'] = trajectories_sindy
traj['sindy']['time'] = sindy_times
with open("vdp-wb.pyc", 'wb') as outfile:
    pickle.dump(traj, outfile)

