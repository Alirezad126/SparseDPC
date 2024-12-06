#!/usr/bin/env python
# coding: utf-8

# ## Install (Colab only)
# Skip this step when running locally.

# In[ ]:


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
from neuromancer.plot import pltCL, pltPhase
torch.manual_seed(22)


# In[3]:


# ground truth system model
gt_model = psl.nonautonomous.TwoTank(seed=9)
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
xmax = 1.
sys = gt_model

# In[4]:
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


# In[5]:

# white-box ODE model with no-plant model mismatch
two_tank = ode.TwoTankParam()                   # ODE system equations implemented in PyTorch
two_tank.c1 = nn.Parameter(torch.tensor(gt_model.c1), requires_grad=False)
two_tank.c2 = nn.Parameter(torch.tensor(gt_model.c2), requires_grad=False)
# integrate continuous time ODE
integrator = integrators.RK4(two_tank, h=torch.tensor(ts))   # using 4th order runge kutta integrator
# symbolic system model
integrator_node = Node(integrator, ['xn', 'u'], ['xn'], name='model')
# In[6]:


# In[19]:
nsteps = 750
step_length = 150

nsteps_test=100
# generate reference
np_refs = psl.signals.step(nsteps_test+1, 1, min=xmin, max=xmax, randsteps=5, rng=np.random.default_rng(20))
R = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps+1, 1)
torch_ref = torch.cat([R, R], dim=-1)

# generate initial data for closed loop simulation
data = {'xn': torch.tensor(sys.get_x0()).reshape(1,1,nx),
        'r': torch_ref}

# In[7]:


from ray import train, tune


# In[8]:


# variables
x = variable('xn')
ref = variable('r')
u = variable('u')

# objectives
regulation_loss = 5.*((x == ref)^2)  # target posistion
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

constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty,
    ]


# In[9]:


def train_func(optim, system, config):
    nsteps = system.nsteps

    l1 = variable([x], lambda x: torch.norm(list(system.parameters())[0]))
    loss_l1 = config["lambda"]*(l1 == 0)    
    objectives = [regulation_loss, loss_l1]
    
    components = [system]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    trainer = Trainer(
        problem,
        None,
        None,
        optimizer=optim,
        epochs=500,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=200,
        epoch_verbose=200
    )
    for _ in range(config['n_iter']):
        train_loader, dev_loader = get_policy_data(nsteps, n_samples)

        trainer.train_data, trainer.dev_data = train_loader, dev_loader 
        trainer.problem = problem
        # Train control policy
        best_model = trainer.train()
    
        # load best trained model
        trainer.model.load_state_dict(best_model)

        nsteps *= 2
        system.nsteps = nsteps
        trainer.badcount = 0

def test(system, test_data):
    nsteps = 100 
    system.nsteps = nsteps

    # perform closed-loop simulation
    trajectories_sindy = system(test_data)

    return torch.nn.functional.mse_loss(trajectories_sindy['xn'], data['r']).item()


# In[10]:


def update_weights(coef, q, epsilon):
    return torch.diag(1 / (coef.flatten()**q + epsilon))

def train_func_weighted(optim, system, config):
    nsteps = system.nsteps

    weight_matrix = torch.eye(list(system.parameters())[0].flatten().shape[0])
    weighted_l1 = variable([x], lambda x: torch.norm(torch.matmul(weight_matrix, list(system.parameters())[0].flatten())))
    loss_l1 = config["lambda"]*((weighted_l1 == 0)) 
    
    # list of constraints and objectives
    objectives = [regulation_loss, loss_l1]
    components = [system]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    # problem.show()
     #  Neuromancer trainer
    trainer = Trainer(
        problem,
        None,
        None,
        optimizer=optim,
        epochs=500,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=200,
        epoch_verbose=200
    )
    
    weights_equal = False

    while not weights_equal:
        train_loader, dev_loader = get_policy_data(nsteps, n_samples)
        trainer.train_data, trainer.dev_data = train_loader, dev_loader
        trainer.problem = problem
        # Train control policy
        best_model = trainer.train()

        # load best trained model
        trainer.model.load_state_dict(best_model)
    
        new_weights = update_weights(list(system.parameters())[0].detach().flatten(), config["q"], config["epsilon"])
        weights_equal = torch.equal(new_weights, weight_matrix)
        weight_matrix = new_weights
        if not weights_equal:
            print("Restarting with new_weights")
        trainer.badcount = 0


# In[11]:


def train_func_p(optim, system, config):
    nsteps = system.nsteps
    p = config['p'] 
    l1 = variable([x], lambda x: torch.norm(list(system.parameters())[0], p))
    loss_l1 = config["lambda"]*(l1 == 0)  
    
    # list of constraints and objectives
    objectives = [regulation_loss, loss_l1]
    components = [system]
    loss = PenaltyLoss(objectives, constraints)
    problem = Problem(components, loss)
    trainer = Trainer(
        problem,
        None,
        None,
        optimizer=optim,
        epochs=500,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=200,
        epoch_verbose=200
    )
    print(f'p: {p}')
    train_loader, dev_loader = get_policy_data(nsteps, n_samples)
    trainer.train_data, trainer.dev_data = train_loader, dev_loader 
    trainer.problem = problem
    
    # Train control policy
    best_model = trainer.train()
    
    # load best trained model
    trainer.model.load_state_dict(best_model)
    system.nsteps = nsteps
    trainer.badcount = 0


# In[12]:


ntests = 100
n_samples = 750    # number of sampled scenarios

def objective(config):
    torch.manual_seed(0)
    n_steps = config["n_steps"]
    max_freq = config["max_freq"]
    max_degree = config["max_degree"]
    sparsity = config["lambda"]

    u_bound = Node(lambda u: torch.sigmoid(u) * (umax-umin) + umin, ['u_unbound'], ['u'])    

    losses = []
    for _ in range(ntests):
        theta_1 = library.PolynomialLibrary(nx, nref, max_degree=max_degree)
        theta_2 = library.FourierLibrary(nx, nref, max_freq=max_freq)

        policy_sindy = sindy.SINDy(library.CombinedLibrary([theta_2,theta_1]), n_out=nu)
        policy_node = Node(policy_sindy,  ['xn', 'r'], ['u_unbound'])
        sindy_system = System([policy_node, u_bound, integrator_node], nsteps=n_steps)
    
        optimizer = torch.optim.AdamW(policy_node.parameters(), lr=0.005)
   
        config["train"](optimizer, sindy_system, config)
        losses.append(test(sindy_system, data))  # Compute test accuracy
    losses = torch.tensor(losses)
    tune.report({"eval_loss": torch.mean(losses).item(), "variance": torch.var(losses).item(),
                 "params": list(policy_sindy.parameters())[0]})  # Report to Tune



search_space = {"n_steps": tune.grid_search([2, 5, 10]), "n_iter": tune.grid_search([1]),
               "max_degree": tune.grid_search([0, 1, 2]), "max_freq": tune.grid_search([0, 2, 5]),
               "train": tune.grid_search([train_func]),
               "lambda": tune.grid_search([1e-1, 1e-5, 1e-7])}

tuner = tune.Tuner( 
    objective,
    tune_config=tune.TuneConfig(
        metric="_metric/eval_loss",
        mode="min",    
    ),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
results.get_dataframe().to_csv("csv/tt.csv")


# In[13]:


config_l1 = results.get_best_result().config

config = config_l1
config['params'] = results.get_best_result().metrics['_metric']['params']
search_space = {"n_steps": tune.grid_search([config["n_steps"]]), "n_iter": tune.grid_search([config["n_iter"]]),
               "max_freq": tune.grid_search([config["max_freq"]]), "max_degree": tune.grid_search([config["max_degree"]]),
               "lambda": tune.grid_search([config["lambda"]]),
               "train": tune.grid_search([train_func_weighted]),
               "q": tune.grid_search([0, 1, 2, 3]),
               "epsilon": tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5])}


# In[14]:


tuner = tune.Tuner( 
    objective,
    tune_config=tune.TuneConfig(
        metric="_metric/eval_loss",
        mode="min",    
    ),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
config_weighted = results.get_best_result().config
config_weighted['params'] = results.get_best_result().metrics['_metric']['params']
results.get_dataframe().to_csv("csv/tt-weighted.csv")


# In[15]:


search_space = {"n_steps": tune.grid_search([config["n_steps"]]), "n_iter": tune.grid_search([config["n_iter"]]),
               "max_freq": tune.grid_search([config["max_freq"]]), "max_degree": tune.grid_search([config["max_degree"]]),
               "lambda": tune.grid_search([config["lambda"]]),
               "train": tune.grid_search([train_func_p]),
               "p": tune.grid_search([0, 0.5, 1, 1.5, 2])}

tuner = tune.Tuner( 
    objective,
    tune_config=tune.TuneConfig(
        metric="_metric/eval_loss",
        mode="min",    
    ),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
config_lp = results.get_best_result().config
config_lp['params'] = results.get_best_result().metrics['_metric']['params']
results.get_dataframe().to_csv("csv/tt-lp.csv")


# In[16]:


search_space = {"n_steps": tune.grid_search([2,]), "n_iter": tune.grid_search([1, 2, 3,]),
               "max_freq": tune.grid_search([config["max_freq"]]), "max_degree": tune.grid_search([config["max_degree"]]),
               "lambda": tune.grid_search([config["lambda"]]),
               "train": tune.grid_search([train_func]),
               }

tuner = tune.Tuner( 
    objective,
    tune_config=tune.TuneConfig(
        metric="_metric/eval_loss",
        mode="min",    
    ),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
config_cirriculum = results.get_best_result().config
config_cirriculum["params"] = results.get_best_result().metrics['_metric']['params']
results.get_dataframe().to_csv("csv/tt-curr.csv")


# In[17]:


configs = [config_l1, config_weighted, config_lp, config_cirriculum]
names = ["reg", "weighted", "lp", "curriculum"]
noises = [0, 1e-3, 1e-1, 1, 1e1, 1e3]
traj = dict()
ntests = 1

for setup in zip(configs, names):
    torch.manual_seed(0)
    config = setup[0]
    name = setup[1]
    traj[name] = dict()

    n_steps = config["n_steps"]
    max_freq = config["max_freq"]
    max_degree = config["max_degree"]
    train = config["train"]
    
    u_bound = Node(lambda u: torch.sigmoid(u) * (umax-umin) + umin, ['u_unbound'], ['u'])

    params = []
    policy_sindy = None
    theta_1 = library.PolynomialLibrary(nx, nref, max_degree=max_degree)
    theta_2 = library.FourierLibrary(nx, nref, max_freq=max_freq)

    policy_sindy = sindy.SINDy(library.CombinedLibrary([theta_2,theta_1]), n_out=nu)
    policy_node = Node(policy_sindy,  ['xn', 'r'], ['u_unbound'])
    sindy_system = System([policy_node, u_bound, integrator_node], nsteps=n_steps)
    


    params = config["params"]
    policy_sindy.set_parameters(params)
    model_parameters = filter(lambda p: p.requires_grad, policy_sindy.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    traj[name]["nparam"] = nparams
    traj[name]["params"] = params
    
    nsteps = 750
    for noise_level in noises:
        noise = Node(lambda x: torch.randn(x.shape)*noise_level + x, ['xn'], ['xnoise'])
        policy_node = Node(policy_sindy, ['xnoise', 'r'], ['u_unbound'])
        sindy_system = System([noise, policy_node, u_bound, integrator_node], nsteps=nsteps)


        # perform closed-loop simulation
        trajectories_sindy = sindy_system(data)

        traj[name][noise_level] = trajectories_sindy
        traj[name][noise_level]['loss'] = torch.nn.functional.mse_loss(trajectories_sindy['xn'], data['r'])


# In[19]:


nsteps = 100  # prediction horizon
n_samples = 2000    # number of sampled scenarios

train_loader, dev_loader = get_policy_data(nsteps, n_samples)

traj["nn"] = dict()
# neural net control policy with hard control action bounds
net = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)
policy = Node(net, ['xn', 'r'], ['u'], name='policy')

objectives = [regulation_loss]

cl_system = System([policy, integrator_node], nsteps=nsteps)

# data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
components = [cl_system]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(components, loss)

optimizer = torch.optim.AdamW(policy.parameters(), lr=0.002)
#  Neuromancer trainer
trainer = Trainer(
    problem,
    train_loader, dev_loader,
    optimizer=optimizer,
    epochs=1000,
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=500,
    epoch_verbose=50
)
# Train control policy
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
traj["nn"]["nparam"] = params
nsteps = 750
for noise_level in noises:
    noise = Node(lambda x: torch.randn(x.shape)*noise_level + x, ['xn'], ['xnoise'])
    policy = Node(net, ['xnoise', 'r'], ['u'], name='policy')
    nn_system = System([noise, policy_node, u_bound, integrator_node], nsteps=nsteps)
    nn_trajectories = nn_system(data)
    traj["nn"][noise_level] = nn_trajectories
    traj["nn"][noise_level]['loss'] = torch.nn.functional.mse_loss(nn_trajectories['xn'], data['r'])

# In[37]:


import pickle

with open('csv/tt_data.pyc', 'ab') as outfile:
    pickle.dump(traj, outfile)

