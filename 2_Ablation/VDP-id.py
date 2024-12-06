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
nsim = 3000   # number of simulation steps in the dataset
nsteps = 3   # number of prediction horizon steps in the loss function
bs = 100     # minibatching batch size
train_loader, dev_loader, test_data = get_data(gt_model, nsim, nsteps, ts, bs)

theta_1 = library.PolynomialLibrary(nx, nu, 3)

theta_2_funs = [lambda X, u: torch.sqrt(X[:,0]), lambda X, u: torch.sqrt(X[:,1])]
theta_2_names = ["sqrt(x_0)", "sqrt(x_1)"]
theta_2 = library.FunctionLibrary(theta_2_funs, nx, nu, theta_2_names)

theta_3 = library.FourierLibrary(nx, nu, 1)

fx = sindy.SINDy(library.CombinedLibrary([theta_1]))


integrator = integrators.Euler(fx, h=ts)
integrator_node = Node(integrator, ['xn', 'u'], ['x'])
x_bound = Node(lambda x: torch.clamp(x, xmin, xmax), ['x'], ['xn'])
dynamics_model = System([integrator_node, x_bound], nsteps=nsteps, name="dynamics_model")

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
loss_l1 = 0.0001*(l1 == 0)


# aggregate list of objective terms and constraints
objectives = [reference_loss, onestep_loss, loss_l1]

# create constrained optimization loss
loss = PenaltyLoss(objectives, [])
# construct constrained optimization problem
problem = Problem([dynamics_model], loss)



optimizer = torch.optim.AdamW(problem.parameters(),
                             lr=0.001)
# trainer
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_data,
    optimizer,
    patience=1000,
    epochs=4000,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    epoch_verbose=1000
)



# %% train
best_model = trainer.train()
problem.load_state_dict(best_model)


print(f"\n\n\n{fx}\n\n\n")
print("Finished with system ID")

# In[4]:


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


# In[5]:




# In[6]:


nsteps_test = 100

torch.manual_seed(22)
data = {'xn': torch.randn(1, 1, nx, dtype=torch.float32),
        'r': torch.zeros(1, nsteps_test+1, nx, dtype=torch.float32)}


# In[7]:


from ray import train, tune


# In[8]:


# variables
x = variable('xn')
ref = variable('r')
u = variable('u')

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
        epochs=250,
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
    system.nsteps = nsteps_test

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
        epochs=250,
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
        epochs=250,
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


ntests = 1
n_samples = 500    # number of sampled scenarios

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
        sindy_system = System([policy_node, u_bound, integrator_node, x_bound], nsteps=n_steps)
    
        optimizer = torch.optim.AdamW(policy_node.parameters(), lr=0.005)
   
        config["train"](optimizer, sindy_system, config)
        losses.append(test(sindy_system, data))  # Compute test accuracy
    losses = torch.tensor(losses)
    tune.report({"eval_loss": torch.mean(losses).item(), "variance": torch.var(losses).item()})  # Report to Tune



search_space = {"n_steps": tune.grid_search([20]), "n_iter": tune.grid_search([1]),
               "max_freq": tune.grid_search([2]), "max_degree": tune.grid_search([0,]),
               "train": tune.grid_search([train_func]),
               "lambda": tune.grid_search([1e-1])}

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
results.get_dataframe().to_csv("csv/vdp-id.csv")


# In[13]:


config_l1 = results.get_best_result().config
config = config_l1
search_space = {"n_steps": tune.grid_search([config["n_steps"]]), "n_iter": tune.grid_search([config["n_iter"]]),
               "max_freq": tune.grid_search([config["max_freq"]]), "max_degree": tune.grid_search([config["max_degree"]]),
               "lambda": tune.grid_search([config["lambda"]]),
               "train": tune.grid_search([train_func_weighted]),
               "q": tune.grid_search([1]),
               "epsilon": tune.grid_search([1e-4])}


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
results.get_dataframe().to_csv("csv/vdp-weighted-id.csv")


# In[15]:


search_space = {"n_steps": tune.grid_search([config["n_steps"]]), "n_iter": tune.grid_search([config["n_iter"]]),
               "max_freq": tune.grid_search([config["max_freq"]]), "max_degree": tune.grid_search([config["max_degree"]]),
               "lambda": tune.grid_search([config["lambda"]]),
               "train": tune.grid_search([train_func_p]),
               "p": tune.grid_search([0, 0.5, 1.5, 2])}

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
results.get_dataframe().to_csv("csv/vdp-lp-id.csv")


# In[16]:


search_space = {"n_steps": tune.grid_search([2,]), "n_iter": tune.grid_search([3,]),
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
results.get_dataframe().to_csv("csv/vdp-curr-id.csv")


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
    for _ in range(ntests):
        theta_1 = library.PolynomialLibrary(nx, nref, max_degree=max_degree)
        theta_2 = library.FourierLibrary(nx, nref, max_freq=max_freq)

        policy_sindy = sindy.SINDy(library.CombinedLibrary([theta_2,theta_1]), n_out=nu)
        policy_node = Node(policy_sindy,  ['xn', 'r'], ['u_unbound'])
        sindy_system = System([policy_node, u_bound, integrator_node, x_bound], nsteps=n_steps)

        optimizer = torch.optim.AdamW(policy_sindy.parameters(), lr=0.005)

        train(optimizer, sindy_system, config)
        params.append(policy_sindy.coef.data)

    params = torch.stack(params)
    new_coef = torch.mean(params, dim=0)
    policy_sindy.set_parameters(torch.nn.Parameter(new_coef))
    policy_sindy.set_parameters(torch.nn.Parameter(new_coef))
    model_parameters = filter(lambda p: p.requires_grad, policy_sindy.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    traj[name]["nparam"] = params
    
    nsteps = 750
    for noise_level in noises:
        noise = Node(lambda x: torch.randn(x.shape)*noise_level + x, ['xn'], ['xnoise'])
        policy_node = Node(policy_sindy, ['xnoise', 'r'], ['u_unbound'])
        sindy_system = System([noise, policy_node, u_bound, integrator_node, x_bound], nsteps=nsteps)


        # perform closed-loop simulation
        trajectories_sindy = sindy_system(data)

        traj[name][noise_level] = trajectories_sindy
        traj[name][noise_level]['loss'] = torch.nn.functional.mse_loss(trajectories_sindy['xn'], data['r'])


# In[19]:


nsteps = 50  # prediction horizon
n_samples = 2000    # number of sampled scenarios

train_loader, dev_loader = get_policy_data(nsteps, n_samples)

traj["nn"] = dict()
# neural net control policy with hard control action bounds
net = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)
policy = Node(net, ['xn', 'r'], ['u'], name='policy')

objectives = [regulation_loss]

cl_system = System([policy, integrator_node, x_bound], nsteps=nsteps)

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
    epochs=100,
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=50,
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
    nn_system = System([noise, policy_node, u_bound, integrator_node, x_bound], nsteps=nsteps)
    nn_trajectories = nn_system(data)
    traj["nn"][noise_level] = nn_trajectories
    traj["nn"][noise_level]['loss'] = torch.nn.functional.mse_loss(nn_trajectories['xn'], data['r'])

# In[37]:


import pickle

with open('csv/vdp_data-id.pyc', 'ab') as outfile:
    pickle.dump(traj, outfile)

