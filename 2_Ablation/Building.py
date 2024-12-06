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
torch.manual_seed(0)
dev = torch.device('cpu')

# In[3]:

nonlin_system_name = "Reno_full"
nonlin_system = psl.systems[nonlin_system_name]
nonlin_sys = nonlin_system(seed=50)

system_name = "LinearReno_full"
system = psl.systems[system_name]
sys = system()

ts = sys.ts
nx = sys.nx
ny = sys.ny
nu = sys.nu
nd = sys.nd
nd_obs = sys.nD_obs
nref = ny

umin = torch.tensor(sys.umin, device=dev)
umax = torch.tensor(sys.umax, device=dev)
def normalize(x, mean, std): 
    return ((x-mean)/std).nan_to_num()
def denormalize(x, mean, std): 
    return ((x*std)+mean).nan_to_num()

umean = torch.tensor(sys.stats['U']['mean'], device=dev)
ustd = torch.tensor(sys.stats['U']['std'], device=dev)
xmean = torch.tensor(sys.stats['X']['mean'], device=dev)
xstd = torch.tensor(sys.stats['X']['std'], device=dev)
ymean = torch.tensor(sys.stats['Y']['mean'], device=dev)
ystd = torch.tensor(sys.stats['Y']['std'], device=dev)
dmean = torch.tensor(sys.stats['Dhidden']['mean'], device=dev)
dstd = torch.tensor(sys.stats['Dhidden']['std'], device=dev)

umin = normalize(umin, umean, ustd)
umax = normalize(umax, umean, ustd)

A = torch.tensor(sys.A, device=dev)
B = torch.tensor(sys.Beta, device=dev)
C = torch.tensor(sys.C, device=dev)
E = torch.tensor(sys.E, device=dev)
F = torch.tensor(sys.F, device=dev)
G = torch.tensor(sys.G, device=dev)
y_ss = torch.tensor(sys.y_ss, device=dev)

def ssm_forward(x, u, d):
    x = denormalize(x, xmean, xstd)
    u = denormalize(u, umean, ustd)
    d = denormalize(d, dmean, dstd)

    x = x @ A.T + u @ B.T + d @ E.T + G.T
    y = x @ C.T + F.T - y_ss

    x = normalize(x, xmean, xstd)
    y = normalize(y, ymean, ystd)
    return x, y

dist_obs = Node(lambda d: d[:, sys.d_idx], ['d'], ['d_obs'])

def get_policy_data(sys, nsteps, n_samples, xmin_range, batch_size, name="train"):
    #  sampled references for training the policy
    batched_xmin = xmin_range.sample((n_samples, 1, nref)).repeat(1, nsteps + 1, 1).to(dev)
    batched_xmax = batched_xmin + 2.

    # sampled disturbance trajectories from the simulation model
    batched_dist = torch.stack([torch.tensor(nonlin_sys.get_D(nsteps), device=dev) for _ in range(n_samples)])

    # sampled nonlinial conditions
    batched_x0 = torch.stack([torch.tensor(nonlin_sys.get_x0(), device=dev).unsqueeze(0) for _ in range(n_samples)])

    data = DictDataset(
        {"x": normalize(batched_x0, xmean, xstd),
         "ymin": normalize(batched_xmin, ymean, ystd),
         "ymax": normalize(batched_xmax, ymean, ystd),
         "d": normalize(batched_dist, dmean, dstd),
         "umin": umin.repeat((n_samples, nsteps, 1)),
         "umax": umax.repeat((n_samples, nsteps, 1)),
         "y": normalize(batched_x0 @ C.T + F.T - y_ss, ymean, ystd)
         
        },
        name=name,
    )
    return DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)
# In[5]:




nsteps_test = 1000

x0 = torch.tensor(nonlin_sys.get_x0(), device=dev).reshape(1, 1, nx)
torch_dist = torch.tensor(nonlin_sys.get_D(nsteps_test+1), device=dev).unsqueeze(0)

np.random.seed(0)
np_refs = psl.signals.step(nsteps_test+1, nref, min=18., max=22., randsteps=5, rng=np.random.default_rng(seed=0))
ymin_val = torch.tensor(np_refs, dtype=torch.float32, device=dev).reshape(1, nsteps_test+1, nref)
ymax_val = ymin_val+2.0

data = {
        "x": normalize(x0, xmean, xstd),
         "ymin": normalize(ymin_val, ymean, ystd),
         "ymax": normalize(ymax_val, ymean, ystd),
         "d": normalize(torch_dist, dmean, dstd),
         "umin": umin.repeat(nsteps_test, 1),
         "umax": umax.repeat(nsteps_test, 1),
         "y": normalize(x0 @ C.T + F.T - y_ss, ymean, ystd)

}

from ray import train, tune, air


# In[8]:


action_weight = 0.1
state_weight = 50.
du_weight = 0.0
u_bound_weight = 1.

def train_func(optim, system, config):
    nsteps = system.nsteps
    # variables
    y = variable('y')
    u = variable('u')
    ymin_val = variable('ymin')
    ymax_val = variable('ymax')
    umin_val = variable('umin')
    umax_val = variable('umax')
    
    l = variable([y], lambda y: torch.norm(list(system.parameters())[0], config['p']))

    loss_l = config['lambda']*((l == 0))
    
    action_loss = action_weight * ((u == 0.0))  # energy minimization
    du_loss = du_weight * (u[:,:-1,:] - u[:,1:,:] == 0.0)  # delta u minimization to prevent agressive changes in control actions
    
    state_lower_bound_penalty = state_weight*(y >= ymin_val)
    state_upper_bound_penalty = state_weight*(y <= ymax_val)

    u_lower_bound_penalty = u_bound_weight*(u >= umin_val)
    u_upper_bound_penalty = u_bound_weight*(u <= umax_val)

    constraints = [state_lower_bound_penalty, state_upper_bound_penalty, u_lower_bound_penalty, u_upper_bound_penalty]
    objectives = [action_loss, loss_l]
    
    components = [system]

    train_loader, dev_loader = [
            get_policy_data(sys, nsteps, n_samples, xmin_range, batch_size, name=name)
            for name in ("train", "dev")]   
    loss = PenaltyLoss(objectives, constraints)
    problem = Problem(components, loss)
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        optimizer=optim,
        epochs=500,
        train_metric='train_loss',
        eval_metric='dev_loss',
        patience=500,
        epoch_verbose=1,
        device=dev
    )
    
    # Train control policy
    best_model = trainer.train()
    
    # load best trained model
    trainer.model.load_state_dict(best_model)


    system.nsteps = nsteps
    trainer.badcount = 0

def test(system, test_data):
    system.nsteps = nsteps_test
    trajectories_sindy = system(test_data)
    sindy_y = denormalize(trajectories_sindy['y'], ymean, ystd)
    sindy_u = denormalize(trajectories_sindy['u'], umean, ustd)

    y_min = denormalize(test_data["ymin"], ymean, ystd)
    y_max = denormalize(test_data["ymax"], ymean, ystd)

    u_min = denormalize(test_data["umin"], umean, ustd)
    u_max = denormalize(test_data["umax"], umean, ustd) 
    u_loss = .1 * torch.sum(torch.abs(sindy_u))

    y_lower = torch.sum(torch.abs(torch.max(sindy_y - y_max, torch.zeros(sindy_y.shape))))
    y_upper = torch.sum(torch.abs(torch.min(sindy_y - y_min, torch.zeros(sindy_y.shape))))

    u_lower = torch.sum(torch.abs(torch.max(sindy_u - u_max, torch.zeros(sindy_u.shape))))
    u_upper = torch.sum(torch.abs(torch.min(sindy_u - u_min, torch.zeros(sindy_u.shape))))



    return (u_loss + state_weight*y_lower +  state_weight*y_upper + 10.*u_lower + 10.*u_upper).item(), trajectories_sindy
    
# In[10]:


def update_weights(coef, q, epsilon):
    return torch.diag(1 / (coef.flatten()**q + epsilon))

def train_func_weighted(optim, system, config):
    nsteps = system.nsteps

    weight_matrix = torch.eye(list(system.parameters())[0].flatten().shape[0])
    weighted_l1 = variable([y], lambda y: torch.norm(torch.matmul(weight_matrix, list(system.parameters())[0].flatten())))
    loss_l1 = config["lambda"]*((weighted_l1 == 0)) 
    u_bound_weight = config['bound']
    u_lower_bound_penalty = u_bound_weight*(u >= umin_val)
    u_upper_bound_penalty = u_bound_weight*(u <= umax_val)

    constraints = [state_lower_bound_penalty, state_upper_bound_penalty, u_lower_bound_penalty, u_upper_bound_penalty]
    # list of constraints and objectives
    objectives = [action_loss, du_loss, loss_l1]
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
        epochs=10000//system.nsteps,
        train_metric='train_loss',
        eval_metric='dev_loss',
        patience=10000//system.nsteps,
        epoch_verbose=100
    )
    
    weights_equal = False

    while not weights_equal:
        train_loader, dev_loader = [
            get_policy_data(modelSystem, nsteps, n_samples, xmin_range, batch_size, name=name)
            for name in ("train", "dev")]
        trainer.problem = problem
        trainer.train_data, trainer.dev_data = train_loader, dev_loader
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




ntests = 1
n_samples = 3000    # number of sampled scenarios
batch_size = 1500
xmin_range = torch.distributions.Uniform(18., 22.)

def objective(config):
    torch.manual_seed(0)
    n_steps = config["n_steps"]
    max_freq = config["max_freq"]
    max_degree = config["max_degree"]
    sparsity = config["lambda"]


    losses = []
    for _ in range(ntests):
        theta_1 = library.FourierLibrary(ny, 2*nref+nd_obs, max_freq=max_freq)
        theta_2 = library.PolynomialLibrary(ny, 2*nref+nd_obs, max_degree=max_degree, interaction=False)
        policy_sindy = sindy.SINDy(library.CombinedLibrary([theta_2]), n_out=nu)
        ssm = Node(ssm_forward, ['x', 'u', 'd'], ['x', 'y'])
        policy_node = Node(policy_sindy,  ['y', 'R'], ['u'])
        combined_y = Node(lambda ymin, ymax, d: torch.cat([ymin, ymax, d], dim=-1),
                  ['ymin', 'ymax', 'd_obs'], ['R'], name="y")
        sindy_system = System([dist_obs, combined_y, policy_node, ssm],
                    nsteps=n_steps,
                    name='sindy_system')

        optimizer = torch.optim.AdamW(policy_sindy.parameters(), lr=0.06)
        config["train"](optimizer, sindy_system, config)
        lossp, traj = test(sindy_system, data)
        losses.append(lossp)  # Compute test accuracy
    losses = torch.tensor(losses)
    tune.report({"eval_loss": torch.mean(losses).item(), "variance": torch.var(losses).item(),
                "params": list(policy_sindy.parameters())[0]})  # Report to Tune



search_space = {"n_steps": tune.grid_search([100]), "n_iter": tune.grid_search([1]),
               "max_freq": tune.grid_search([0]), "max_degree": tune.grid_search([1]),
               "train": tune.grid_search([train_func]),
               "bound": tune.grid_search([1.]), "p": tune.grid_search([1]),
               "lambda": tune.grid_search([0])}

tuner = tune.Tuner( 
    objective,
    tune_config=tune.TuneConfig(
        metric="_metric/eval_loss",
        mode="min",
           
    ),
    run_config=air.RunConfig(verbose=1),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
results.get_dataframe().to_csv("csv/reno-sample.csv")

config_l1 = results.get_best_result().config
config = config_l1
config["params"] = results.get_best_result().metrics['_metric']['params']

# In[13]:

"""

search_space = {"n_steps": tune.grid_search([config["n_steps"]]), "n_iter": tune.grid_search([config["n_iter"]]),
               "max_freq": tune.grid_search([config["max_freq"]]), "max_degree": tune.grid_search([config["max_degree"]]),
               "lambda": tune.grid_search([config["lambda"]]), "bound": tune.grid_search([config["bound"]]),
               "train": tune.grid_search([train_func_weighted]),
               "q": tune.grid_search([1, 0]),
               "epsilon": tune.grid_search([1e-1, 1e-2, 1e-4])}


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
results.get_dataframe().to_csv("csv/reno-weighted-sample.csv")


# In[15]:


search_space = {"n_steps": tune.grid_search([config["n_steps"]]), "n_iter": tune.grid_search([config["n_iter"]]),
               "max_freq": tune.grid_search([config["max_freq"]]), "max_degree": tune.grid_search([config["max_degree"]]),
               "lambda": tune.grid_search([config["lambda"]]), "bound": tune.grid_search([config["bound"]]),
               "train": tune.grid_search([train_func_p]),
               "p": tune.grid_search([0.5, 1.5, 2])}

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
results.get_dataframe().to_csv("csv/reno-lp-sample.csv")


# In[16]:

search_space = {"n_steps": tune.grid_search([config["n_steps"]]), "n_iter": tune.grid_search([2]),
               "max_freq": tune.grid_search([config["max_freq"]]), "max_degree": tune.grid_search([config["max_degree"]]),
               "lambda": tune.grid_search([config["lambda"]]), "bound": tune.grid_search([config["bound"]]),
               "train": tune.grid_search([train_func]), "nstep": tune.grid_search([True])
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
config_cirriculum['params'] = results.get_best_result().metrics['_metric']['params']
results.get_dataframe().to_csv("csv/reno-nstep-sample.csv")


search_space = {"n_steps": tune.grid_search([config["n_steps"]]), "n_iter": tune.grid_search([1,2,3]),
               "max_freq": tune.grid_search([config["max_freq"]]), "max_degree": tune.grid_search([config["max_degree"]]),
               "lambda": tune.grid_search([config["lambda"]]), "bound": tune.grid_search([config["bound"]]),
               "train": tune.grid_search([train_func]), "nstep": tune.grid_search([False])
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
config_bound = results.get_best_result().config
config_bound = results.get_best_result().metrics['_metric']['params']
results.get_dataframe().to_csv("csv/reno-bound.csv")
"""
# In[17]:

configs = [config_l1]#, config_weighted, config_lp, config_cirriculum]
names = ["reg"]#, "weighted", "lp", "nstep"]
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
    print(f"{config['params']}\n\n\n")
    params = config["params"]
    
    theta_1 = library.FourierLibrary(ny, 2*nref+nd_obs, max_freq=max_freq)
    theta_2 = library.PolynomialLibrary(ny, 2*nref+nd_obs, max_degree=max_degree, interaction=False)

    policy_sindy = sindy.SINDy(library.CombinedLibrary([theta_1,theta_2]), n_out=nu)
    policy_node = Node(policy_sindy,  ['y', 'R'], ['u'])
    state_model = Node(ssm_forward, ['x', 'u', 'd'], ['x', 'y'], name='SSM')
    combined_y = Node(lambda ymin, ymax, d: torch.cat([ymin, ymax, d], dim=-1),
              ['ymin', 'ymax', 'd_obs'], ['R'], name="y")
    sindy_system = System([dist_obs, combined_y, policy_node, state_model],
                nsteps=n_steps,
                name='sindy_system')

        
    
    policy_sindy.set_parameters(params)
    model_parameters = filter(lambda p: p.requires_grad, policy_sindy.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    traj[name]["nparam"] = nparams
    traj[name]["params"] = params
    
    nsteps = nsteps_test
    for noise_level in noises:
        noise = Node(lambda x: torch.randn(x.shape)*noise_level + x, ['x'], ['xnoise'])
        policy_node = Node(policy_sindy, ['y', 'R'], ['u'])
        ssm_node = Node(ssm_forward, ['xnoise', 'u', 'd'], ['x', 'y'])
        dist_model = lambda d: d[:, sys.d_idx]
        dist_obs = Node(dist_model, ['d'], ['d_obs'], name='dist_obs')
        combined_y = Node(lambda ymin, ymax, d: torch.cat([ymin, ymax, d], dim=-1),
                  ['ymin', 'ymax', 'd_obs'], ['R'], name="y")
        sindy_system = System([noise, dist_obs, combined_y, policy_node, ssm_node],
                    nsteps=n_steps,
                    name='sindy_system')



        # perform closed-loop simulation

        lossy, traject = test(sindy_system, data)
        traj[name][noise_level] = traject
        traj[name][noise_level]["loss"] = lossy
        print(f"{noise_level} for {name} finished.")

# In[19]:


nsteps = 100  # prediction horizon
n_samples = 3000    # number of sampled scenarios

train_loader, dev_loader = [get_policy_data(sys, nsteps, n_samples, xmin_range, batch_size, name=name)
            for name in ("train", "dev")]

traj["nn"] = dict()
# neural net control policy with hard control action bounds
net = blocks.MLP_bounds(insize=ny+2*nref+nd_obs, outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)
policy = Node(net, ['y', 'R'], ['u'], name='policy')

ssm = Node(ssm_forward, ['x', 'u', 'd'], ['x', 'y'], name='SSM')

cl_system = System([dist_obs, combined_y, policy_node, ssm], nsteps=nsteps)

# data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
y = variable('y')
u = variable('u')
ymin_val = variable('ymin')
ymax_val = variable('ymax')
    
action_loss = 0.1 * ((u == 0.0))  # energy minimization
    
state_lower_bound_penalty = state_weight*(y >= ymin_val)
state_upper_bound_penalty = state_weight*(y <= ymax_val)


constraints = [state_lower_bound_penalty, state_upper_bound_penalty]
objectives = [action_loss]
# create constrained optimization loss
loss = PenaltyLoss(objectives, [state_upper_bound_penalty, state_lower_bound_penalty])
# construct constrained optimization problem
problem = Problem([cl_system], loss)

optimizer = torch.optim.AdamW(policy.parameters(), lr=0.02)
#  Neuromancer trainer
trainer = Trainer(
    problem,
    train_loader, dev_loader,
    optimizer=optimizer,
    epochs=200,
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=200,
    epoch_verbose=20
)
# Train control policy
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
traj["nn"]["nparam"] = params
nsteps = nsteps_test
for noise_level in noises:
    noise = Node(lambda x: torch.randn(x.shape)*noise_level + x, ['x'], ['xnoise'])
    ssm = Node(ssm_forward, ['xnoise', 'u', 'd'], ['x', 'y'], name='y=Cx')
    nn_system = System([noise, dist_obs, combined_y, policy, noise, ssm],
                   nsteps=n_steps,
                   name='nn_system')
    lossy, traject = test(nn_system, data)
    traj["nn"][noise_level] = traject
    traj["nn"][noise_level]["loss"] = lossy

# In[37]:


import pickle

with open('csv/reno_data-sample.pyc', 'ab') as outfile:
    pickle.dump(traj, outfile)

