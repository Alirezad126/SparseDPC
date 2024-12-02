#!/usr/bin/env python
# coding: utf-8

# In[61]:


import torch
from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.lines import Line2D

from neuromancer.system import Node, System
from neuromancer.dynamics import integrators
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.plot import pltOL

import sindy
import library



dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(dev)
torch.manual_seed(0);


# In[62]:


# ground truth system
system_name = "LinearReno_ROM40"
system = psl.systems[system_name]
modelSystem = system()
ts = modelSystem.ts
nx = modelSystem.nx
ny = modelSystem.ny
nu = modelSystem.nu
nd = modelSystem.nd
raw = modelSystem.simulate(nsim=1000)


# In[63]:


def get_data(sys, nsim, nsteps, ts):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size

    """
    sim = sys.simulate(nsim=nsim, ts=ts)
    nx = sys.nx
    nu = sys.nu
    nd = sys.nd
    ny = sys.ny
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    def normalize(x, mean, std):
        return (x - mean) / std

    X = normalize(sim['X'][:length], modelSystem.stats['X']['mean'], modelSystem.stats['X']['std'])
    X = torch.tensor(X).reshape(nbatch, nsteps, nx)
    Y = normalize(sim['Y'][:length], modelSystem.stats['Y']['mean'], modelSystem.stats['Y']['std'])
    Y = torch.tensor(Y).reshape(nbatch, nsteps, ny)
    U = normalize(sim['U'][:length], modelSystem.stats['U']['mean'], modelSystem.stats['U']['std'])
    U = torch.tensor(U).reshape(nbatch, nsteps, nu)
    D = normalize(sim['D'][:length], modelSystem.stats['D']['mean'], modelSystem.stats['D']['std'])
    D = torch.tensor(D).reshape(nbatch, nsteps, nd)

    return {"X": X, "yn": Y[:,:1], "Y": Y, "U": U, "D": D}

def get_splits(sys, nsim, nsteps, ts, bs):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size
    :param bs: (int) batch size
    """
    train_data, dev_data = [get_data(sys, nsim, nsteps, ts) for _ in range(2)]

    train_data = DictDataset(train_data, name="train")
    train_loader = DataLoader(train_data, batch_size=bs, collate_fn=train_data.collate_fn, shuffle=True)

    dev_data = DictDataset(dev_data, name="dev")
    dev_loader = DataLoader(dev_data, batch_size=bs, collate_fn=dev_data.collate_fn, shuffle=True)

    test_data = get_data(sys, nsim, nsim, ts)
    test_data["name"] = "test"

    return train_loader, dev_loader, test_data


# In[64]:


n_latent = 10 # latent state space dimension

# latent state estimator
encoder = blocks.MLP(ny, n_latent, bias=True,
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ReLU,
                hsizes=[40])
#encoder = sindy.SINDy(library.PolynomialLibrary(ny, max_f=1,), n_out=n_latent)

encode_sym = Node(encoder, ['yn'], ['xn'], name='encoder')


# In[65]:


# construct latent NODE model in Neuromancer
fx = blocks.MLP(n_latent+nu+nd, n_latent, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.Tanh,
                    hsizes=[40, 40])

fx = sindy.SINDy(library.FourierLibrary(n_latent, nu+nd, max_freq=2), n_out=n_latent)
combined = Node(lambda u, d: torch.cat([u,d], axis=-1), ['U', 'D'], ['c'])


# In[67]:


# integrate NODE with adjoint-based solver
fxEuler = integrators.Euler(fx, h=ts)
model = Node(fxEuler, ['xn', 'c'], ['xn'], name='NODE')


# In[68]:


# latent output model
decoder = blocks.MLP(n_latent, ny, bias=True,
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ReLU,
                hsizes=[40])

decoder = sindy.SINDy(library.PolynomialLibrary(n_latent, max_degree=1), n_out=ny)
decode_sym = Node(decoder, ['xn'], ['y'], name='decoder')


# In[69]:


nsim = 1000   # number of simulation steps in the dataset
nsteps = 200   # number of prediction horizon steps in the loss function
bs = 10    # minibatching batch size
train_loader, dev_loader, test_data = get_splits(modelSystem, nsim, nsteps, ts, bs)


# In[70]:


# latent NODE rollout
dynamics_model = System([combined, model, decode_sym], name='system', nsteps=nsteps)


# In[71]:


# %% Constraints + losses:
y = variable("Y")                      # observed
yhat = variable('y')                   # predicted output

# trajectory tracking loss
reference_loss = 5.*(yhat == y)^2
reference_loss.name = "ref_loss"

# one step tracking loss
onestep_loss = 1.*(yhat[:, 1, :] == y[:, 1, :])^2
onestep_loss.name = "onestep_loss"


# In[72]:


# putting things together
nodes = [encode_sym, dynamics_model]
objectives = [reference_loss, onestep_loss]
constraints = []

# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)

# construct constrained optimization problem
problem = Problem(nodes, loss)
# plot computational graph
problem.show()


# In[73]:


optimizer = torch.optim.Adam(problem.parameters(),
                                lr=0.003)
logger = BasicLogger(args=None, savedir='test', verbosity=1,
                        stdout=['dev_loss', 'train_loss'])

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_data,
    optimizer,
    patience=100,
    warmup=500,
    epochs=1000,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=logger,
    device=dev
)


# In[74]:


best_model = trainer.train()
problem.load_state_dict(best_model)
# In[26]:


nonlin_sys = psl.systems['Reno_ROM40']()
sys = psl.systems['LinearReno_ROM40']()
import numpy as np

nu = sys.nu
nref = ny

normalize = lambda x, mean, std: (x-mean)/std
denormalize = lambda x, mean, std: (x*std)+mean

umean = torch.tensor(sys.stats['U']['mean'], device=dev)
ustd = torch.tensor(sys.stats['U']['std'], device=dev)
xmean = torch.tensor(sys.stats['X']['mean'], device=dev)
xstd = torch.tensor(sys.stats['X']['std'], device=dev)
ymean = torch.tensor(sys.stats['Y']['mean'], device=dev)
ystd = torch.tensor(sys.stats['Y']['std'], device=dev)
dmean = torch.tensor(sys.stats['D']['mean'], device=dev)
dstd = torch.tensor(sys.stats['D']['std'], device=dev)
umin = torch.tensor(sys.umin, device=dev)
umax = torch.tensor(sys.umax, device=dev)
umin = normalize(umin, umean, ustd)
umax = normalize(umax, umean, ustd)


# In[27]:


def get_policy_data(sys, nsteps, n_samples, xmin_range, batch_size, name="train"):
    #  sampled references for training the policy
    batched_xmin = xmin_range.sample((n_samples, 1, nref)).repeat(1, nsteps + 1, 1).to(dev)
    batched_xmax = batched_xmin + 2.

    # sampled disturbance trajectories from the simulation model
    batched_dist = torch.stack([torch.tensor(nonlin_sys.get_D(nsteps), device=dev) for _ in range(n_samples)])

    # sampled nonlinial conditions
    batched_x0 = torch.stack([torch.tensor(nonlin_sys.get_x0(), device=dev).unsqueeze(0) for _ in range(n_samples)])
    batched_y0 = batched_x0.cpu().numpy() @ sys.C.T + sys.F.T - sys.y_ss
    batched_y0 = torch.tensor(batched_y0, device=dev)
    data = DictDataset(
        {
            
         "ymin": normalize(batched_xmin, ymean, ystd),
         "ymax": normalize(batched_xmax, ymean, ystd),
         "D": normalize(batched_dist, dmean, dstd),
         "umin": umin.repeat((n_samples, nsteps, 1)),
         "umax": umax.repeat((n_samples, nsteps, 1)),
         "y": normalize(batched_y0, ymean, ystd),
         "yn": normalize(batched_y0, ymean, ystd),
         
        },
        name=name,
    )
    return DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)


# In[49]:


nsteps_test = 1000

x0 = torch.tensor(nonlin_sys.get_x0(), device=dev).reshape(1, 1, nx)
y0 = x0 @ torch.tensor(sys.C.T) + torch.tensor(sys.F.T) + torch.tensor(sys.y_ss)
torch_dist = torch.tensor(nonlin_sys.get_D(nsteps_test+1), device=dev).unsqueeze(0)

np_refs = psl.signals.step(nsteps_test+1, nref, min=18., max=22., randsteps=3, rng=np.random.default_rng(seed=0))
ymin_val = torch.tensor(np_refs, dtype=torch.float32, device=dev).reshape(1, nsteps_test+1, nref)
ymax_val = ymin_val+2.0

data = {
    
         "ymin": normalize(ymin_val, ymean, ystd),
         "ymax": normalize(ymax_val, ymean, ystd),
         "D": normalize(torch_dist, dmean, dstd),
         "umin": umin.repeat(nsteps_test, 1),
         "umax": umax.repeat(nsteps_test, 1),
         "xn": encoder(normalize(y0, ymean, ystd)),
         "y": normalize(y0, ymean, ystd),
         
}


# In[29]:


action_weight = 0.1
state_weight = 50.
du_weight = 0.0
u_bound_weight = .1

def train_func(optim, system, sparsity):
    nsteps = system.nsteps
    # variables
    y = variable('y')
    u = variable('U')
    ymin_val = variable('ymin')
    ymax_val = variable('ymax')
    umin_val = variable('umin')
    umax_val = variable('umax')
    
    l = variable([y], lambda y: torch.norm(list(system.parameters())[0], p))

    loss_l = sparsity*((l == 0))
    
    action_loss = action_weight * ((u == 0.0))  # energy minimization
    #du_loss = du_weight * (u[:,:-1,:] - u[:,1:,:] == 0.0)  # delta u minimization to prevent agressive changes in control actions
    
    state_lower_bound_penalty = state_weight*(y >= ymin_val)
    state_upper_bound_penalty = state_weight*(y <= ymax_val)

    u_lower_bound_penalty = u_bound_weight*(u >= umin_val)
    u_upper_bound_penalty = u_bound_weight*(u <= umax_val)

    constraints = [state_lower_bound_penalty, state_upper_bound_penalty, u_lower_bound_penalty, u_upper_bound_penalty]
    objectives = [action_loss, loss_l]

    dynamics_model.nsteps = nsteps
    components = [encode_sym, system]

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
        epoch_verbose=100,
        device=dev
    )
    
    # Train control policy
    best_model = trainer.train()
    
    # load best trained model
    trainer.model.load_state_dict(best_model)


    system.nsteps = nsteps
    trainer.badcount = 0


# In[30]:


def test(test_data):

    trajectories_sindy = test_data

    sindy_y = denormalize(trajectories_sindy['y'], ymean, ystd)
    sindy_u = denormalize(trajectories_sindy['U'], umean, ustd)

    y_min = denormalize(test_data["ymin"], ymean, ystd)
    y_max = denormalize(test_data["ymax"], ymean, ystd)

    u_min = denormalize(test_data["umin"], umean, ustd)
    u_max = denormalize(test_data["umax"], umean, ustd)
    
    u_loss = action_weight * torch.sum(torch.abs(sindy_u))

    y_lower = torch.sum(torch.abs(torch.max(sindy_y - y_max, torch.zeros(sindy_y.shape, device=dev))))
    y_upper = torch.sum(torch.abs(torch.min(sindy_y - y_min, torch.zeros(sindy_y.shape, device=dev))))

    u_lower = torch.sum(torch.abs(torch.max(sindy_u - u_max, torch.zeros(sindy_u.shape, device=dev))))
    u_upper = torch.sum(torch.abs(torch.min(sindy_u - u_min, torch.zeros(sindy_u.shape, device=dev))))
    

    return (u_loss + state_weight*y_lower +  state_weight*y_upper + 10*u_lower + 10*u_upper).item()


# In[31]:


torch.manual_seed(0)

n_samples = 3000    # number of sampled scenarios
batch_size = 200
xmin_range = torch.distributions.Uniform(18., 22.)

max_degree = 1
max_freq = 3
sparsity = 0
p = 1
nref = ny
theta_1 = library.FourierLibrary(ny, 2*nref+1, max_freq=max_freq, include_cos=False)
theta_2 = library.PolynomialLibrary(ny, 2*nref+1, max_degree=max_degree, interaction=False)
names = ['1', 'y0', 'y1', 'y2', 'y3', 'y4', 'y5',
        'y0_min', 'y1_min', 'y2_min', 'y3_min', 'y4_min', 'y5_min',
        'y0_max', 'y1_max', 'y2_max', 'y3_max', 'y4_max', 'y5_max',
         'd']
theta_2.function_names = names
poly_sindy = sindy.SINDy(theta_2, n_out=nu)  
four_sindy = sindy.SINDy(theta_1, n_out=nu)  


# In[32]:


torch.cuda.empty_cache()


# In[37]:


nsteps = 80
policy_node = Node(poly_sindy,  ['y', 'R'], ['U']).to(dev)
combined_y = Node(lambda ymin, ymax, d: torch.cat([ymin, ymax, d], dim=-1),
              ['ymin', 'ymax', 'd_obs'], ['R'], name="y")
dist_obs = Node(lambda d: d[:, sys.d_idx], ['D'], ['d_obs'])
poly_system = System([dist_obs, combined_y, policy_node, combined, model, decode_sym],
                nsteps=nsteps,
                name='sindy_system').to(dev)
optimizer = torch.optim.AdamW(poly_sindy.parameters(), lr=.05)


# In[38]:


train_func(optimizer, poly_system, sparsity)


# In[50]:


policy_node = Node(poly_sindy,  ['y', 'R'], ['u_un']).to(dev)
u_bound_node = Node(lambda u: torch.clamp(u, umin, umax), ['u_un'], ['U'])

test_system = System([dist_obs, combined_y, policy_node, u_bound_node, combined, model, decode_sym],
                nsteps=nsteps_test,
                name='sindy_system').to(dev)
import time


# In[51]:

times_sindy = []
trajectories_sindy = None
for i in range(50):
    start = time.time()
    trajectories_sindy = test_system(data)
    end = time.time() - start
    times_sindy.append(end)

trajectories_sindy['times'] = times_sindy
# In[ ]:


nsteps = 100  # prediction horizon
n_samples = 3000    # number of sampled scenarios

net = blocks.MLP(insize=ny+2*nref+nd_obs, outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'])
policy = Node(net, ['y', 'ymin', 'ymax', 'd_obs'], ['U'], name='policy')


cl_system = System([dist_obs, policy, combined, model, decode_sym], nsteps=nsteps)

optimizer = torch.optim.AdamW(policy.parameters(), lr=0.006)
train_func(optimizer, cl_system, 0.0)

policy = Node(net, ['y', 'ymin', 'ymax', 'd_obs'], ['u_un'], name='policy')
cl_system = System([dist_obs, policy, u_bound_node, combined, model, decode_sym], nsteps=nsteps_test)

trajectories_nn = None
times_nn = []
for i in range(50):
    start = time.time()
    trajectories_nn = cl_system(data)
    end = time.time() - start
    times_nn.append(time)

trajectories_nn['times'] = times_nn

traj = dict()
traj['nn'] = trajectories_nn
traj['sindy'] = trajectories_sindy


# In[1]:


import pickle

with open('reno-bb.pyc', 'ab') as outfile:
    pickle.dump(traj, outfile)
