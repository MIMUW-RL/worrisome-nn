import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import copy
from dataclasses import dataclass, field
from collections import namedtuple
import cma
import argparse
import symbolic_simple2
import gym
import sys

sys.path.append("../environment")

gym.register(
    id="Pendulum-v1",
    entry_point="pendulum:PendulumEnv",
    max_episode_steps=200,
)


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--control")
parser.add_argument("-r", "--resume", required=False)
parser.add_argument("-i", "--iter", required=False, default=1500)
parser.add_argument("-s", "--sigma", required=False, default=2.5, type=float)
parser.add_argument("-b", "--batch", required=False, default=10, type=int)
parser.add_argument("-p", "--popsize", required=False, default=32, type=int)
args = parser.parse_args()


BATCHSIZE = args.batch
h = 0.05

device = torch.device("cpu")

u = eval("symbolic_simple2.{}()".format(args.control)).to(device)
print(u)
u.eval()
# torch.save(u, "symbolic_" + args.control + "_pendulum_base.pt")

if args.resume:
    u = torch.load(args.resume)
    u.eval()


def integrate(params):
    rews = []
    env = gym.make("Pendulum-v1")
    env.set_precision(np.float64)
    u.p = torch.nn.Parameter(torch.tensor(params).to(device), requires_grad=False)

    for b in range(BATCHSIZE):
        env.reset()
        s = env.get_obs()
        totrew = 0
        done = False
        i = 0
        while not done:
            act = u(torch.unsqueeze(torch.Tensor(s), 0))
            act = u.unscale_action(act.detach().numpy())
            s, r, done, _ = env.step(act)
            totrew += r
            i += 1
        rews.append(-totrew)
    return np.mean(rews)


params0 = u.p.detach().numpy()
print(f"params0 {params0}")
sigma0 = args.sigma

x, es = cma.fmin2(integrate, params0, sigma0, options={"maxfevals": args.iter, "popsize": args.popsize})


u.p = torch.nn.Parameter(torch.tensor(es.result.xbest).to(device), requires_grad=False)

env = gym.make("Pendulum-v1")
rews = []
for b in range(100):
    env.reset()
    s = env.get_obs()
    totrew = 0
    done = False
    while not done:
        act = u(torch.unsqueeze(torch.Tensor(s), 0))
        act = u.unscale_action(act.detach().numpy())
        s, r, done, _ = env.step(act)
        totrew += r
    rews.append(totrew)
print(np.mean(rews), np.std(rews))

print(f"saving model that achieved f={es.result.fbest} ({es.result.xbest} {next(u.parameters()).device}")
if args.resume:
    torch.save(u, f"{u.name}_cma_pendulum_retrained.pt")
else:
    torch.save(u, f"{u.name}_cma_pendulum_final_model.pt")

u = torch.load(f"{u.name}_cma_pendulum_fav_model.pt").to(device)
print(f"loaded {u} {u.p.detach().numpy()} {next(u.parameters()).device}")
env = gym.make("Pendulum-v1")
rews = []
for b in range(100):
    env.reset()
    s = env.get_obs()
    totrew = 0
    done = False
    while not done:
        act = u(torch.unsqueeze(torch.Tensor(s), 0))
        act = u.unscale_action(act.detach().numpy())
        s, r, done, _ = env.step(act)
        totrew += r
    rews.append(totrew)
print(np.mean(rews), np.std(rews))

u.p = torch.nn.Parameter(torch.tensor(es.result.xfavorite), requires_grad=False)
print(f"saving favorite model ({es.result.xfavorite}")
if args.resume:
    torch.save(u, f"{u.name}_cma_pendulum_fav_retrained.pt")
else:
    torch.save(u, f"{u.name}_cma_pendulum_fav_model.pt")
