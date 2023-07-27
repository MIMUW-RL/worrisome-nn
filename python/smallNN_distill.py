# flake8: noqa: E811
import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gym
import numpy as np
import torch as th
from torch import nn
from math import pi
import itertools
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
import pickle as pkl

actordir = str(sys.argv[1])
model = th.load(actordir + "td3_actor_400_300_ReLU.pth").cpu()
# model = th.load(actordir + f"final_model.pt").cpu()
model.eval()

print(model.mu)
print(model.type)


if "pendulum" in actordir:
    from pendulum_symbolic1 import SmallNet

    net = SmallNet()

    optimizer = th.optim.Adam(net.parameters(), lr=0.0001)

    def unscale_action(scaled_action: np.ndarray) -> np.ndarray:
        low, high = -2.0, 2.0
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    # pendulum evaluate the controller on the grid
    N = 50
    MV_TR = 8
    s = np.arange(-pi, pi, (2 * pi) / N)
    Nx = s.shape[0]
    v = np.arange(-MV_TR, MV_TR, (2 * MV_TR) / (2 * N))
    Ny = v.shape[0]
    s1 = np.reshape(np.cos(s), (-1, 1))
    s2 = np.reshape(np.sin(s), (-1, 1))
    se = np.hstack([s1, s2])
    xnp = np.array([np.append(x[0], x[1]) for x in itertools.product(se, v)])
    x = th.Tensor(xnp).cpu()
    y = model(x)
    ynp = y.detach().numpy()
    ynp = unscale_action(ynp)

    np.savetxt(actordir + "x.csv", xnp, delimiter=",")
    np.savetxt(actordir + "y.csv", ynp, delimiter=",")


if "cartpole" in actordir:
    from cartpole_swingup_symbolic2 import SmallNet

    net = SmallNet()

    optimizer = th.optim.Adam(net.parameters(), lr=0.0001)

    def unscale_action(scaled_action: np.ndarray) -> np.ndarray:
        low, high = -1.0, 1.0
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    # cartpole load data from the replay buffer
    datasize = 5e04
    with open(actordir + "replay_buffer.pkl", "rb") as f:
        buffer = pkl.load(f)
    data = buffer.sample(int(datasize))
    obs = data[0]
    xnp = th.Tensor(obs).cpu()
    x = th.Tensor(xnp).cpu()
    y = model(xnp)
    ynp = y.detach().numpy()
    ynp = unscale_action(ynp)
    np.savetxt(actordir + "x.csv", xnp, delimiter=",")
    np.savetxt(actordir + "y.csv", ynp, delimiter=",")


ITERS = 25000


loss = nn.MSELoss()

Y = th.Tensor(ynp)
bestloss = np.inf
bestnet = None
for i in range(ITERS):
    yhat = net(x)
    yhat = net.unscale_action(yhat)
    l = loss(yhat, Y)
    l.backward()
    optimizer.step()
    print(l.detach())
    if l.detach() < bestloss:
        bestnet = net
        bestloss = l.detach()
        print(f"best net found (l={bestloss})")


# save the trained net
net = bestnet
yhat = net(x)
yhat = net.unscale_action(yhat)
th.save(net, f"{net.name}.pt")
np.savetxt("yhat.csv", yhat.detach().numpy(), delimiter=",")

if "pendulum" in actordir:
    yhat = np.reshape(yhat.detach().numpy(), (Nx, Ny))
    fig, ax = plt.subplots()
    ax.imshow(yhat, cmap="bwr", interpolation="nearest", extent=[-np.pi, np.pi, -MV_TR, MV_TR])
    plt.savefig("y.png", dpi=300)
