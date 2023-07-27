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
model = th.load(actordir + "actor.pth").cpu()
# model = th.load(actordir + f"final_model.pt").cpu()
model.eval()

print(model.mu)
"""
with open('model_data.txt', 'w') as f:
    f.write(model.mu)

for name, para in model.named_parameters():
    print('{}: {}'.format(name, para.shape))
    np.savetxt( f"{name}.csv", para.detach().numpy(), delimiter=',')

exit(1)
"""

if "pendulum" in actordir:

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
    y = model(xnp)
    ynp = y.detach().numpy()
    ynp = unscale_action(ynp)
    np.savetxt(actordir + "x.csv", xnp, delimiter=",")
    np.savetxt(actordir + "y.csv", ynp, delimiter=",")

ITERS = 100
pysr = PySRRegressor(
    niterations=ITERS,
    binary_operators=["+", "*", "div"],
    unary_operators=[],
    model_selection="accuracy",
    populations=150,
    procs=10,
    maxsize=30,
    population_size=50,
    #    batching = True,
    #    batch_size = 1000,
    loss="loss(x, y) = abs(x - y)",  # Custom loss function (julia syntax)
)

# call to the pysr symbolic regression
r = pysr.fit(xnp, ynp)


print(pysr)
print(pysr.sympy())
