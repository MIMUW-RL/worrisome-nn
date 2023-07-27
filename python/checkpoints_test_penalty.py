from asyncio import constants
import sympy
import torch as th
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
import pandas as pd
from os import listdir
from numpy import log, abs
from math import pi
import itertools
from gym import register
import cartpole_swingup_symbolic2
import pendulum_symbolic2
import scipy.stats as st


sys.path.append("../environment")


register(
    id="CartPoleSwingUp-v0",
    entry_point="cartpole_swingup_modif:CartPoleSwingUpV0",
    max_episode_steps=500,
)

dir = str(sys.argv[1])

filens = listdir(dir)
print(filens)
filen = None

models = []
filenss = []
for fn in filens:
    if ".pt" in fn[-4:]:
        model = th.load(dir + fn).cpu()
        model.eval()
        models.append(model)
        filenss.append(fn)

if "pendulum" in dir:
    BASEEPISODE = 200
    gym.register(
        id="Pendulum-v1",
        entry_point="pendulum:PendulumEnv",
        max_episode_steps=BASEEPISODE,
    )
    envname = "Pendulum-v1"
    env = gym.make(envname)
    BASEDT = 0.05

    BREAKAFTERDONE = False

elif "swingup" in dir:
    envname = "CartPoleSwingUp-v0"
    env = gym.make(envname)
    env.set_no_terminate(True)
    BASEDT = 0.01
    BASEEPISODE = 1000
    BREAKAFTERDONE = True


episodes = 100

df = {
    "file_name": [],
    "formula": [],
    "parametrized_formula": [],
    "penalty05_mean": [],
    "penalty05_std": [],
    "penalty01_mean": [],
    "penalty01_std": [],
    "penalty1_mean": [],
    "penalty1_std": [],
    "penalty05_explicit_mean": [],
    "penalty05_explicit_std": [],
    "penalty01_explicit_mean": [],
    "penalty01_explicit_std": [],
    "penalty1_explicit_mean": [],
    "penalty1_explicit_std": [],
}


for n, model in enumerate(models):

    penalties05 = []
    penalties01 = []
    penalties1 = []
    penalties05_explicit = []
    penalties01_explicit = []
    penalties1_explicit = []

    for e in range(episodes):

        env.set_dt(BASEDT)
        env._max_episode_steps = BASEEPISODE
        env.set_explicit(False)
        env.set_precision(np.float64)
        env.set_vel_penalty(0.5)
        print(env._max_episode_steps)
        obs = env.reset()
        state = env.state
        initobs = obs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in dir:
                act = act[0]
            (obs, r, done, _) = env.step(act)

            tot_r += r
            i += 1
        print(tot_r)
        penalties05.append(tot_r)

        env.set_explicit(True)
        env.set_precision(np.float64)
        env.set_vel_penalty(0.5)
        print(env._max_episode_steps)
        obs = env.reset()
        state = env.state
        initobs = obs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in dir:
                act = act[0]
            (obs, r, done, _) = env.step(act)

            tot_r += r
            i += 1
        print(tot_r)
        penalties05_explicit.append(tot_r)

        env.set_dt(BASEDT)
        env._max_episode_steps = BASEEPISODE
        env.set_explicit(False)
        env.set_precision(np.float64)
        env.set_vel_penalty(0.1)
        print(env._max_episode_steps)
        obs = env.reset()
        state = env.state
        initobs = obs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in dir:
                act = act[0]
            (obs, r, done, _) = env.step(act)

            tot_r += r
            i += 1
        print(tot_r)
        penalties01.append(tot_r)

        env.set_explicit(True)
        env.set_precision(np.float64)
        env.set_vel_penalty(0.1)
        print(env._max_episode_steps)
        obs = env.reset()
        state = env.state
        initobs = obs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in dir:
                act = act[0]
            (obs, r, done, _) = env.step(act)

            tot_r += r
            i += 1
        print(tot_r)
        penalties01_explicit.append(tot_r)

        env.set_dt(BASEDT)
        env._max_episode_steps = BASEEPISODE
        env.set_explicit(False)
        env.set_precision(np.float64)
        env.set_vel_penalty(1.0)
        print(env._max_episode_steps)
        obs = env.reset()
        state = env.state
        initobs = obs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in dir:
                act = act[0]
            (obs, r, done, _) = env.step(act)

            tot_r += r
            i += 1
        print(tot_r)
        penalties1.append(tot_r)

        env.set_explicit(True)
        env.set_precision(np.float64)
        env.set_vel_penalty(1.0)
        print(env._max_episode_steps)
        obs = env.reset()
        state = env.state
        initobs = obs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in dir:
                act = act[0]
            (obs, r, done, _) = env.step(act)

            tot_r += r
            i += 1
        print(tot_r)
        penalties1_explicit.append(tot_r)

    df["file_name"].append(filenss[n])
    try:
        df["formula"].append(model.to_str())
    except AttributeError:
        df["formula"].append(model.__class__.__name__)
    try:
        df["parametrized_formula"].append(model.to_str_raw())
    except AttributeError:
        df["parametrized_formula"].append(model.__class__.__name__)

    df["penalty05_mean"].append(np.mean(penalties05))
    df["penalty05_std"].append(np.std(penalties05))
    df["penalty05_explicit_mean"].append(np.mean(penalties05_explicit))
    df["penalty05_explicit_std"].append(np.std(penalties05_explicit))
    df["penalty01_mean"].append(np.mean(penalties01))
    df["penalty01_std"].append(np.std(penalties01))
    df["penalty01_explicit_mean"].append(np.mean(penalties01_explicit))
    df["penalty01_explicit_std"].append(np.std(penalties01_explicit))
    df["penalty1_mean"].append(np.mean(penalties1))
    df["penalty1_std"].append(np.std(penalties1))
    df["penalty1_explicit_mean"].append(np.mean(penalties1_explicit))
    df["penalty1_explicit_std"].append(np.std(penalties1_explicit))

print(df)
df = pd.DataFrame(data=df)
df.to_csv(f"{dir}{envname}_penalties.csv")
