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
    env.set_no_terminate(False)
    BASEDT = 0.01
    BASEEPISODE = 500
    BREAKAFTERDONE = True

elif "cartpole" in dir:
    envname = "CartPoleContinuous-v0"
    env = gym.make(envname)
    BASEDT = 0.02
    BASEEPISODE = 1000
    BREAKAFTERDONE = True

episodes = 100

df = {
    "file_name": [],
    "formula": [],
    "parametrized_formula": [],
    "reward_mean": [],
    "reward_std": [],
    "reward16_mean": [],
    "reward16_std": [],
    "reward_halfdt_mean": [],
    "reward_halfdt_std": [],
    "explicit_reward_mean": [],
    "explicit_reward_std": [],
    "explicit_reward16_mean": [],
    "explicit_reward16_std": [],
    "explicit_reward_halfdt_mean": [],
    "explicit_reward_halfdt_std": [],
    "imexplicit_rew_discrep_mean": [],
    "imexplicit_rew_discrep_std": [],
    "prec_discrep_mean": [],
    "prec_discrep_std": [],
    "prec_exp_discrep_mean": [],
    "prec_exp_discrep_std": [],
}


for n, model in enumerate(models):

    observed_vel = []
    rewards = []
    rewards025 = []
    rewards16 = []
    explicit_rewards = []
    explicit_rewards025 = []
    explicit_rewards16 = []

    imexplicit_rew_discrep = []

    prec_discrep = []
    prec_exp_discrep = []

    for e in range(episodes):

        env.set_dt(BASEDT)
        env._max_episode_steps = BASEEPISODE
        env.set_explicit(False)
        env.set_precision(np.float64)
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
            if "SmallNet" in filenss[n]:
                act = act[0]
            (obs, r, done, _) = env.step(act)

            observed_vel.append(obs[2])
            tot_r += r
            i += 1
        print(tot_r)
        rewards.append(tot_r)

        env.set_precision(np.float16)
        obs = env.reset()
        env.set_state(state)
        obs = initobs

        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in filenss[n]:
                act = act[0]
            (obs, r, done, _) = env.step(act)

            observed_vel.append(obs[2])
            tot_r += r
            i += 1
        print(tot_r)
        rewards16.append(tot_r)

        env.set_dt(BASEDT / 2.0)
        env._max_episode_steps = BASEEPISODE * 2
        env.set_precision(np.float64)

        obs = env.reset()
        env.set_state(state)
        obs = initobs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in filenss[n]:
                act = act[0]
            (obs, r, done, _) = env.step(act)
            tot_r += r
            i += 1
        print(tot_r)
        rewards025.append(tot_r)

        # explicit Euler rewards
        env.set_explicit(True)
        env.set_dt(BASEDT)
        env._max_episode_steps = BASEEPISODE
        env.set_precision(np.float64)

        env.reset()
        env.set_state(state)
        obs = initobs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in filenss[n]:
                act = act[0]
            (obs, r, done, _) = env.step(act)
            tot_r += r
            i += 1
        print(tot_r)
        explicit_rewards.append(tot_r)

        env.set_precision(np.float16)
        env.reset()
        env.set_state(state)
        obs = initobs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in filenss[n]:
                act = act[0]
            (obs, r, done, _) = env.step(act)
            tot_r += r
            i += 1
        print(tot_r)
        explicit_rewards16.append(tot_r)

        env.set_dt(BASEDT / 2.0)
        env._max_episode_steps = BASEEPISODE * 2
        env.set_precision(np.float64)

        env.reset()
        env.set_state(state)
        obs = initobs
        print(f"{model.__class__.__name__}, state={env.state}")
        i = 0
        tot_r = 0
        done = False
        while not done:
            act = model(th.unsqueeze(th.Tensor(obs), 0))
            act = model.unscale_action(act.detach().numpy())
            if "SmallNet" in filenss[n]:
                act = act[0]
            (obs, r, done, _) = env.step(act)
            tot_r += r
            i += 1
        print(tot_r)
        explicit_rewards025.append(tot_r)

        print(f"imexplicit_rew_discrep={np.abs(rewards[-1] - explicit_rewards[-1])}")
        imexplicit_rew_discrep.append(np.abs(rewards[-1] - explicit_rewards[-1]))

        print(f"prec_discrep={np.abs(rewards[-1] - rewards16[-1])}")
        prec_discrep.append(np.abs(rewards[-1] - rewards16[-1]))
        print(f"prec_exp_discrep={np.abs(explicit_rewards[-1] - explicit_rewards16[-1])}")
        prec_exp_discrep.append(np.abs(explicit_rewards[-1] - explicit_rewards16[-1]))

    df["file_name"].append(filenss[n])
    try:
        df["formula"].append(model.to_str())
    except AttributeError:
        df["formula"].append(model.__class__.__name__)
    try:
        df["parametrized_formula"].append(model.to_str_raw())
    except AttributeError:
        df["parametrized_formula"].append(model.__class__.__name__)

    df["reward_mean"].append(np.mean(rewards))
    df["reward_std"].append(np.std(rewards))
    df["explicit_reward_mean"].append(np.mean(explicit_rewards))
    df["explicit_reward_std"].append(np.std(explicit_rewards))

    df["reward16_mean"].append(np.mean(rewards16))
    df["reward16_std"].append(np.std(rewards16))

    df["imexplicit_rew_discrep_mean"].append(np.mean(imexplicit_rew_discrep))
    df["imexplicit_rew_discrep_std"].append(np.std(imexplicit_rew_discrep))

    df["reward_halfdt_mean"].append(np.mean(rewards025))
    df["reward_halfdt_std"].append(np.std(rewards025))
    df["explicit_reward_halfdt_mean"].append(np.mean(explicit_rewards025))
    df["explicit_reward_halfdt_std"].append(np.std(explicit_rewards025))

    df["explicit_reward16_mean"].append(np.mean(explicit_rewards16))
    df["explicit_reward16_std"].append(np.std(explicit_rewards16))

    df["prec_discrep_mean"].append(np.mean(prec_discrep))
    df["prec_discrep_std"].append(np.std(prec_discrep))

    df["prec_exp_discrep_mean"].append(np.mean(prec_exp_discrep))
    df["prec_exp_discrep_std"].append(np.std(prec_exp_discrep))

print(df)
df = pd.DataFrame(data=df)
df.to_csv(f"{dir}{envname}_results_precisions.csv")
