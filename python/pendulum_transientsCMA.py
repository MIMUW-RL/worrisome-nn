import torch
import numpy as np
from multiprocessing.pool import Pool
import pendulum_symbolic1
import pendulum_symbolic2
import symbolic_simple2
import pandas as pd
from collections import OrderedDict
import cma
import gym
import sys

# (improved) pendulum model
sys.path.append("../environment")
import pendulum

LEN = 1000
gym.register(
    id="Pendulum-v1",
    entry_point="pendulum:PendulumEnv",
    max_episode_steps=LEN,
)

env = gym.make("Pendulum-v1")


def integrate(statei):
    env.reset()
    env.set_state(statei)
    s = env.get_obs()
    totrew = 0
    for i in range(LEN):
        ua = u(torch.unsqueeze(torch.tensor(s), dim=0))
        # ua = u(torch.tensor(s))
        ua = u.unscale_action(ua.detach().numpy())
        s, r, done, _ = env.step(ua)
        totrew += r
        if done:
            break
    return totrew


# controllern = "symbolic_complex7A_cma_pendulum_fav_model.pt"
# u = torch.load(f"pendulum/Symbolic_cma_finetune/{controllern}").cpu()
# controllern = "symbolic_complex7A_final_model.pt"
# u = torch.load(f"pendulum/Symbolic_simple2_finetune/{controllern}").cpu()
# controllern = "OriginalNN_ReLU400300"
name = sys.argv[1]
controllern = name.split("/")[-1]
u = torch.load(name).cpu()
u.eval()

# the dictionary exported to csv
df = {
    "file_name": [],
    "formula": [],
    "h": [],
    "explicit": [],
    "ep_len": [],
    "reward": [],
    "init_omega": [],
    "init_theta": [],
}


# implement CMA-ES for finding transient solutions (achieving high rewards)

found_transients = {}

opts = cma.CMAOptions()
opts.set("bounds", [[None, -8.0], [None, 8.0]])
opts.set("maxfevals", 1500)


RESTART = 10

steps = [0.05] * RESTART + [0.025] * RESTART + [0.0125] * RESTART
for e in [False, True]:
    for H in steps:
        env.set_explicit(e)
        env.set_dt(H)
        print(env.explicit)
        x0 = np.random.uniform([-np.pi / 2, -8 / 2], [np.pi / 2, 8 / 2])
        sigma0 = 1
        x, es = cma.fmin2(integrate, x0, sigma0, opts)
        print(es.result_pretty())
        # save best in csv
        df["file_name"].append(controllern)
        try:
            df["formula"].append(u.to_str())
        except AttributeError:
            df["formula"].append(controllern)
        df["h"].append(H)
        df["explicit"].append(env.explicit)
        df["ep_len"].append(LEN)
        df["reward"].append(es.result.fbest)
        df["init_omega"].append(es.result.xbest[0])
        df["init_theta"].append(es.result.xbest[1])
df = pd.DataFrame(data=df)
df = df.sort_values(["explicit", "h", "reward"], ascending=True).groupby(["explicit", "h"]).head()
df = df.sort_values(["explicit", "h"], ascending=False)
df.to_csv(f"{controllern.split('.')[0]}_transients_hof.csv")
