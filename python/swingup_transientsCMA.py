import torch
import numpy as np
from multiprocessing.pool import Pool
import pendulum_symbolic1
import symbolic_simple2
import pandas as pd
from collections import OrderedDict
import cma
import sys
from gym import register
import gym

# swing-up cart-pole model
sys.path.append("../environment")
import cartpole_swingup_modif


STEPS = 2000

register(
    id="CartPoleSwingUp-v0",
    entry_point="cartpole_swingup_modif:CartPoleSwingUpV0",
    max_episode_steps=STEPS,
)

env = gym.make("CartPoleSwingUp-v0")
env.set_vel_penalty(0.5)
env.set_no_terminate(True)


def integrate(statei):
    env.reset()
    env.set_state(cartpole_swingup_modif.State(*statei))
    totrew = 0
    s = cartpole_swingup_modif.CartPoleSwingUpEnv._get_obs(env.state)
    for i in range(STEPS):
        ua = u(torch.unsqueeze(torch.tensor(s), dim=0))
        ua = u.unscale_action(ua.detach().numpy())
        s, r, done, _ = env.step(ua[0])
        totrew += r
        if done:
            break

    return totrew


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
    "reward": [],
    "init_xpos": [],
    "init_xdot": [],
    "init_theta": [],
    "init_thetadot": [],
}


# implement CMA-ES for finding transient solutions (achieving high rewards)

found_transients = {}

opts = cma.CMAOptions()
bound = 0.5
opts.set("bounds", [[-bound, -bound, np.pi - bound, -bound], [bound, bound, np.pi + bound, bound]])
# opts.set("bounds", [[-bound, -bound, -bound, -bound], [bound, bound, bound, bound]])
opts.set("maxfevals", 1000)

RESTART = 10

steps = [0.01] * RESTART + [0.005] * RESTART + [0.0025] * RESTART
for e in [True, False]:
    for H in steps:
        env.set_explicit(e)
        env.set_dt(H)
        print(env.explicit)
        x0 = np.random.uniform([-bound, -bound, np.pi - bound, -bound], [bound, bound, np.pi + bound, bound])
        # x0 = np.random.uniform([-bound, -bound, -bound, -bound], [bound, bound, bound, bound])
        sigma0 = 0.2
        x, es = cma.fmin2(integrate, x0, sigma0, opts)
        print(es.result_pretty())
        # sample around the best sol
        xbest = es.result.xbest
        xbestorig = xbest
        fbest = es.result.fbest
        # h = 0.01
        # for i in range(1000):
        #    sx = xbest + np.random.uniform([-h] * 4, [h] * 4)
        #    sr = integrate(sx)
        #    if sr < fbest:
        #        print(f"improved sampled x {xbest} ({fbest})")
        #        xbest = sx
        #        fbest = sr

        # save best in csv
        df["file_name"].append(controllern)
        try:
            df["formula"].append(u.to_str())
        except AttributeError:
            df["formula"].append(controllern)
        df["h"].append(H)
        df["explicit"].append(env.explicit)
        df["reward"].append(fbest)
        df["init_xpos"].append(xbest[0])
        df["init_xdot"].append(xbest[1])
        df["init_theta"].append(xbest[2])
        df["init_thetadot"].append(xbest[3])

df = pd.DataFrame(data=df)

df = df.sort_values(["explicit", "h", "reward"], ascending=True).groupby(["explicit", "h"]).head()
df = df.sort_values(["explicit", "h"], ascending=False)
df.to_csv(f"{controllern.split('.')[0]}_transients_hof.csv")
