import sympy
import torch as th
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
import pandas as pd
from os import listdir
from numpy import log, abs
import pickle
import gym_cartpole_swingup

dir = str(sys.argv[1])

filens = listdir(dir)
filen = None

for fn in filens:
    if ("hall_of_fame" in fn) and ("lock" not in fn) and (".csv" in fn):
        csvfilen = fn

hof = pd.read_csv(dir + csvfilen, delimiter=",")

if "pendulum" in dir:
    env = gym.make("Pendulum-v1")
elif "swingup" in dir:
    env = gym.make("CartPoleSwingUp-v0")
elif "cartpole" in dir:
    env = gym.make("CartPoleContinuous-v0")

results = []
episodes = 100

# check the reward of the original controller
model = th.load(dir + "actor.pth").cpu()
model.eval()
print("original model params:")
print(model.mu)
print(model.type)
rewards = []
observed_vel = []
for e in range(episodes):
    obs = env.reset()
    i = 0
    tot_r = 0
    done = False
    while not done:
        act = model(th.unsqueeze(th.Tensor(obs), 0))
        # act = model( th.Tensor(obs) )
        act = model.unscale_action(act.detach().numpy())
        (obs, r, done, _) = env.step(act)
        observed_vel.append(obs[2])
        tot_r += r
        i += 1
        # frames.append(env.render(mode="rgb_array"))
        # env.render()
    print(tot_r)
    rewards.append(tot_r)
print(np.min(observed_vel), np.max(observed_vel), np.mean(observed_vel), np.std(observed_vel))
results.append((0, "NN", np.mean(rewards), np.std(rewards)))
np.savetxt(f"{dir[:-1]}_rewards.csv", np.array(results), fmt="%s, %s, %.16s, %.16s")
print(("NN", np.mean(rewards), np.std(rewards)))


Maxsymbols = 5


def log_abs(x):
    return log(abs(x))


def inv(x):
    return 1.0 / x


for r in range(hof.shape[0]):

    eq = hof.iloc[r]["Equation"]
    c = hof.iloc[r]["Complexity"]
    rewards = []
    pysympy = sympy.sympify(eq)
    print(pysympy)

    symb = ""
    for i in range(Maxsymbols):
        s = f"x{i}"
        # if s in eq:
        if symb == "":
            symb += s
        else:
            symb += f", {s}"

    fvars = sympy.symbols(symb)

    lambd = sympy.lambdify(fvars, pysympy, modules=[{"log_abs": log_abs, "inv": inv}, "numpy"])

    for e in range(episodes):
        obs = env.reset()
        i = 0
        tot_r = 0
        done = False
        while not done:

            act = np.array([lambd(*obs)])
            # print(act)
            (obs, r, done, _) = env.step(act.astype(float))
            # print(obs)
            tot_r += r
            i += 1
            # frames.append(env.render(mode="rgb_array"))
            # env.render()
        print(tot_r)
        rewards.append(tot_r)
    results.append((c, eq, np.mean(rewards), np.std(rewards)))
    np.savetxt(f"{dir}rewards.csv", np.array(results), fmt="%s, %s, %.16s, %.16s")
    print((eq, np.mean(rewards), np.std(rewards)))

cs = [int(t[0]) for t in results]
print(cs)
cns = [t[1] for t in results]
means = [t[2] for t in results]
stds = [t[3] for t in results]


colors = []


colors.append("red")
for i in range(1, len(cs)):
    if (means[i] > means[0]) or (means[i] - stds[i] > means[0] - stds[0]):
        colors.append("blue")
    else:
        colors.append("lightblue")


fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.barh(cs, means, xerr=stds, align="center", alpha=0.5, ecolor="black", capsize=2, color=colors)
ax.yaxis.tick_right()
ax.set_aspect("auto")
ax.set_yticks(cs)
cs_labels = np.array(cs).astype(str)
cs_labels[0] = "NN"
ax.set_yticklabels(cs_labels)
ax.yaxis.grid(True)


# Save the figure and show
plt.savefig(f"{dir}rewards.png", dpi=300)
plt.show()


env.close()
