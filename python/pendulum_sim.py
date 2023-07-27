import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from math import pi
import symbolic_simple2
import sys
import pandas as pd

# (improved) pendulum model
sys.path.append("../environment")
import pendulum


gym.register(
    id="Pendulum-v1",
    entry_point="pendulum:PendulumEnv",
    max_episode_steps=1000,
)

model = torch.load("pendulum/Symbolic_finetune_CMA/symbolic_complex7A_cma_pendulum_fav_model.pt").cpu()
model.eval()

print(model)

MV_TR = 8

print(model.to_str())

traj = {"obs_0": [], "obs_1": [], "obs_2": [], "a": [], "r": []}


def save_frames_as_gif(frames, path="./", filename="gym_animation.gif"):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, fps=60)


env = gym.make("Pendulum-v1")

env.set_precision(np.float16)
obs = env.reset()
theta = pi / 2 - 1e-10
omega = 5.0
theta, omega = [np.random.uniform(0, 1) * 2 * pi - pi, np.random.uniform(0, 1) * 2 - 1]

theta, omega = [0.856278697567971, -7.83273495109455]
print(theta, omega)

env.set_state(np.array([theta, omega]))
obs[0] = np.cos(theta)
obs[1] = np.sin(theta)
obs[2] = omega


env.set_dt(0.05)
env.set_explicit(False)

print(env.dt)


def control(obs):
    c = model(torch.tensor(np.reshape(obs, (1, -1))))
    return model.unscale_action(c.detach().numpy())


i = 0

tot_r = 0

frames = []
rewards = []
print(f"obs_0={obs}")
print(obs.dtype)

while i < 1000:

    act = control(obs)
    traj["obs_0"].append(obs[0])
    traj["obs_1"].append(obs[1])
    traj["obs_2"].append(obs[2])
    traj["a"].append(act[0])

    (obs, r, _, _) = env.step(act)
    # print(f"a_{i}={act}", f"o_{i+1}={obs}", f"r_{i+1}={r}")
    traj["r"].append(r)
    rewards.append(r)
    tot_r += r
    i += 1
    # frames.append(env.render(mode="rgb_array"))
    # env.render()

df = pd.DataFrame(traj)
df.to_csv("traj.csv", header=False)

env.close()
print(tot_r)
# save_frames_as_gif(frames, filename = 'proper_control.gif')
