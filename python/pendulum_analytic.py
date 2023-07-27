import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import copy
import control_ai_manifolds.python.pendulum.pendulum_symbolic2 as pendulum_symbolic2
import control_ai_manifolds.python.pendulum.pendulum_symbolic1 as pendulum_symbolic1


def obs(s):
    theta, thetadot = s[0], s[1]
    return torch.Tensor([torch.cos(theta), torch.sin(theta), thetadot])


MV_TR = 8


def pendulum(s, a, h):
    r = torch.zeros((2,))
    a = torch.clamp(a, -2.0, 2.0)
    newthdot = s[1] + h * (3.0 * a + 30.0 / 2.0 * torch.sin(s[0]))
    newthdot = torch.clamp(newthdot, -MV_TR, MV_TR)
    newth = s[0] + h * newthdot
    r[1] = newthdot
    r[0] = newth
    return r


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


def r(s, a):
    th, thdot = s[0], s[1]
    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (a**2)
    return costs


u = pendulum_symbolic1.Symbolic8A()
# torch.save(u, f"{u.name}.pt")
# exit(1)
optimizer = torch.optim.Adam(u.parameters(), lr=0.001)
h = 0.05
BTCH = 1
LEN = 30
STEPS = 5000
EVAL = 50
EVAL_LIM = -1

evals = []
param_traject = []
rewards = []
best_rew = np.inf
for i in range(STEPS + 1):
    # print(f"gradient step {i}")
    optimizer.zero_grad()
    totrew = 0
    for b in range(BTCH):
        s = torch.rand((2,))
        s[0] = s[0] * 2 * torch.pi
        s[0] -= torch.pi
        s[1] = s[1] * 16
        s[1] -= 8
        for j in range(LEN):
            ua = u(torch.unsqueeze(obs(s), dim=0))
            ua = u.unscale_action(ua)
            s = pendulum(s, ua, h)
            totrew += r(s, ua)
    totrew /= BTCH
    totrew.backward()
    optimizer.step()
    param_traject.append(u.p.detach().numpy().copy())

    if i <= EVAL_LIM:
        print(f"current pi evaluation at {i}")
        with torch.no_grad():
            rews = []
            for e in range(10):
                totrew = 0.0
                high = np.array([np.pi, 1])
                s = np.random.uniform(low=-high, high=high)
                s = torch.tensor(s)
                for j in range(200):
                    ua = u(torch.unsqueeze(obs(s), dim=0))
                    ua = u.unscale_action(ua)
                    s = pendulum(s, ua, h)
                    re = r(s, ua)
                    totrew += re
                cur_rew = totrew.cpu().numpy()
                rews.append(cur_rew)

            print(np.mean(rews))
            if np.mean(rews) < best_rew:
                print(f"current best reward {np.mean(rews)}, saving model")
                torch.save(u, f"{u.name}_best_model.pt")
                best_rew = np.mean(rews)
            evals.append(np.mean(rews))
        rewards.append(np.mean(rews))
    if i > EVAL_LIM:
        if (i % EVAL == 0) and (i > 0):
            print(f"current pi evaluation at {i}")
            with torch.no_grad():
                rews = []
                for e in range(10):
                    totrew = 0.0
                    high = np.array([np.pi, 1])
                    s = np.random.uniform(low=-high, high=high)
                    s = torch.tensor(s)
                    for j in range(200):
                        ua = u(torch.unsqueeze(obs(s), dim=0))
                        ua = u.unscale_action(ua)
                        s = pendulum(s, ua, h)
                        re = r(s, ua)
                        totrew += re
                    cur_rew = totrew.cpu().numpy()
                    rews.append(cur_rew)

                print(np.mean(rews))
                if np.mean(rews) < best_rew:
                    print(f"current best reward {np.mean(rews)}, saving model")
                    torch.save(u, f"{u.name}_best_model.pt")
                    best_rew = np.mean(rews)
                evals.append(np.mean(rews))
            rewards.append(np.mean(rews))
        else:
            rewards.append(0)

param_traject = np.vstack(param_traject)
rew_traject = np.vstack(rewards)
full_traject = np.hstack([param_traject, rew_traject])

hdr = ""
for i in range(param_traject.shape[1]):
    hdr += f"p{i},"
hdr += "reward"
np.savetxt(f"{u.name}_param_traject.csv", full_traject, delimiter=",", header=hdr)

for i in range(param_traject.shape[1]):
    plt.plot(param_traject[:, i])

plt.savefig(f"{u.name}_param_traject.png")
print(u.p.detach())
torch.save(u, f"{u.name}_final_model.pt")

plt.clf()
plt.plot(evals)
# plt.show()
plt.savefig(f"{u.name}_analytic_training.png")

EP = 10
with torch.no_grad():
    rewards = []
    for i in range(EP):
        totrew = 0
        high = np.array([np.pi, 1])
        s = np.random.uniform(low=-high, high=high)
        s = torch.tensor(s)
        for j in range(200):
            ua = u(torch.unsqueeze(obs(s), dim=0))
            ua = u.unscale_action(ua)
            s = pendulum(s, ua, h)
            re = r(s, ua)
            totrew += re
        rewards.append(totrew.cpu().numpy())
    print(f"final model eval ({EP} episodes)={np.mean(rewards)}")
