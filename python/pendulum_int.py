from scipy.integrate import solve_ivp, RK45
import numpy as np
from numpy import arccos, cos, sin, abs, exp, tanh, log
import matplotlib.pyplot as plt
import random
from math import pi
import torch as th
from torch import nn
import torch.nn.functional as F

import symbolic_simple2

G = 10

MV_TR = 8

cname = "pendulum/Symbolic_finetune2/symbolic_complex9A_final_model.pt"
model = th.load(cname).cpu()
model.eval()

# print(model.mu)


def mv_trunc(y):
    if y[1] >= MV_TR:
        y[1] = MV_TR
    elif y[1] <= -MV_TR:
        y[1] = -MV_TR
    return y


def torque_trunc(y):
    if y >= 2:
        y = 2
    elif y <= -2:
        y = -2
    return y


def solve_semii_Euler(fcn, t_span, y0, h):
    y = y0
    N = int(t_span[1] / h)
    ts = [t_span[0] + x * h for x in range(N)]
    ys = [y0]
    for i in range(N):
        y1 = y.copy()
        y1[1] = (y + h * fcn(h, y))[1]

        y1[0] = (y + h * fcn(h, y1))[0]
        y = mv_trunc(y1)
        ys.append(y.copy())
    ys = np.vstack(ys).T
    print(ys.shape)
    return ys


def solve_i_Euler(fcn, t_span, y0, h):
    y = y0
    N = int(t_span[1] / h)
    ITR = 5
    ts = [t_span[0] + x * h for x in range(N)]
    ys = [y0]
    for i in range(N):
        y1 = y.copy()

        for j in range(ITR - 1):
            y1 = y + h * fcn(h, y1)

        y = mv_trunc(y1)
        ys.append(y1)
    ys = np.vstack(ys).T
    print(ys.shape)
    return ys


def solve_Euler(fcn, t_span, y0, h):
    y = y0
    N = int(t_span[1] / h)
    ts = [t_span[0] + x * h for x in range(N)]
    ys = [y0]
    for i in range(N):
        y = y + h * fcn(h, y)

        ys.append(mv_trunc(y.copy()))
        print(y)
    ys = np.vstack(ys).T
    print(ys.shape)
    return ys


def unscale_action(scaled_action):
    low = -2.0
    high = 2.0
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


def pendulum(t, y):
    dy = np.zeros((2,))
    y = mv_trunc(y)

    s1 = cos(y[0])
    s2 = sin(y[0])
    s3 = y[1]

    # torque = -2 * s2 - (8 * s2 + 2 * s3) / s1
    # s = th.unsqueeze(th.Tensor([s1, s2, s3]), dim=0)
    s = th.unsqueeze(th.Tensor([s1, s2, s3]), dim=0)
    ta = model(s)
    torque = ta.detach().numpy().squeeze()
    torque = model.unscale_action(torque)
    torque = torque_trunc(torque)
    print(f"torque={torque}")
    # x0 = s1
    # x1 = s2
    # x2 = s3

    dy[0] = y[1]
    dy[1] = 3.0 * torque + 3.0 / 2.0 * G * s2
    return dy


np.set_printoptions(16)

# ic = [random.uniform(0, 1) * 2 * pi - pi, random.uniform(0, 1) * 2 *  MV_TR - MV_TR ]
# ic = np.array([-3., -6.], dtype=np.double)
#
ic = np.array([-2.9931527672508436, 1.01293626720169])

tstamps = np.linspace(0, 10, 100_001)

print(pendulum(0, ic))

# h = 0.004998
h = 0.05
T = 30
# rk = solve_ivp(pendulum, [0, T], ic, rtol=1e-06)
e = solve_Euler(pendulum, [0, T], ic, h)
esi = solve_semii_Euler(pendulum, [0, T], ic, h)
# ei = solve_i_Euler(pendulum, [0, T], ic, h)
print(esi[:, -1])
plt.title(f"Pendulum solved by Euler schemes for {cname}")
plt.plot(e[0], e[1], label="Euler")
# plt.plot(ei[0], ei[1], label="implicit Euler")
plt.plot(esi[0], esi[1], label="semi-implicit Euler")
# plt.plot(rk.y[0], rk.y[1], label="RK45")
plt.legend()
# plt.savefig(f"h{h}_ic{ic[0]}_{ic[1]}.png", dpi=300)
plt.show()
