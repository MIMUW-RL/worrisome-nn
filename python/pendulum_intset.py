from scipy.integrate import solve_ivp, RK45
import numpy as np
from numpy import arccos, cos, sin, abs, exp, tanh, log
import matplotlib.pyplot as plt
from matplotlib import rc
import random
from math import pi
from symbol import import_stmt
import torch as th
from torch import nn
import torch.nn.functional as F
import control_ai_manifolds.python.pendulum.pendulum_symbolic1 as pendulum_symbolic1
import seaborn as sns

rc("text", usetex=True)
G = 10
MV_TR = 8

cname = "NN_controller/Symbolic_simple2_finetune/symbolic_complex19A_final_model.pt"
cnameshort = cname.split("/")[-1].split(".")[0]
print(cnameshort)

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
    return ys


def solve_Euler(fcn, t_span, y0, h):
    y = y0
    N = int(t_span[1] / h)
    ts = [t_span[0] + x * h for x in range(N)]
    ys = [y0]
    for i in range(N):
        y = y + h * fcn(h, y)
        ys.append(mv_trunc(y.copy()))
    ys = np.vstack(ys).T
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
    s = th.unsqueeze(th.Tensor([s1, s2, s3]), dim=0)
    ta = model(s)
    torque = ta.detach().numpy().squeeze()
    torque = model.unscale_action(torque)
    torque = torque_trunc(torque)
    # x0 = s1
    # x1 = s2
    # x2 = s3

    dy[0] = y[1]
    dy[1] = 3.0 * torque + 3.0 / 2.0 * G * s2
    return dy


np.set_printoptions(16)

h = 0.0025
T = 10
ICs = 100

# sample
IChigh = [pi, -5]
IClow = [0, 5]
e_s = []
esi_s = []
ei_s = []
for i in range(ICs):
    ic = [random.uniform(0, 1) * (IChigh[j] - IClow[j]) + IClow[j] for j in range(len(IClow))]
    print(f"ic={ic}")
    # rk = solve_ivp(pendulum, [0, T], ic, rtol=1e-06).y
    # rk = np.array(rk)
    # rk_s.append( rk )
    # e_s.append( solve_Euler(pendulum, [0, T], ic, h) )
    esi_traj = solve_semii_Euler(pendulum, [0, T], ic, h)
    esi_s.append(esi_traj)
    print(f"finalpt={esi_traj[:, -1]}\n")
    # ei_s.append( solve_i_Euler(pendulum, [0, T], ic, h))

sname = "semi-implicit Euler"
data = esi_s
c = "b"


plt.title(f"Pendulum solved by the {sname} scheme ($h={h}$)")
for j in range(len(data)):
    plt.plot(data[j][0], data[j][1], c=c, alpha=0.3)
plt.xlabel("angle $\omega$ (unnormalized)")
plt.ylabel("velocity $\Theta$")
plt.savefig(f"h{h}_ICs{ICs}_{sname.replace(' ', '')}_{cnameshort}.png", dpi=300)
plt.show()
