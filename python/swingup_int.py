import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import copy
from dataclasses import dataclass, field
from collections import namedtuple
import matplotlib.pylab as pylab

import cartpole_swingup_symbolic2
import cma

# swing-up cart-pole model
params = {
    "legend.fontsize": "xx-large",
    #    "figure.figsize": (15, 5),
    "axes.labelsize": "xx-large",
    "axes.titlesize": "xx-large",
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
}
pylab.rcParams.update(params)


@dataclass(frozen=True)
class CartParams:
    """Parameters defining the Cart."""

    width: float = 1 / 3
    height: float = 1 / 6
    mass: float = 0.5


@dataclass(frozen=True)
class PoleParams:
    """Parameters defining the Pole."""

    width: float = 0.05
    length: float = 0.6
    mass: float = 0.5


@dataclass
class CartPoleSwingUpParams:  # pylint: disable=no-member,too-many-instance-attributes
    """Parameters for physics simulation."""

    gravity: float = 9.82
    forcemag: float = 10.0
    deltat: float = 0.01
    friction: float = 0.1
    x_threshold: float = 2.4
    cart: CartParams = field(default_factory=CartParams)
    pole: PoleParams = field(default_factory=PoleParams)
    masstotal: float = field(init=False)
    mpl: float = field(init=False)

    def __post_init__(self):
        self.masstotal = self.cart.mass + self.pole.mass
        self.mpl = self.pole.mass * self.pole.length


def _get_obs(state):
    x_pos, x_dot, theta, theta_dot = state
    return [x_pos, x_dot, torch.cos(theta), torch.sin(theta), theta_dot]


def _reward_fn(state, action, next_state):
    return torch.cos(next_state[2])


params = CartPoleSwingUpParams()
State = namedtuple("State", "x_pos x_dot theta theta_dot")


def _transition_fn(state, action, explicit):
    action = action[0] * params.forcemag

    sin_theta = torch.sin(state.theta)
    cos_theta = torch.cos(state.theta)

    xdot_update = (
        -2 * params.mpl * (state.theta_dot**2) * sin_theta
        + 3 * params.pole.mass * params.gravity * sin_theta * cos_theta
        + 4 * action
        - 4 * params.friction * state.x_dot
    ) / (4 * params.masstotal - 3 * params.pole.mass * cos_theta**2)
    thetadot_update = (
        -3 * params.mpl * (state.theta_dot**2) * sin_theta * cos_theta
        + 6 * params.masstotal * params.gravity * sin_theta
        + 6 * (action - params.friction * state.x_dot) * cos_theta
    ) / (4 * params.pole.length * params.masstotal - 3 * params.mpl * cos_theta**2)

    delta_t = params.deltat
    if explicit:
        return State(
            x_pos=state.x_pos + state.x_dot * delta_t,
            x_dot=state.x_dot + xdot_update * delta_t,
            theta=state.theta + state.theta_dot * delta_t,
            theta_dot=state.theta_dot + thetadot_update * delta_t,
        )
    else:
        x_dot = state.x_dot + xdot_update * delta_t
        x_pos = state.x_pos + x_dot * delta_t
        theta_dot = state.theta_dot + thetadot_update * delta_t
        theta = state.theta + theta_dot * delta_t
        return State(
            x_pos=x_pos,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
        )


def _terminal(state):
    return bool(abs(state.x_pos) > params.x_threshold)


def step(state, action, explicit=True):
    # Valid action
    action = torch.clamp(action, -1, 1)
    state = _transition_fn(state, action, explicit)
    return state


BATCHSIZE = 10


# model = torch.load("cartpole_swingup/symbolic_simple/symbolic_complex21_cma_swingup_fav_model.pt").cpu()
model = torch.load("cartpole_swingup/symbolic_simple/td3_actor_400_300_ReLU.pth").cpu()
# print(model.p)
# model = torch.load("cartpole_swingup/symbolic1/actor.pth").cpu()
model.eval()


traj = []
STEPS = 2000

istate = State(
    *torch.normal(
        mean=torch.tensor([0.0, 0.0, np.pi, 0.0]),
        std=0.2,
    )
)
# istate = State(*torch.tensor([0.249760365563774, 0.0652270051995776, 2.03054525101292, -0.197796827821508]))

istate = State(*torch.tensor([-0.013613304802616, -0.499271467702198, 3.37706123873966, 0.494928453961448]))
# istate = State(*torch.tensor([0.499978595166148, 0.498098261225103, 2.64159273252545, 0.493215075043544]))
# istate = State(*torch.tensor([-0.474131601481899, -0.208240959970631, 3.51627492294, -0.413298990955921]))

done = False
explicit = True
i = 0
totrew = 0

xse = []
xsi = []
xsh = []

xdotse = []
xdotsi = []
xdotsh = []

thetase = []
thetasi = []
thetash = []

thetadotse = []
thetadotsi = []
thetadotsh = []


state = istate
while not done and i < STEPS:
    ta = model(torch.unsqueeze(torch.Tensor(_get_obs(state)), dim=0))
    state = step(state, ta, explicit=True)
    i += 1

    totrew += torch.squeeze(_reward_fn(state, ta, state))
    thetase.append(state[1].detach().numpy())
    thetadotse.append(state[3].detach().numpy())
    xse.append(state[0].detach().numpy())
    xdotse.append(state[2].detach().numpy())
    # if _terminal(state):
    #    break

print(totrew)
totrew = 0
i = 0
state = istate
while not done and i < STEPS:
    ta = model(torch.unsqueeze(torch.Tensor(_get_obs(state)), dim=0))
    state = step(state, ta, explicit=False)
    i += 1

    totrew += torch.squeeze(_reward_fn(state, ta, state))
    thetasi.append(state[2].detach().numpy())
    thetadotsi.append(state[3].detach().numpy())
    xsi.append(state[0].detach().numpy())
    xdotsi.append(state[1].detach().numpy())
    # if _terminal(state):
    #    break

print(totrew)
totrew = 0
i = 0
state = istate
params.deltat = 0.005
while not done and i < STEPS:
    ta = model(torch.unsqueeze(torch.Tensor(_get_obs(state)), dim=0))
    state = step(state, ta, explicit=False)
    i += 1
    totrew += torch.squeeze(_reward_fn(state, ta, state))
    thetash.append(state[1].detach().numpy())
    thetadotsh.append(state[3].detach().numpy())
    xsh.append(state[0].detach().numpy())
    xdotsh.append(state[2].detach().numpy())
    # if _terminal(state):
    #    break

print(totrew)

plt.rcParams["text.usetex"] = True
# plt.plot(thetase)
plt.plot(thetash, label=r"$\theta_t$")
print(thetash[-1])
# plt.plot(thetash)
# plt.show()
# plt.plot(xse)
plt.plot(xsh, label=r"$x_t$")
# plt.plot(xsh)
# plt.show()

# plt.plot(xdotse)
# plt.plot(xdotsi[:500])
# plt.plot(xdotsh)
# plt.show()
plt.xlabel("t")
plt.title("example transient for deep NN controller")

# plt.plot(thetadotse)
# plt.plot(thetadotsi[:1000])
# plt.plot(thetadotsh)
plt.legend()
plt.show()
# plt.savefig("symbolic_transient.pdf")
