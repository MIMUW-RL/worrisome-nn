import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import copy
from dataclasses import dataclass, field
from collections import namedtuple

import cartpole_swingup_symbolic2
import cma
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resume", required=False)
parser.add_argument("-i", "--iter", required=False, default=1000)
parser.add_argument("-s", "--sigma", required=False, default=1, type=float)
parser.add_argument("-b", "--batch", required=False, default=10, type=int)
args = parser.parse_args()

# swing-up cart-pole model implemented in torch
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


def _transition_fn(state, action):
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
    return State(
        x_pos=state.x_pos + state.x_dot * delta_t,
        x_dot=state.x_dot + xdot_update * delta_t,
        theta=state.theta + state.theta_dot * delta_t,
        theta_dot=state.theta_dot + thetadot_update * delta_t,
    )


def _terminal(state):
    return bool(abs(state.x_pos) > params.x_threshold)


def step(state, action):
    # Valid action
    action = torch.clamp(action, -1, 1)
    state = _transition_fn(state, action)
    return state


BATCHSIZE = args.batch

u = torch.load("cartpole_swingup/SmallNet/SmallNet25.pt")
print(u)
u.eval()

if args.resume:
    u = torch.load(args.resume)
    u.eval()


def integrate(params):
    rews = []

    u.load_vec_param(params)

    for b in range(BATCHSIZE):
        with torch.no_grad():
            s = State(
                *torch.normal(
                    mean=torch.FloatTensor([0.0, 0.0, np.pi, 0.0]),
                    std=0.2,
                )
            )
            totrew = 0
            for i in range(500):

                ua = u(torch.FloatTensor(_get_obs(s)))
                ua = u.unscale_action(ua)
                prevs = s
                s = step(s, ua)
                totrew += _reward_fn(prevs, ua, s)
                if _terminal(s):
                    break
            rews.append(-totrew)
    return np.mean(rews)


params0 = u.vectorize_param()
print(f"params0 {params0}")
sigma0 = args.sigma

x, es = cma.fmin2(integrate, params0, sigma0, options={"maxfevals": args.iter, "popsize": 64})

u.p = torch.nn.Parameter(torch.tensor(es.result.xbest), requires_grad=False)
print(f"saving model that achieved f={es.result.fbest} ({es.result.xbest}")
if args.resume:
    torch.save(u, f"{u.name}_cma_swingup_retrained.pt")
else:
    torch.save(u, f"{u.name}_cma_swingup_final_model.pt")

u.p = torch.nn.Parameter(torch.tensor(es.result.xfavorite), requires_grad=False)
print(f"saving favorite model ({es.result.xfavorite}")
if args.resume:
    torch.save(u, f"{u.name}_cma_swingup_fav_retrained.pt")
else:
    torch.save(u, f"{u.name}_cma_swingup_fav_model.pt")
