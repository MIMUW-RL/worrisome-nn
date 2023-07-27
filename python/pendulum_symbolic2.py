# flake8: noqa: E501
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def replaceX(s):
    s = s.replace("x[:,2]", "x2")
    s = s.replace("x[:,1]", "x1")
    s = s.replace("x[:,0]", "x0")
    s = s.replace("self.p[0]", "p0")
    s = s.replace("self.p[1]", "p1")
    s = s.replace("self.p[2]", "p2")
    s = s.replace("self.p[3]", "p3")
    s = s.replace("self.p[4]", "p4")
    s = s.replace("self.p[5]", "p5")
    return s


class Symbolic7Orig(nn.Module):
    name = "symbolic_complex7Orig"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        r = ((x[:, 2] * x[:, 0]) + x[:, 1]) / -2.3512144
        return r

    def to_str(self):
        return "(((x2 * x0) + x1) / -2.3512144)"

    def to_str_raw(self):
        return replaceX("(((x[:,2] * x[:,0]) + x[:,1]) / -2.3512144)")

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic7A(nn.Module):
    name = "symbolic_complex7A"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.3512144, 1.0, 1.0]))

    def forward(self, x):
        r = (self.p[1] * (x[:, 2] * x[:, 0]) + self.p[2] * x[:, 1]) / self.p[0]
        return r

    def to_str(self):
        return replaceX(f"(({self.p[1]}*(x[:,2] * x[:,0]) + {self.p[2]}*x[:,1] ) / {self.p[0]})")

    def to_str_raw(self):
        return replaceX("((self.p[1] * (x[:,2] * x[:,0]) + self.p[2] * x[:,1]) / self.p[0])")

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic7B(nn.Module):
    name = "symbolic_complex7B"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([1.0 / -2.3512144, 1.0, 1.0]))

    def forward(self, x):
        r = (self.p[1] * (x[:, 2] * x[:, 0]) + self.p[2] * x[:, 1]) * self.p[0]
        return r

    def to_str(self):
        return replaceX(f"(({self.p[1]}*(x[:,2] * x[:,0]) + {self.p[2]}*x[:,1] ) / {self.p[0]})")

    def to_str_raw(self):
        return replaceX("((self.p[1] * (x[:,2] * x[:,0]) + self.p[2] * x[:,1]) / self.p[0])")

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic9A(nn.Module):
    name = "symbolic_complex9A"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.326582, 1.0, 1.0, 1.0]))

    def forward(self, x):
        r = (((self.p[1] * x[:, 2] + self.p[2] * x[:, 1]) * x[:, 0]) + self.p[3] * x[:, 1]) / self.p[0]
        return r

    def to_str(self):
        return replaceX(f"(((({self.p[1]}*x[:,2] + {self.p[2]}*x[:,1]) * x[:,0]) + {self.p[3]}*x[:,1]) / {self.p[0]})")

    def to_str_raw(self):
        return replaceX("((((self.p[1]*x[:,2] + self.p[2]*x[:,1]) * x[:,0]) + self.p[3]*x[:,1]) / self.p[0])")

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic9B(nn.Module):
    name = "symbolic_complex9B"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([1.0 / -2.326582, 1.0, 1.0, 1.0, 1.0]))

    def forward(self, x):
        r = (((self.p[1] * x[:, 2] + self.p[2] * x[:, 1]) * x[:, 0]) + self.p[3] * x[:, 1] + self.p[4]) * self.p[0]
        return r

    def to_str(self):
        return replaceX(
            f"(((({self.p[1]}*x[:,2] + {self.p[2]}*x[:,1]) * x[:,0]) + {self.p[3]}*x[:,1] + {self.p[4]}) / {self.p[0]})"
        )

    def to_str_raw(self):
        return replaceX(
            "((((self.p[1]*x[:,2] + self.p[2]*x[:,1]) * x[:,0]) + self.p[3]*x[:,1] + self.p[4]) / self.p[0])"
        )

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic17A(nn.Module):
    name = "symbolic_complex17A"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-0.07611064, -0.7107913, 1.0, 1.0, 1.0]))

    def forward(self, x):
        r = ((self.p[2] * x[:, 2] + self.p[3] * x[:, 1]) * x[:, 0] * self.p[4]) / (
            (x[:, 2] * ((x[:, 2] * (x[:, 0] * x[:, 0])) * self.p[0])) + self.p[1]
        )
        return r

    def to_str(self):
        return replaceX(
            f"((({self.p[2]}*x[:,2] + {self.p[3]}*x[:,1]) * x[:,0] * {self.p[4]}) / ((x[:,2] * ((x[:,2] * (x[:,0] * x[:,0])) * {self.p[0]})) + {self.p[1]}))"
        )

    def to_str_raw(self):
        return replaceX(
            "(((self.p[2]*x[:,2] + self.p[3]*x[:,1]) * x[:,0] * self.p[4]) / ((x[:,2] * ((x[:,2] * (x[:,0] * x[:,0])) * self.p[0])) + self.p[1]))"
        )

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic13A(nn.Module):
    name = "symbolic_complex13A"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-0.060752857, -0.64416397, 1.0, 1.0]))

    def forward(self, x):
        r = ((x[:, 2] * self.p[2] + x[:, 1] * self.p[3]) * x[:, 0]) / ((x[:, 2] * (x[:, 2] * self.p[0])) + self.p[1])
        return r

    def to_str_raw(self):
        return replaceX(
            "(((x[:,2]*self.p[2] + x[:,1]*self.p[3]) * x[:,0]) / ((x[:,2] * (x[:,2] * self.p[0])) + self.p[1]))"
        )

    def to_str(self):
        return replaceX(
            f"(((x[:,2]*{self.p[2]} + x[:,1]*{self.p[3]}) * x[:,0]) / ((x[:,2] * (x[:,2] * {self.p[0]})) + {self.p[1]}))"
        )

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic19A(nn.Module):
    name = "symbolic_complex19A"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([0.2807797, 0.14147963, -0.061397437, -0.8070769, 1.0, 1.0]))

    def forward(self, x):
        r = ((self.p[4] * x[:, 2] + (x[:, 1] / self.p[0])) * x[:, 0]) / (
            (((x[:, 1] / self.p[1]) + self.p[5] * x[:, 2]) * (x[:, 2] * self.p[2])) + self.p[3]
        )
        return r

    def to_str_raw(self):
        return replaceX(
            "(((self.p[4]*x[:,2] + (x[:,1] / self.p[0])) * x[:,0]) / ((((x[:,1] / self.p[1]) + self.p[5]*x[:,2]) * (x[:,2] * self.p[2])) + self.p[3]))"
        )

    def to_str(self):
        return replaceX(
            f"((({self.p[4]}*x[:,2] + (x[:,1] / {self.p[0]})) * x[:,0]) / ((((x[:,1] / {self.p[1]}) + {self.p[5]}*x[:,2]) * (x[:,2] * {self.p[2]})) + {self.p[3]}))"
        )

    def unscale_action(self, scaled_action):
        return scaled_action
