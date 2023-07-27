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


class SymbolicLandajuelaOrig(nn.Module):
    name = "symbolic_LandajuelaOrig"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.000000, -8.0, -2.0]))

    def forward(self, x):
        r = self.p[0] * x[:, 1] + (self.p[1] * x[:, 1] + self.p[2] * x[:, 2]) / (x[:, 0])
        return r

    def to_str(self):
        return f"{self.p[0]} *x1 + ({self.p[1]}*x1 + {self.p[2]}*x2) / (x0)"

    def to_str_raw(self):
        return replaceX("self.p[0] * x[:,1] + (self.p[1] * x[:,1]  + self.p[2] * x[:,2]) / (x[:,0])")

    def unscale_action(self, scaled_action):
        return scaled_action


class SymbolicLandajuelaOrig2(nn.Module):
    name = "symbolic_LandajuelaOrig2"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-7.08, -13.39, -3.12, 0.27]))

    def forward(self, x):
        r = self.p[0] * x[:, 1] + (self.p[1] * x[:, 1] + self.p[2] * x[:, 2]) / (x[:, 0]) + self.p[3]
        return r

    def to_str(self):
        return f"{self.p[0]} *x1 + ({self.p[1]}*x1 + {self.p[2]}*x2) / (x0) + {self.p[3]}"

    def to_str_raw(self):
        return replaceX("self.p[0] * x[:,1] + (self.p[1] * x[:,1]  + self.p[2] * x[:,2]) / (x[:,0]) + self.p[3]")

    def unscale_action(self, scaled_action):
        return scaled_action


class SymbolicLandajuela(nn.Module):
    name = "symbolic_Landajuela"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-7.08, -13.39, -3.12, 0.27]))

    def forward(self, x):
        r = self.p[0] * x[:, 1] + (self.p[1] * x[:, 1] + self.p[2] * x[:, 2]) / (x[:, 0]) + self.p[3]
        return r

    def to_str(self):
        return f"{self.p[0]} *x1 + ({self.p[1]}*x1 + {self.p[2]}*x2) / (x0) + {self.p[3]}"

    def to_str_raw(self):
        return replaceX("self.p[0] * x[:,1] + (self.p[1] * x[:,1]  + self.p[2] * x[:,2]) / (x[:,0]) + self.p[3]")

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic7Orig(nn.Module):
    name = "symbolic_complex7Orig"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.0000005]))

    def forward(self, x):
        r = torch.tanh(x[:, 2] / torch.tanh(x[:, 0])) * self.p[0]
        return r

    def to_str(self):
        return f"tanh((x2) / tanh(x0)) * {self.p[0]} "

    def to_str_raw(self):
        return replaceX("torch.tanh( x[:,2] / torch.tanh(x[:,0])) * self.p[0]")

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic7A(nn.Module):
    name = "symbolic_complex7A"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.0000005, 1.0, 1.0]))

    def forward(self, x):
        r = torch.tanh((self.p[1] * x[:, 2]) / torch.tanh(self.p[2] * x[:, 0])) * self.p[0]
        return r

    def to_str(self):
        return f"tanh(({self.p[1]}*x2) / tanh({self.p[2]}*x0)) * {self.p[0]} "

    def to_str_raw(self):
        return replaceX("torch.tanh( (self.p[1]*x[:,2]) / torch.tanh(self.p[2]*x[:,0])) * self.p[0]")

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic7B(nn.Module):
    name = "symbolic_complex7B"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.0000005, 1.0, 1.0, 0.0, 0.0]))

    def forward(self, x):
        r = torch.tanh((self.p[1] * x[:, 2] + self.p[3]) / torch.tanh(self.p[2] * x[:, 0])) * self.p[0] + self.p[4]
        return r

    def to_str(self):
        return f"tanh(({self.p[1]}*x2 + {self.p[3]}) / tanh({self.p[2]}*x0)) * {self.p[0]} + {self.p[4]}"

    def to_str_raw(self):
        return replaceX(
            "torch.tanh( (self.p[1]*x[:,2] + self.p[3]) / torch.tanh(self.p[2]*x[:,0])) * self.p[0] + self.p[4]"
        )

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic8Orig(nn.Module):
    name = "symbolic_complex8Orig"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.0000005]))

    def forward(self, x):
        r = torch.tanh((x[:, 2] + x[:, 1]) / x[:, 0]) * self.p[0]
        return r

    def to_str(self):
        return f"tanh(( x2 + x1) / x0) * {self.p[0]} "

    def to_str_raw(self):
        return replaceX("torch.tanh( (x[:,2] + x[:,1]) / x[:,0]) * self.p[0]")

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic8A(nn.Module):
    name = "symbolic_complex8A"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.0000005, 1.0, 1.0]))

    def forward(self, x):
        r = torch.tanh((self.p[1] * x[:, 2] + x[:, 1]) / (self.p[2] * x[:, 0])) * self.p[0]
        return r

    def to_str(self):
        return f"tanh(( {self.p[1].detach()}*x2 + x1) / ({self.p[2].detach()}*x0)) * {self.p[0]} "

    def to_str_raw(self):
        return replaceX("torch.tanh( (self.p[1]*x[:,2] + x[:,1]) / (self.p[2]*x[:,0])) * self.p[0]")

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic8B(nn.Module):
    name = "symbolic_complex8B"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.0000005, 1.0, 1.0, 1.0, 0.0, 0.0]))

    def forward(self, x):
        r = (
            torch.tanh((self.p[1] * x[:, 2] + self.p[3] * x[:, 1] + self.p[4]) / (self.p[2] * x[:, 0])) * self.p[0]
            + self.p[5]
        )
        return r

    def to_str(self):
        return (
            f"tanh(( {self.p[1].detach()}*x2 + {self.p[3].detach()}*x1 + {self.p[4].detach()}) "
            + "/ ({self.p[2].detach()}*x0)) * {self.p[0]} + {self.p[5].detach()}"
        )

    def to_str_raw(self):
        return replaceX(
            "torch.tanh( (self.p[1]*x[:,2] + self.p[3]*x[:,1] + self.p[4]) / (self.p[2]*x[:,0] )) * self.p[0] + self.p[5]"
        )

    def unscale_action(self, scaled_action):
        return scaled_action


class Symbolic9Orig(nn.Module):
    name = "symbolic_complex9Orig"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.0000005]))

    def forward(self, x):
        r = torch.tanh((x[:, 2] + x[:, 1]) / torch.tanh(x[:, 0])) * self.p[0]
        return r

    def to_str(self):
        return f"tanh( ( x2 + x1) / tanh(x0)) * {self.p[0]}"

    def to_str_raw(self):
        return replaceX("torch.tanh((x[:,2] + x[:,1]) / torch.tanh(x[:,0])) * self.p[0]")

    def unscale_action(self, scaled_action):
        return scaled_action


# example symbolic controller
class Symbolic9A(nn.Module):
    name = "symbolic_complex9A"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.0000005, 1.0, 1.0]))

    def forward(self, x):
        r = torch.tanh((self.p[1] * x[:, 2] + self.p[1] * x[:, 1]) / torch.tanh(self.p[2] * x[:, 0])) * self.p[0]
        return r

    def to_str(self):
        return f"tanh( ({self.p[1].detach()}*x2 + x1) / tanh({self.p[2].detach()}*x0))*{self.p[0]} "

    def to_str_raw(self):
        return replaceX(
            "torch.tanh((self.p[1] * x[:,2] + self.p[1] * x[:,1]) / torch.tanh(self.p[2] * x[:,0])) * self.p[0]"
        )

    def unscale_action(self, scaled_action):
        return scaled_action


# example symbolic controller
class Symbolic9B(nn.Module):
    name = "symbolic_complex9B"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([-2.0000005, 1.0, 1.0, 1.0, 0.0]))

    def forward(self, x):
        r = (
            torch.tanh((self.p[1] * x[:, 2] + self.p[2] * x[:, 1]) / torch.tanh(self.p[3] * x[:, 0])) * self.p[0]
            + self.p[4]
        )
        return r

    def to_str(self):
        return (
            f"tanh(({self.p[1].detach()}*x2 + {self.p[2].detach()}*x1) / tanh({self.p[3].detach()}*x0))*{self.p[0]}"
            + " + {self.p[4]}"
        )

    def to_str_raw(self):
        return replaceX(
            "torch.tanh((self.p[1] * x[:,2] + self.p[2] * x[:,1]) / torch.tanh(self.p[3] * x[:,0])) * self.p[0] + self.p[4]"
        )

    def unscale_action(self, scaled_action):
        return scaled_action


# a simple NN controller
class SmallNet(nn.Module):
    name = "SmallNet"

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

    def unscale_action(self, scaled_action):
        low = -2.0
        high = 2.0
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def to_str_raw(self):
        return replaceX("tba")

    def save_weights(self):
        with open("model_data.txt", "w") as f:
            f.write("3 -> 10 (relu) -> 1 (tanh) ")
        for name, para in self.named_parameters():
            print("{}: {}".format(name, para.shape))
            np.savetxt(f"{name}.csv", para.detach().numpy(), delimiter=",")
        return 0

    def to_str(self):
        return "3 -> 10 (relu) -> 1 (tanh) "
