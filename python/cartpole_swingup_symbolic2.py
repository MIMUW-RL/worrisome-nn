# flake8: noqa: E501
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def replaceX(s, p):
    N = 10
    s = s.replace("torch.tanh", "tanh")
    for i in range(N):
        s = s.replace(f"x[:, {i}]", f"x{i}")
        if f"self.p[{i}]" in s:
            s = s.replace(f"self.p[{i}]", str(p[i].detach().numpy()))
    return s


# 17
# (((x3 * 2.5899072) + x4) / (((x4 * ((x3 * 4.591682) + x4)) * -0.1480232) + -2.0037756))
class Symbolic17(nn.Module):
    name = "symbolic_complex17"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(
            torch.FloatTensor(
                [
                    2.5899072,
                    1.0,
                    4.591682,
                    1.0,
                    -0.1480232,
                    -2.0037756,
                ]
            )
        )

    def forward(self, x):
        r = ((x[:, 3] * self.p[0]) + self.p[1] * x[:, 4]) / (
            ((x[:, 4] * ((x[:, 3] * self.p[2]) + self.p[3] * x[:, 4])) * self.p[4]) + self.p[5]
        )
        return r

    def to_str(self):
        return replaceX(
            "((x[:, 3] * self.p[0]) + self.p[1] * x[:, 4]) / (((x[:, 4] * ((x[:, 3] * self.p[2]) + self.p[3] * x[:, 4])) * self.p[4]) + self.p[5])",
            self.p,
        )

    def to_str_raw(self):
        return self.to_str()

    def unscale_action(self, scaled_action):
        return scaled_action


# 19
# (((x3 * 3.0173452) + x4) / ((((x4 + x3) * ((x3 * 4.591682) + x4)) * -0.1480232) + -1.6779317))
class Symbolic19(nn.Module):
    name = "symbolic_complex19"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.FloatTensor([3.0173452, 1.0, 1.0, 1.0, 4.591682, 1.0, -0.1480232, -1.6779317]))

    def forward(self, x):
        r = ((x[:, 3] * self.p[0]) + self.p[1] * x[:, 4]) / (
            (((self.p[2] * x[:, 4] + self.p[3] * x[:, 3]) * ((x[:, 3] * self.p[4]) + self.p[5] * x[:, 4])) * self.p[6])
            + self.p[7]
        )
        return r

    def to_str(self):
        return replaceX(
            "((x[:, 3] * self.p[0]) + self.p[1] * x[:, 4]) / ((((self.p[2] * x[:, 4] + self.p[3] * x[:, 3]) * ((x[:, 3] * self.p[4]) + self.p[5]*x[:, 4])) * self.p[6]) + self.p[7])",
            self.p,
        )

    def to_str_raw(self):
        return self.to_str()

    def unscale_action(self, scaled_action):
        return scaled_action


# 21
# (((x3 * 3.9907198) + x4) / (((((x3 * 4.0490804) + x4) * ((x3 * 4.0139728) + x4)) * -0.19251533) + -1.1983613))
class Symbolic21(nn.Module):
    name = "symbolic_complex21"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(
            torch.FloatTensor(
                [
                    3.9907198,
                    1,
                    4.0490804,
                    1,
                    4.0139728,
                    1,
                    -0.19251533,
                    -1.1983613,
                ]
            )
        )

    def forward(self, x):
        r = ((x[:, 3] * self.p[0]) + self.p[1] * x[:, 4]) / (
            (
                (((x[:, 3] * self.p[2]) + self.p[3] * x[:, 4]) * ((x[:, 3] * self.p[4]) + self.p[5] * x[:, 4]))
                * self.p[6]
            )
            + self.p[7]
        )
        return r

    def to_str(self):
        return replaceX(
            "((x[:, 3] * self.p[0]) + self.p[1] * x[:, 4]) / (((((x[:, 3] * self.p[2]) + self.p[3] * x[:, 4]) * ((x[:, 3] * self.p[4]) + self.p[5] * x[:, 4])) * self.p[6]) + self.p[7])",
            self.p,
        )

    def to_str_raw(self):
        return self.to_str()

    def unscale_action(self, scaled_action):
        return scaled_action


# 23
# ((((x3 * 3.942978) + -0.20499013) + x4) / (((((x3 * 4.0139728) + x4) * ((x3 * 4.0139728) + x4)) * -0.20009479) + -1.056604))
class Symbolic23(nn.Module):
    name = "symbolic_complex23"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(
            torch.FloatTensor(
                [
                    3.942978,
                    -0.20499013,
                    1,
                    4.0139728,
                    1,
                    4.0139728,
                    1,
                    -0.20009479,
                    -1.056604,
                ]
            )
        )

    def forward(self, x):
        r = (((x[:, 3] * self.p[0]) + self.p[1]) + self.p[2] * x[:, 4]) / (
            (
                (((x[:, 3] * self.p[3]) + self.p[4] * x[:, 4]) * ((x[:, 3] * self.p[5]) + self.p[6] * x[:, 4]))
                * self.p[7]
            )
            + self.p[8]
        )
        return r

    def to_str(self):
        return replaceX(
            "(((x[:, 3] * self.p[0]) + self.p[1]) + self.p[2] * x[:, 4]) / (((((x[:, 3] * self.p[3]) + self.p[4] * x[:, 4]) * ((x[:, 3] * self.p[5]) + self.p[6] * x[:, 4])) * self.p[7]) + self.p[8])",
            self.p,
        )

    def to_str_raw(self):
        return self.to_str()

    def unscale_action(self, scaled_action):
        return scaled_action


# 25
# ((((x3 * 3.9907198) + -0.2920652) + x4) / (((((x3 * 3.9907198) + x4) * ((x3 * 3.9907198) + (x4 + -0.58206606))) * -0.19918783) + -1.1192385))
class Symbolic25(nn.Module):
    name = "symbolic_complex25"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(
            torch.FloatTensor(
                [
                    3.9907198,
                    -0.2920652,
                    1,
                    3.9907198,
                    1,
                    3.9907198,
                    1,
                    -0.58206606,
                    -0.19918783,
                    -1.1192385,
                ]
            )
        )

    def forward(self, x):
        r = (((x[:, 3] * self.p[0]) + self.p[1]) + self.p[2] * x[:, 4]) / (
            (
                (
                    ((x[:, 3] * self.p[3]) + self.p[4] * x[:, 4])
                    * ((x[:, 3] * self.p[5]) + (self.p[6] * x[:, 4] + self.p[7]))
                )
                * self.p[8]
            )
            + self.p[9]
        )
        return r

    def to_str(self):
        return replaceX(
            "(((x[:, 3] * self.p[0]) + self.p[1]) + self.p[2] * x[:, 4]) / (((((x[:, 3] * self.p[3]) + self.p[4] * x[:, 4]) * ((x[:, 3] * self.p[5]) + (self.p[6] * x[:, 4] + self.p[7]))) * self.p[8]) + self.p[9])",
            self.p,
        )

    def to_str_raw(self):
        return self.to_str()

    def unscale_action(self, scaled_action):
        return scaled_action


# 15
# (((((x3 / 0.092210256) + -1.0281227) * ((x2 + 1.294536) * x2)) + x4) * -0.11011119)
class Symbolic15(nn.Module):
    name = "symbolic_complex15"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(
            torch.FloatTensor(
                [
                    1 / 0.092210256,
                    -1.0281227,
                    1,
                    1.294536,
                    1,
                    1,
                    -0.11011119,
                ]
            )
        )

    def forward(self, x):
        r = (
            (((x[:, 3] * self.p[0]) + self.p[1]) * ((self.p[2] * x[:, 2] + self.p[3]) * self.p[4] * x[:, 2]))
            + self.p[5] * x[:, 4]
        ) * self.p[6]
        return r

    def to_str(self):
        return replaceX(
            "((((x[:, 3] * self.p[0]) + self.p[1]) * ((self.p[2] * x[:, 2] + self.p[3]) * self.p[4] * x[:, 2])) + self.p[5] * x[:, 4]) * self.p[6]",
            self.p,
        )

    def to_str_raw(self):
        return self.to_str()

    def unscale_action(self, scaled_action):
        return scaled_action


# 27
# ((((x3 * 3.9907198) + -0.2077273) + (x4 + (x1 * -0.15355267))) / (((((x3 * 3.9907198) + x4) * ((x3 * 3.130886) + x4)) * -0.21450245) + -1.1192385))
class Symbolic27(nn.Module):
    name = "symbolic_complex27"

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(
            torch.FloatTensor(
                [
                    3.9907198,
                    -0.2077273,
                    1,
                    -0.15355267,
                    3.9907198,
                    1,
                    3.130886,
                    1,
                    -0.21450245,
                    -1.1192385,
                ]
            )
        )

    def forward(self, x):
        r = (((x[:, 3] * self.p[0]) + self.p[1]) + (self.p[2] * x[:, 4] + (x[:, 1] * self.p[3]))) / (
            (
                (((x[:, 3] * self.p[4]) + self.p[5] * x[:, 4]) * ((x[:, 3] * self.p[6]) + self.p[7] * x[:, 4]))
                * self.p[8]
            )
            + self.p[9]
        )

        return r

    def to_str(self):
        return replaceX(
            "(((x[:, 3] * self.p[0]) + self.p[1]) + (self.p[2] * x[:, 4] + (x[:, 1] * self.p[3]))) / (((((x[:, 3] * self.p[4]) + self.p[5] * x[:, 4]) * ((x[:, 3] * self.p[6]) + self.p[7] * x[:, 4])) * self.p[8]) + self.p[9])",
            self.p,
        )

    def to_str_raw(self):
        return self.to_str()

    def unscale_action(self, scaled_action):
        return scaled_action


# a simple NN controller
class SmallNet(nn.Module):
    name = "SmallNet"

    def __init__(self):
        super().__init__()
        self.n1 = 5
        self.n2 = 50
        self.n3 = 1
        self.name = f"SmallNet{self.n2}"
        self.fc1 = nn.Linear(self.n1, self.n2, bias=False)
        self.fc2 = nn.Linear(self.n2, self.n3, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

    def unscale_action(self, scaled_action):
        return scaled_action

    def to_str_raw(self):
        return self.to_str()

    def vectorize_param(self):
        params = []
        params.append(self.fc1.weight.detach().numpy().reshape(-1))
        params.append(self.fc2.weight.detach().numpy().reshape(-1))
        return np.hstack(params)

    def load_vec_param(self, vec):
        self.fc1.weight = torch.nn.Parameter(
            torch.reshape(torch.FloatTensor(vec[: self.n1 * self.n2]), (self.n2, self.n1))
        )
        self.fc2.weight = torch.nn.Parameter(
            torch.reshape(torch.FloatTensor(vec[self.n1 * self.n2 :]), (self.n3, self.n2))
        )

    def save_weights(self):
        with open("model_data.txt", "w") as f:
            f.write(f"{self.n1} -> {self.n2} (relu) -> {self.n3} (tanh) ")
        for name, para in self.named_parameters():
            print("{}: {}".format(name, para.shape))
            np.savetxt(f"{name}.csv", para.detach().numpy(), delimiter=",")
        return 0

    def to_str(self):
        return f"5 -> {self.n2} (relu) -> 1 (tanh) "
