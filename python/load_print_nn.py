import torch as th
from os import listdir
import sys
import numpy as np
import pendulum_symbolic1
import pendulum_symbolic2
import cartpole_swingup_symbolic2


dir = str(sys.argv[1])
filens = listdir(dir)
filen = None

models = []
filenss = []
for fn in filens:
    if (".pth" in fn[-4:]) or (".pt" in fn[-4:]):
        print(fn)
        model = th.load(dir + fn).cpu()
        model.eval()
        models.append(model)
        filenss.append(fn)

        with open(dir + f'{fn.split(".")[0]}_model_data.txt', "w") as f:
            try:
                f.write(model.to_str())
            except AttributeError:
                f.write(model.__class__.__name__)

        for name, para in model.named_parameters():
            print("{}: {}".format(name, para.shape))
            np.savetxt(dir + f"{fn}_{name}.csv", para.detach().numpy(), delimiter=",")
