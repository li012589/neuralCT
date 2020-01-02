import numpy as np
import torch
from torch import nn
import flow
import train
import utils
import math
import h5py
# Set gobal variables.

rootFolder = "./demo/MNIST_relax_True_shift_True_T_300_depthLevel_1_l16_M3_H680/"
device = torch.device("cpu")
dtype = torch.float32
dataset = "./database/mnist.npz"
labdata = "./database/mnistlabs.npz"

# Load paremeters

with h5py.File(rootFolder+"/parameter.hdf5","r") as f:
    n = int(np.array(f["n"]))
    numFlow = int(np.array(f["numFlow"]))
    lossPlotStep = int(np.array(f["lossPlotStep"]))
    hidden = int(np.array(f["hidden"]))
    nlayers = int(np.array(f["nlayers"]))
    nmlp = int(np.array(f["nmlp"]))
    lr = int(np.array(f["lr"]))
    batchSize = int(np.array(f["batchSize"]))
    epochs = int(np.array(f["epochs"]))
    K = int(np.array(f["K"]))
    
# Build the target.

from utils import MDSampler,load
dataset = load(dataset).to(device).to(dtype)
datasetlabs = torch.from_numpy(np.argmax(np.load(labdata)["arr_0"],axis=1)).to(device).to(dtype)
target = MDSampler(dataset)
    
# Rebuild the model.

def innerBuilder(num):
    maskList = []
    for i in range(nlayers):
        if i %2==0:
            b = torch.zeros(num)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[i] = 1
            b=b.reshape(1,num)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    fl = flow.RNVP(maskList, [utils.SimpleMLPreshape([num]+[hidden]*nmlp+[num],[nn.Softplus()]*nmlp+[None]) for _ in range(nlayers)], [utils.SimpleMLPreshape([num]+[hidden]*nmlp+[num],[nn.Softplus()]*nmlp+[utils.ScalableTanh(num)]) for _ in range(nlayers)])
    return fl

from utils import flowBuilder

f = flowBuilder(n,numFlow,innerBuilder,1).to(device).to(dtype)

# Load saving.

import os
import glob
name = max(glob.iglob(rootFolder+"savings/"+'*.saving'), key=os.path.getctime)
print("load saving at "+name)
saved = torch.load(name,map_location=device)
f.load(saved);

# Calculate modes in the latent space.

from matplotlib import pyplot as plt
from utils import logit_back,logit

d0 = f.layerList[0].elements[:n]
d1 = f.layerList[0].elements[n:]
omega = (1/(torch.exp(d0+d1))).detach()
omega, idx = torch.sort(omega)

from matplotlib import pyplot as plt

# Set gobal variables.
data = np.load("./ScatterMNIST.npz")
labs = data["LAB"]
dots = data["RES"]

trajs = np.load("./trajsMNIST.npz")["arr_0"]

plt.figure(figsize=(6,6))
plotdata = [[] for _ in range(10)]
colormap = ["black","peru","darkorange","tan","olive","green","red","lightslategray","blue","purple"]
#colormap = [["#"+hex(i)[2:]] for i in range(14548591,16674671,212608)]

for no,lb in enumerate(labs):
    i = int(lb.item())
    plotdata[i].append((dots[no][0],dots[no][1]))
plotdata = [np.array(plotdata[i])[:3000] for i in range(10)]
for i in range(10):
    if len(plotdata[i]) == 0:
        continue
    plt.scatter(plotdata[i][:,0],plotdata[i][:,1],c=colormap[i],label=str(i),alpha=0.3)
plt.legend()
plt.plot(trajs[:,idx[0]],trajs[:,idx[1]],linewidth=4,color = "darkorange")
plt.xlabel("$n_1$",fontsize = "x-large")
plt.ylabel("$n_2$",fontsize = "x-large")
plt.savefig("subplotTimeElve.pdf")
plt.show()
