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

d0 = f.layerList[0].elements[:n]
d1 = f.layerList[0].elements[n:]
omega = (1/(torch.exp(d0+d1))).detach()
omega, idx = torch.sort(omega)

# Calculate modes in the latent space.

from matplotlib import pyplot as plt

frnvp = utils.extractFlow(f).to(dtype)
# cluster plot.
def plot2DCluster(model,data,label,idx):
    tmp = torch.randn(data.shape)
    datap = torch.cat([data,tmp],dim = 1)
    zs = model.forward(datap)[0].detach()[:,:28*28]
    plt.figure(figsize=(12,12))
    plotdata = [[] for _ in range(10)]
    colormap = ["black","peru","darkorange","tan","olive","green","red","lightslategray","blue","purple"]
    for no,lb in enumerate(label):
        i = int(lb.item())
        plotdata[i].append((zs[no][idx[0]],zs[no][idx[1]]))
    plotdata = [np.array(plotdata[i]) for i in range(10)]
    for i in range(10):
        if len(plotdata[i]) == 0:
            continue
        plt.scatter(plotdata[i][:,0],plotdata[i][:,1],c=colormap[i],label=str(i))
    plt.legend()
    plt.show()

RES = []
LAB = []
sampleSize = 700
for _ in range(1000):
    idxs = np.random.randint(0,dataset.shape[0],sampleSize)
    sampleset = dataset[idxs]
    samplelabs = datasetlabs[idxs]
    zs = frnvp.forward(sampleset)[0].detach()[:,idx[:2]]
    RES.append(zs)
    LAB.append(samplelabs)
RES = torch.cat(RES,0).detach().numpy()
LAB = torch.cat(LAB,0).detach().numpy()

np.savez("ScatterMNIST.npz",RES=RES,LAB=LAB)
#plot2DCluster(f,sampleset,samplelabs,idx)

