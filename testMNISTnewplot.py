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

sample1 = target.sample(1)
plt.figure()
plt.imshow(logit_back(sample1[:,:784].reshape(28,28)),cmap="gray")

zsample = f.forward(sample1)[0].detach()
zsample1 = zsample[:,:784]

'''
zsample1 = f.forward(sample1)[0][:,:784].detach()
zsample = torch.cat([zsample1,torch.zeros_like(zsample1)],1)
'''

from utils import timeEvolve, buildSource

latentSource = buildSource(f).to(dtype)

trajs = timeEvolve(latentSource,0.005,30000,1,initalPoint=zsample.to(dtype)).reshape(-1,zsample.shape[1]).detach()

plt.figure()
plt.scatter(trajs[:,idx[0]].numpy(),trajs[:,idx[1]].numpy())
plt.plot(trajs[:,idx[0]].numpy(),trajs[:,idx[1]].numpy())

L = 10

selectedIdx = [i for i in range(1,trajs.shape[0],trajs.shape[0]//(L*L))]

selectedTrajs = trajs[selectedIdx,:]

'''
frnvp = utils.extractFlow(f).to(dtype)

selectedTrajsQ = logit_back(frnvp.inverse(selectedTrajs[:,:784])[0].reshape(L,L,28,28)).permute([0,2,1,3]).reshape(L*28,L*28).detach().numpy()
'''

_selectedTrajsQ = f.inverse(selectedTrajs)[0]
selectedTrajsQ = logit_back(_selectedTrajsQ[:,:784].reshape(L,L,28,28)).permute(0,2,1,3).reshape(L*28,L*28).detach().numpy()

from utils import measureM

momentums = measureM(selectedTrajs).detach().numpy()
momentumsp = momentums.reshape(10,10)

Qmomentums = measureM(selectedTrajs,idx,300).detach().numpy()
Qmomentumsp = Qmomentums.reshape(10,10)

'''
Qmomentums = measureM(_selectedTrajsQ).detach().numpy()
Qmomentumsp = Qmomentums.reshape(10,10)
'''

plt.figure(figsize=(12,12))
plt.imshow(selectedTrajsQ,cmap="gray")

'''
plt.figure(figsize=(12,1))
plt.plot(np.arange(momentums.shape[0]),momentums)
'''

plt.figure(figsize=(10,10))
for i in range(L):
    plt.subplot(L,1,i+1)
    plt.axhline(y=momentumsp[0,0],color="r",)
    plt.axhline(y=Qmomentumsp[0,0],color="k",)
    plt.plot(np.arange(momentumsp.shape[1]),momentumsp[i,:])
    plt.plot(np.arange(Qmomentumsp.shape[1]),Qmomentumsp[i,:])

plt.show()
