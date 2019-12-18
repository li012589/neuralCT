import numpy as np
import torch
from torch import nn
import flow
import train
import utils
import math
import h5py

# Set gobal variables.

rootFolder = "./demo/Model_CC(=O)NC(C)C(=O)NC_Batch_200_T_300_depthLevel_1_l8_M2_H128/"
device = torch.device("cpu")
dtype = torch.float32
smile = "CC(=O)NC(C)C(=O)NC"
dataset = "./database/alanine-dipeptide-3x250ns-heavy-atom-positions.npz"

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
    Nepochs = int(np.array(f["Nepochs"]))
    K = int(np.array(f["K"]))
    fix = np.array(f["fix"])
    scaling = float(np.array(f["scaling"]))
    
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
f.load(saved)

SAMPLES = []
for _ in range(100):
    samples = f.sample(7500)[0][:,:30].detach()
    SAMPLES.append(samples)
samples = torch.cat(SAMPLES,dim=0)
Phi, Psi = utils.alanineDipeptidePhiPsi(samples.reshape(-1,10,3))

np.savez("FlowSamples.npz",samples.numpy())

from matplotlib import pyplot as plt

H,xedges,yedges = np.histogram2d(Psi.detach().numpy().reshape(-1),Phi.detach().numpy().reshape(-1),bins=500)

Hnorm = -np.log(H/np.linalg.norm(H))

plt.figure()
plt.imshow(Hnorm, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap=plt.cm.jet)

plt.figure()
plt.hist2d(Phi.detach().numpy().reshape(-1),Psi.detach().numpy().reshape(-1),bins=100,cmap=plt.cm.jet)

plt.show()