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

loadrange = ['arr_0','arr_1','arr_2']
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

# Load saving.

d0 = f.layerList[0].elements[:n]
d1 = f.layerList[0].elements[n:]
omega = (1/(torch.exp(d0+d1))).detach()

omega, idx = torch.sort(omega)

from utils import loadmd, variance, smile2mass
SMILE = smile2mass(smile)
pVariance = torch.tensor([variance(torch.tensor(item),K) for item in SMILE]).reshape(1,-1).repeat(3,1).permute(1,0).reshape(-1).to(dtype)
theta = loadmd("./database/alanine-dipeptide-3x250ns-backbone-dihedrals.npz",loadrange,1,[0,0,0]).to(dtype)
data = loadmd("./database/alanine-dipeptide-3x250ns-heavy-atom-positions.npz",loadrange,scaling,fix).to(dtype)

perm = np.arange(data.shape[0])
np.random.shuffle(perm)
data = data[perm]
theta = theta[perm]

batchsize, halfdim = data.shape[0], data.shape[1]
p = torch.randn(batchsize,data.shape[-1]).to(data)*pVariance
data = torch.cat([data,p], dim=1)

L = 8
L2 = L*L

sample = data[0].reshape(1,-1)

latent = f.forward(sample)[0].detach()

omega0 = 0.5/torch.exp(-f.layerList[0].elements[idx[0]])
omega1 = 0.5/torch.exp(-f.layerList[0].elements[idx[1]])

latent = latent[0].repeat(L2,1)
for j in range(L):
    Q0 = -omega0 + j/(L-1) * 2*omega0 - f.layerList[0].shift[idx[0]]
    for i in range(L):
        Q1 = -omega1 + i/(L-1) * 2*omega1 - f.layerList[0].shift[idx[1]]
        latent[i*L+j,idx[0]] = Q0
        latent[i*L+j,idx[1]] = Q1

interpolation = f.inverse(latent)[0].detach()

np.savez('MeshPlotInterpolation.npz', interpolation[:, :30].numpy())
print("saving interpolation in MeshPlotInterpolation.npz")
