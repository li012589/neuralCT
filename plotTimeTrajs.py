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

from utils import loadmd, variance, smile2mass
SMILE = smile2mass(smile)
pVariance = torch.tensor([variance(torch.tensor(item),K) for item in SMILE]).reshape(1,-1).repeat(3,1).permute(1,0).reshape(-1).to(dtype)
data = loadmd("./database/alanine-dipeptide-3x250ns-heavy-atom-positions.npz",loadrange,scaling,fix).to(dtype)

perm = np.arange(data.shape[0])
np.random.shuffle(perm)
data = data[perm]
batchsize, halfdim = data.shape[0], data.shape[1]
p = torch.randn(batchsize,data.shape[-1]).to(data)*pVariance
data = torch.cat([data,p], dim=1)


Allsample = data[:,:30].reshape(-1,30)
AllPhi, AllPsi = utils.alanineDipeptidePhiPsi(Allsample.reshape(-1,10,3))

H,xedges,yedges = np.histogram2d(AllPsi.detach().numpy().reshape(-1),AllPhi.detach().numpy().reshape(-1),bins=500)

Hnorm = -np.log(H/np.linalg.norm(H))

Allsample = torch.from_numpy(np.load("./FlowSamples.npz")["arr_0"])
AllPhi, AllPsi = utils.alanineDipeptidePhiPsi(Allsample.reshape(-1,10,3))

H,xedges,yedges = np.histogram2d(AllPsi.detach().numpy().reshape(-1),AllPhi.detach().numpy().reshape(-1),bins=500)

Hnormp = -np.log(H/np.linalg.norm(H))

from utils import timeEvolve, buildSource
dtype = torch.float64
f = f.to(dtype)

sample = data[np.random.randint(data.shape[0],size=10)].reshape(10,-1).to(dtype)
latent = f.forward(sample)[0].detach()

latentSource = buildSource(f).to(dtype)
trajs = timeEvolve(latentSource,0.0005,10000,1,initalPoint=latent.to(dtype))
print(trajs)

trajs = trajs.to(dtype)
physicalTrajs = f.inverse(trajs.reshape(-1,trajs.shape[-1]))[0].reshape(trajs.shape)
physicalQtrajs = physicalTrajs[:,:,:30]
Phi, Psi = utils.alanineDipeptidePhiPsi(physicalQtrajs[:,0,:].reshape(-1,10,3))

angleTrajs = np.array([[Phi[i].item(),Psi[i].item()] for i in range(Phi.shape[0])])

print(angleTrajs.shape)

from matplotlib import pyplot as plt

plt.figure()
plt.imshow(Hnorm, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap=plt.cm.jet)

#plt.scatter(angleTrajs[:,0],angleTrajs[:,1])
plt.plot(angleTrajs[:,0],angleTrajs[:,1])

plt.figure()
plt.imshow(Hnormp, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap=plt.cm.jet)
#plt.scatter(angleTrajs[:,0],angleTrajs[:,1])
plt.plot(angleTrajs[:,0],angleTrajs[:,1])
plt.show()
