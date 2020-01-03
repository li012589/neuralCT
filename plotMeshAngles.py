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
#p = torch.randn(batchsize,data.shape[-1]).to(data)*pVariance
#data = torch.cat([data,p], dim=1)

target1Phi = -150
target1Psi = -25

target2Phi = -75
target2Psi = -25

Allsample = data
AllPhi, AllPsi = utils.alanineDipeptidePhiPsi(Allsample.reshape(-1,10,3))

id1Phi = torch.nonzero(AllPhi.to(torch.int) == target1Phi)[:,0].reshape(-1).numpy()
id1Psi = torch.nonzero(AllPsi.to(torch.int) == target1Psi)[:,0].reshape(-1).numpy()

id1 = int(np.intersect1d(id1Phi,id1Psi)[0])

id2Phi = torch.nonzero(AllPhi.to(torch.int) == target2Phi)[:,0].reshape(-1).numpy()
id2Psi = torch.nonzero(AllPsi.to(torch.int) == target2Psi)[:,0].reshape(-1).numpy()

id2 = int(np.intersect1d(id2Phi,id2Psi)[0])

frnvp = utils.extractFlow(f).to(dtype)

sample1 = data[id1].reshape(1,-1)
sample2 = data[id2].reshape(1,-1)

zsample1 = frnvp.forward(sample1)[0].detach()
zsample2 = frnvp.forward(sample2)[0].detach()

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

L = 10

'''
delta = (zsample2-zsample1)/L
dim=2

SAMPLES = []
for i in range(int(L+1)):
    _tmp = deepcopy(zsample1)
    _dim = idx[:dim]
    _tmp[:,_dim] = zsample1[:,_dim]+(delta*i)[:,_dim]
    SAMPLES.append(frnvp.inverse(_tmp)[0].detach())
ximgs = torch.cat(SAMPLES,0).detach()

'''

dims = [5,15,30]
ZSAMPLES=[[] for _ in range(len(dims))]
for j,dim in enumerate(dims):
    for i in range(int(L+1)):
        _tmp = deepcopy(zsample1)
        _dim = idx[:dim]
        _tmp[:,_dim] = utils.slerp(i/float(L),zsample1[:,_dim].reshape(-1),zsample2[:,_dim].reshape(-1)).detach()
        ZSAMPLES[j].append(_tmp)
    #ZSAMPLES[j].append(zsample2)
Zs = [torch.cat(term,0) for term in ZSAMPLES]
Xs = [frnvp.inverse(term)[0].detach() for term in Zs]

EnXs = torch.cat([frnvp.energy(term) for term in Xs]).detach().reshape(3,-1).numpy()
EnZs = torch.cat([frnvp.prior.energy(term) for term in Zs]).detach().reshape(3,-1).numpy()

np.savez("meshAnglesp.npz",arr0 = Xs[0].numpy(),arr1 = Xs[1].numpy(),arr2 = Xs[2].numpy())
print("saving mesh data at meshAngles.npz")

figx = plt.figure(figsize=(10,8))
axx = figx.subplots(len(EnXs),1)
ax = axx[0]
at2 = AnchoredText("(a)",loc='lower center', prop=dict(size=17), frameon=False,bbox_to_anchor=(0., 1.),bbox_transform=ax.transAxes)
ax.add_artist(at2)

steps = np.arange(0,EnXs.shape[1])

for i in range(len(EnXs)):
    axx1 = axx[i]
    axx1.plot(EnXs[i,:],color="blue")
    axx1.scatter(steps,EnXs[i,:],color="blue")
    axx1.tick_params(axis='y', labelcolor="blue")
    if i != len(EnXs)-1:
        axx1.set_xticks([])
    axx2 = axx1.twinx()
    axx2.plot(EnZs[i,:],color="orange")
    axx2.scatter(steps,EnZs[i,:],color="orange")
    if i == 1:
        axx1.set_ylabel('Physical Energy',fontsize="x-large")
        axx2.set_ylabel('Gaussian Energy',fontsize="x-large")
    axx2.tick_params(axis='y', labelcolor="orange")

axx1.set_xlabel('steps', fontsize="x-large")
plt.savefig("MeshAnglesA.pdf")

plt.figure(figsize=(10,4))
ax = plt.gca()
at2 = AnchoredText("(b)",loc='lower center', prop=dict(size=17), frameon=False,bbox_to_anchor=(0., 1.),bbox_transform=ax.transAxes)
ax.add_artist(at2)
PHI=list()
PSI=list()
for i in range(3):
    Phi, Psi = utils.alanineDipeptidePhiPsi(Xs[i].reshape(-1,10,3))
    PHI.append(Phi.detach().reshape(-1))
    PSI.append(Psi.detach().reshape(-1))

H,xedges,yedges = np.histogram2d(AllPsi.detach().numpy().reshape(-1),AllPhi.detach().numpy().reshape(-1),bins=500)
Hnorm = -np.log(H/np.linalg.norm(H))

plt.imshow(Hnorm, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap=plt.cm.jet)
plt.ylim(-50,50)
plt.ylabel("$\Psi$",fontsize="x-large")
plt.xlabel("$\Phi$",fontsize="x-large")
color = ['darkblue','darkgreen','crimson']
for i in range(3):
    plt.scatter(PHI[i].detach().numpy(), PSI[i].detach().numpy(),color=color[i])
    plt.plot(PHI[i].detach().numpy(),PSI[i].detach().numpy(),color=color[i],label=str(i+1)+"th trajectory")
plt.legend()
plt.savefig("MeshAnglesB.pdf")
plt.show()

