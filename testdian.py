import utils

import numpy as np
import torch
from torch import nn
import flow
import train
import utils
import math
import h5py

smile = "CC(=O)NC(C)C(=O)NC"
K = 300
dtype = torch.float32
loadrange = ['arr_0','arr_1','arr_2']
scaling = 10.0
fix = [0. ,     2.3222, 0.    ]
Nsamples = 5
Npersample = 1000

from utils import loadmd, variance, smile2mass
SMILE = smile2mass(smile)
pVariance = torch.tensor([variance(torch.tensor(item),K) for item in SMILE]).reshape(1,-1).repeat(3,1).permute(1,0).reshape(-1).to(dtype)
theta = loadmd("./database/alanine-dipeptide-3x250ns-backbone-dihedrals.npz",loadrange,1,[0,0,0]).to(dtype)
data = loadmd("./database/alanine-dipeptide-3x250ns-heavy-atom-positions.npz",loadrange,scaling,fix).to(dtype)

#perm = np.arange(data.shape[0])
#np.random.shuffle(perm)
#data = data[perm][:Nsamples* Npersample, :]
#theta = theta[perm][:Nsamples* Npersample, :]

sampletheta = theta.reshape(-1,2)
print(sampletheta.shape)
sample = data[:,:30].reshape(-1,30)
Phi, Psi = utils.alanineDipeptidePhiPsi(sample.reshape(-1,10,3))
#print(Phi.detach().numpy(),Psi.detach().numpy())
#print((sampletheta*180/np.pi).numpy())

H,xedges,yedges = np.histogram2d(Psi.detach().numpy().reshape(-1),Phi.detach().numpy().reshape(-1),bins=500)

from matplotlib import pyplot as plt

plt.figure()
plt.hist2d(Phi.detach().numpy().reshape(-1),Psi.detach().numpy().reshape(-1),bins=100,cmap=plt.cm.jet)

plt.figure()
plt.imshow(H, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap=plt.cm.jet)

Hnorm = -np.log(H/np.linalg.norm(H))
plt.figure()
plt.imshow(Hnorm, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap=plt.cm.jet)


plt.show()