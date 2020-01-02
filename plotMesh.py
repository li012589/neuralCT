import numpy as np
import torch
from torch import nn
import flow
import train
import utils
import math
import h5py

rootFolder = "./demo/Model_CC(=O)NC(C)C(=O)NC_Batch_200_T_300_depthLevel_1_l8_M2_H128/"
device = torch.device("cpu")
dtype = torch.float32
smile = "CC(=O)NC(C)C(=O)NC"
dataset = "./database/alanine-dipeptide-3x250ns-heavy-atom-positions.npz"

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

from matplotlib import pyplot as pl
from matplotlib.offsetbox import AnchoredText

from utils import loadmd, variance, smile2mass
SMILE = smile2mass(smile)
pVariance = torch.tensor([variance(torch.tensor(item),K) for item in SMILE]).reshape(1,-1).repeat(3,1).permute(1,0).reshape(-1).to(dtype)
theta = loadmd("./database/alanine-dipeptide-3x250ns-backbone-dihedrals.npz",loadrange,1,[0,0,0]).to(dtype)
data = loadmd("./database/alanine-dipeptide-3x250ns-heavy-atom-positions.npz",loadrange,scaling,fix).to(dtype)

perm = np.arange(data.shape[0])
np.random.shuffle(perm)
data = data[perm]
theta = theta[perm]

ximgs = torch.from_numpy(np.load("meshAngles.npz")["arr_0"])

Phi, Psi = utils.alanineDipeptidePhiPsi(ximgs.reshape(-1,10,3))

Allsample = data
AllPhi, AllPsi = utils.alanineDipeptidePhiPsi(Allsample.reshape(-1,10,3))
H,xedges,yedges = np.histogram2d(AllPsi.detach().numpy().reshape(-1),AllPhi.detach().numpy().reshape(-1),bins=500)

Hnorm = -np.log(H/np.linalg.norm(H))
plt.imshow(Hnorm, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap=plt.cm.jet)
ax = plt.gca()
at2 = AnchoredText("(b)",loc='lower center', prop=dict(size=17), frameon=False,bbox_to_anchor=(0., 1.),bbox_transform=ax.transAxes)
ax.add_artist(at2)
plt.ylim(-70,180)
plt.ylabel("$Dihedral\ angles, \Psi$",fontsize="x-large")
plt.xlabel("$Dihedral\ angles, \Phi$",fontsize="x-large")
plt.scatter(Phi.detach().numpy(), Psi.detach().numpy(),color="crimson")
plt.plot(Phi.detach().numpy(),Psi.detach().numpy(),color="crimson",linewidth=3)
plt.savefig("MeshAngles.pdf")
plt.show()