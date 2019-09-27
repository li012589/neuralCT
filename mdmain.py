import numpy as np

import argparse

import torch
from torch import nn

import flow
import train
import utils

import math
import h5py

#torch.manual_seed(42)
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('Learning  parameters')
parser.add_argument("-folder", default=None,help = "Folder to save and load")
group.add_argument("-epochs", type=int, default=300, help="Number of epoches to train")
group.add_argument("-batch", type=int, default=200, help="Batch size of train")
group.add_argument("-cuda", type=int, default=-1, help="If use GPU")
group.add_argument("-lr", type=float, default=0.001, help="Learning rate")
group.add_argument("-save", action='store_true',help="If save or not")
group.add_argument("-load", action='store_true' ,help="If load or not")
group.add_argument("-save_period", type=int, default=10, help="Steps to save in train")
group.add_argument("-K",type=float, default=300, help="Temperature")
group.add_argument("-double", action='store_true',help="Use double or single")

group = parser.add_argument_group('Network parameters')
group.add_argument("-hdim", type=int, default=128, help="Hidden dimension of mlps")
group.add_argument("-numFlow", type=int, default=1, help="Number of flows")
group.add_argument("-nlayers", type=int, default=8, help="Number of mlps in rnvp")
group.add_argument("-nmlp", type=int, default=2, help="Number of layers of mlps")

group = parser.add_argument_group('Target parameters')
group.add_argument("-dataset", default="./database/alanine-dipeptide-3x250ns-heavy-atom-positions.npz", help="Path to training data")
group.add_argument("-baseDataSet",default=None,help="Known CV data base")
group.add_argument("-miBatch",type=int,default=5, help="Batch size when evaluate MI")
group.add_argument("-miSample",type=int,default=1000, help="Sample when evaluate MI")
group.add_argument("-loadrange",default=3,type=int,help="Array nos to load from npz file")
group.add_argument("-smile", default="CC(=O)NC(C)C(=O)NC",help="smile expression")
group.add_argument("-scaling",default=10,type=float,help = "Scaling factor of npz data, default is for nm to ångströms")
group.add_argument("-fixx",default=0,type=float,help="Offset of x axis")
group.add_argument("-fixy",default=0,type=float,help="Offset of y axis")
group.add_argument("-fixz",default=0,type=float,help="Offset of z axis")

group = parser.add_argument_group("Analysis parameters")
group.add_argument("-interpolation", default=0, type=int, help="Mode except 0,1 to interpolation")


args = parser.parse_args()

device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

if args.folder is None:
    rootFolder = './opt/Model_' + args.smile+"_Batch_"+str(args.batch)+"_T_"+str(args.K)+"_depthLevel_"+str(args.numFlow)+'_l'+str(args.nlayers)+'_M'+str(args.nmlp)+'_H'+str(args.hdim)+"/"
    print("No specified saving path, using",rootFolder)
else:
    rootFolder = args.folder
    print("Using specified path",args.folder)
if rootFolder[-1] != '/':
    rootFolder += '/'
utils.createWorkSpace(rootFolder)
if not args.load:
    n = 3*len([i for i in args.smile if i.isalpha()])
    numFlow = args.numFlow
    lossPlotStep = args.save_period
    hidden = args.hdim
    nlayers = args.nlayers
    nmlp = args.nmlp
    lr = args.lr
    batchSize = args.batch
    Nepochs = args.epochs
    K = args.K
    fix = np.array([args.fixx,args.fixy,args.fixz])
    scaling = args.scaling
    with h5py.File(rootFolder+"/parameter.hdf5","w") as f:
        f.create_dataset("n",data=n)
        f.create_dataset("numFlow",data=numFlow)
        f.create_dataset("lossPlotStep",data=lossPlotStep)
        f.create_dataset("hidden",data=hidden)
        f.create_dataset("nlayers",data=nlayers)
        f.create_dataset("nmlp",data=nmlp)
        f.create_dataset("lr",data=lr)
        f.create_dataset("batchSize",data=batchSize)
        f.create_dataset("Nepochs",data=Nepochs)
        f.create_dataset("K",data=K)
        f.create_dataset("fix",data=fix)
        f.create_dataset("scaling",data=scaling)
else:
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

from utils import MDSampler,loadmd
from utils import variance,smile2mass

loadrange = ["arr_" + str(i) for i in range(args.loadrange)]
dataset = loadmd(args.dataset,loadrange,scaling,fix).to(device)
SMILE = smile2mass(args.smile)

if not args.double:
    dataset = dataset.to(torch.float32)

if args.double:
    pVariance = torch.tensor([variance(torch.tensor(item).double(),K) for item in SMILE],dtype=torch.float64).reshape(1,-1).repeat(3,1).permute(1,0).reshape(-1)
else:
    pVariance = torch.tensor([variance(torch.tensor(item),K) for item in SMILE],dtype=torch.float32).reshape(1,-1).repeat(3,1).permute(1,0).reshape(-1)
target = MDSampler(dataset,pVariance = pVariance)

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

f = flowBuilder(n,numFlow,innerBuilder,1).to(device)

if not args.double:
    f = f.to(torch.float32)

if args.load:
    import os
    import glob
    name = max(glob.iglob(rootFolder+"savings/"+'*.saving'), key=os.path.getctime)
    print("load saving at "+name)
    saved = torch.load(name,map_location=device)
    f.load(saved)
    name = max(glob.iglob(rootFolder+"records/"+'*.hdf5'), key=os.path.getctime)
    with h5py.File(name,"r") as h5:
        LOSS =np.array(h5["LOSS"])
        LOSSVAL =np.array(h5["LOSSVAL"])
    d0 = f.layerList[0].elements[:n]
    d1 = f.layerList[0].elements[n:]
    omega = (1/(torch.exp(d0+d1))).detach()
    print("omega",omega)

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(np.sort(omega.numpy()))
    plt.figure()
    plt.plot(LOSSVAL)
    plt.plot(LOSS)

    plt.show()

    sample1 = target.sample(1)

    latent1 = f.forward(sample1)[0].detach()

    from copy import deepcopy
    lat1 = deepcopy(latent1)
    lat2 = deepcopy(latent1)

    omega, idx = torch.sort(omega)

    omega0 = 1/torch.exp(-f.layerList[0].elements[idx[0]])
    omega1 = 1/torch.exp(-f.layerList[0].elements[idx[1]])
    L = 8
    L2 = L*L
    #take the first sample and put it in the batch dimension
    latents = latent1.repeat(L2, 1)
    for j in range(L):
        Q0 = -omega0 + j/(L-1) * 2*omega0 - f.layerList[0].shift[idx[0]]
        for i in range(L):
            Q1 = -omega1 + i/(L-1) * 2*omega1 - f.layerList[0].shift[idx[1]]
            latents[i*L+j, idx[0]] = Q0
            latents[i*L+j, idx[1]] = Q1

    x = f.inverse(latents)[0].detach().numpy()[:,:n]
    np.savez(args.smile+'_interpolation.npz', x)
    print("Generated interpolation data:",args.smile+'_interpolation.npz')

    lats1 = lat1.repeat(100,1)
    for i in range(100):
        Q0 = -omega0 + i/(100-1) * 2*omega0 - f.layerList[0].shift[idx[0]]
        lats1[i,idx[0]]=Q0

    x1 = f.inverse(lats1)[0].detach().numpy()[:,:n]
    np.savez(args.smile+'_idx0.npz', x1)
    print("Generated mode 0 interpolation data:",args.smile+"_idx0.npz")

    lats2 = lat2.repeat(100,1)
    for i in range(100):
        Q1 = -omega1 + i/(100-1) * 2*omega1 - f.layerList[0].shift[idx[1]]
        lats2[i,idx[1]]=Q1

    x2 = f.inverse(lats2)[0].detach().numpy()[:,:n]
    np.savez(args.smile+'_idx1.npz', x2)
    print("Generated mode 1 interpolation data:",args.smile+"_idx1.npz")

    if args.interpolation > 2:
        omega2 = 1/torch.exp(-f.layerList[0].elements[idx[args.interpolation]])
        lat3 = deepcopy(latent1)
        lats3 = lat3.repeat(100,1)
        for i in range(100):
            Q2 = -omega2 + i/(100-1) * 2*omega2 - f.layerList[0].shift[idx[args.interpolation]]
            lats3[i,idx[1]]=Q2

        x2 = f.inverse(lats3)[0].detach().numpy()[:,:n]
        np.savez(args.smile+'_idx'+str(args.interpolation)+'.npz', x2)
        print("Generated mode "+str(args.interpolation)+" interpolation data:",args.smile+'_idx'+str(args.interpolation)+'.npz')

    data = loadmd(args.dataset,loadrange,scaling,fix).to(torch.float32)
    if args.dataset=="./database/alanine-dipeptide-3x250ns-heavy-atom-positions.npz" and args.baseDataSet is None:
        theta = loadmd("./database/alanine-dipeptide-3x250ns-backbone-dihedrals.npz",loadrange,1,[0,0,0]).to(torch.float32)
    elif args.baseDataSet is None:
        import pdb
        pdb.set_trace()
    else:
        theta = loadmd(args.baseDataSet,loadrange,1.0,[0.0,0.0,0.0])

    #randomly shuffle both
    Nsamples = 5
    Npersample = 1000
    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)
    data = data[perm][:Nsamples* Npersample, :]
    theta = theta[perm][:Nsamples* Npersample, :]

    batchsize, halfdim = data.shape[0], data.shape[1]
    p = torch.randn(batchsize,data.shape[-1]).to(data)*pVariance

    data = torch.cat([data,p], dim=1)

    z = f.forward(data)[0]
    #logp, z = model.logprob(data, return_z=True)
    print ('z', z, z.shape)
    z = z.detach().cpu().numpy()

    #from midemo import mi_batch
    from thirdparty import kraskov_mi
    mi_phi = []
    mi_psi = []
    Nk = 6
    for k in range(Nk):
        for sample in range(Nsamples):
            mi_phi.append(kraskov_mi(theta[sample*Npersample:(sample+1)*Npersample, 0].reshape(-1, 1), z[sample*Npersample:(sample+1)*Npersample, idx[k]].reshape(-1, 1) ))
            mi_psi.append( kraskov_mi(theta[sample*Npersample:(sample+1)*Npersample, 1].reshape(-1, 1), z[sample*Npersample:(sample+1)*Npersample, idx[k]].reshape(-1, 1) ))

    mi_phi = np.array(mi_phi)
    mi_phi = mi_phi.reshape(Nk, Nsamples)

    mi_psi = np.array(mi_psi)
    mi_psi = mi_psi.reshape(Nk, Nsamples)

    plt.errorbar(np.arange(Nk)+1, mi_phi.mean(axis=1), yerr=mi_phi.std(axis=1)/np.sqrt(Nsamples), fmt='o-', label='$I(Q_k:\Phi)$', markerfacecolor='none', markeredgewidth=2, capsize=8, lw=2)

    plt.errorbar(np.arange(Nk)+1, mi_psi.mean(axis=1), yerr=mi_psi.std(axis=1)/np.sqrt(Nsamples), fmt='o-', label='$I(Q_k:\Psi)$', markerfacecolor='none', markeredgewidth=2, capsize=8, lw=2)

    plt.show()


    import pdb
    pdb.set_trace()

LOSS = train.forwardLearn(target,f,batchSize,Nepochs,lr,saveSteps = lossPlotStep,savePath=rootFolder)

