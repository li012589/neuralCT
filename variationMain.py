import numpy as np

import argparse

import torch
from torch import nn

import flow
import source
import train
import utils

import math
import h5py

#torch.manual_seed(42)
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('Learning  parameters')
parser.add_argument("-folder", default=None,help = "Folder to save and load")
group.add_argument("-epochs", type=int, default=5000, help="Number of epoches to train")
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
group.add_argument("-shift",action="store_true",help="Shift latent variable or not")
group.add_argument("-relax",action="store_true",help="Trainable latent p or not")

group = parser.add_argument_group('Target parameters')
group.add_argument("-source",type=int, default=0,help="# of source, 0 for Ring2d, 1 for HarmonicChain")

args = parser.parse_args()

device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

if args.folder is None:
    rootFolder = './opt/variationModel_source_'+str(args.source)+'_Shift_'+str(args.shift) +"_T_"+str(args.K)+"_depthLevel_"+str(args.numFlow)+'_l'+str(args.nlayers)+'_M'+str(args.nmlp)+'_H'+str(args.hdim)+"/"
    print("No specified saving path, using",rootFolder)
else:
    rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'
utils.createWorkSpace(rootFolder)
if not args.load:
    numFlow = args.numFlow
    lossPlotStep = args.save_period
    hidden = args.hdim
    nlayers = args.nlayers
    nmlp = args.nmlp
    lr = args.lr
    batchSize = args.batch
    epochs = args.epochs
    K = args.K
    with h5py.File(rootFolder+"/parameter.hdf5","w") as f:
        f.create_dataset("numFlow",data=numFlow)
        f.create_dataset("lossPlotStep",data=lossPlotStep)
        f.create_dataset("hidden",data=hidden)
        f.create_dataset("nlayers",data=nlayers)
        f.create_dataset("nmlp",data=nmlp)
        f.create_dataset("lr",data=lr)
        f.create_dataset("batchSize",data=batchSize)
        f.create_dataset("epochs",data=epochs)
        f.create_dataset("K",data=K)
else:
    with h5py.File(rootFolder+"/parameter.hdf5","r") as f:
        numFlow = int(np.array(f["numFlow"]))
        lossPlotStep = int(np.array(f["lossPlotStep"]))
        hidden = int(np.array(f["hidden"]))
        nlayers = int(np.array(f["nlayers"]))
        nmlp = int(np.array(f["nmlp"]))
        lr = int(np.array(f["lr"]))
        batchSize = int(np.array(f["batchSize"]))
        epochs = int(np.array(f["epochs"]))
        K = int(np.array(f["K"]))

if args.source == 0:
    target = source.Ring2d()
elif args.source == 1:
    target = source.HarmonicChain(32,1)
else:
    raise Exception("No such source for target")

n = target.nvars[0]

if args.double:
    target = target.to(torch.float64)

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

f = flowBuilder(n,numFlow,innerBuilder,1,relax=args.relax,shift=args.shift).to(device)

if not args.double:
    f = f.to(torch.float32)

if args.load:
    import os
    import glob
    from matplotlib import pyplot as plt
    name = max(glob.iglob(rootFolder+"savings/"+'*.saving'), key=os.path.getctime)
    print("load saving at "+name)
    saved = torch.load(name,map_location=device)
    f.load(saved)

    d0 = f.layerList[0].elements[:2]
    d1 = f.layerList[0].elements[2:]
    omega = (1/(torch.exp(d0+d1)))
    print("shift:",f.layerList[0].shift)
    print("scaling:",f.layerList[0].elements)
    print("omega0:",omega[0].item(),"omega1:",omega[1].item())

    if args.source == 0:
        t =f.sample(1000)[0]
        z = f.forward(t)[0].detach().numpy()
        r = torch.sqrt(t[:,0]**2+t[:,1]**2).detach().numpy()
        theta = torch.atan2(t[:,1],t[:,0]).detach().numpy()

        plt.figure()
        plt.scatter(z[:,0],r)
        plt.scatter(z[:,0],theta)
        plt.title("Q1")

        plt.figure()
        plt.scatter(z[:,1],r)
        plt.scatter(z[:,1],theta)
        plt.title("Q2")

        plt.show()

    elif args.source == 1:
        fig = plt.figure(figsize=(8, 5))    

        plt.subplot(121)
        omega, idx = torch.sort(omega)

        klist = np.arange(len(omega)) +1

        omegalist = 2*np.sin(klist*np.pi/(2*len(omega)+2))

        plt.plot(klist, omega.detach().cpu().numpy(), 'o', color=colors[0], markerfacecolor='none', markeredgewidth=2)
        plt.plot(klist, omegalist, color=colors[0], lw=2, label='analytical')
        plt.xlabel('$k$')
        plt.ylabel('$\omega_k$')
        plt.legend(loc='lower right', frameon=False)

        plt.subplot(122)
        idx = idx[:2]

        batch_size = 1
        dim = 64
        x,_ = f.sample(batch_size)
        z,_ = f.inverse(x)

        jacobian = torch.zeros(batch_size, dim, dim)
        for i in range(dim):
            jacobian[:, i, :] = torch.autograd.grad(z[:, i], x, grad_outputs=torch.ones(batch_size, device=x.device), create_graph=True)[0]

        j = np.arange(dim//2)+1

        data = jacobian.detach().numpy()#
        sign = [1, -1]
        for batch in range(batch_size):
            for n, i in enumerate(idx):#
                plt.plot(j, data[batch, dim//2+i, dim//2:], 'o', label='$k=%g$'%(n+1), color=colors[n], markerfacecolor='none', markeredgewidth=2)#
                plt.plot(j, sign[n]*np.sqrt(2/(dim//2+1))*np.sin(j*(n+1)*np.pi/(dim//2+1)), '-', color=colors[n], lw=2)

        plt.legend(handlelength=1, frameon=False)

        plt.xlabel('$i$')
        plt.ylabel(r'$\nabla_{q_i} Q_k$')
        plt.show()

    else:
        raise Exception("No such source")

    import pdb
    pdb.set_trace()

LOSS = train.learn(target,f,batchSize,epochs,lr,saveSteps = lossPlotStep,savePath=rootFolder)





