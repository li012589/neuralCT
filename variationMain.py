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
group.add_argument("-hdim", type=int, default=680, help="Hidden dimension of mlps")
group.add_argument("-numFlow", type=int, default=1, help="Number of flows")
group.add_argument("-nlayers", type=int, default=16, help="Number of mlps in rnvp")
group.add_argument("-nmlp", type=int, default=3, help="Number of layers of mlps")
group.add_argument("-shift",action="store_false",help="Shift latent variable or not")
group.add_argument("-relax",action="store_false",help="Trainable latent p or not")

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
    target = source.HarmonicChain(4,1)
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

    '''
    t0 = t[0].reshape(1,2).detach().requires_grad_()
    print("jacobian:",utils.jacobian(qf.forward(t0)[0],t0))
    t = t.detach().numpy()
    T = T.detach().numpy()
    plt.scatter(t[:,0],t[:,1])
    plt.scatter(T[:,0],T[:,1])
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    #ax1.scatter(t[:,1]-t[:,0],T[:,0])
    ax1.scatter(t[:,0],T[:,0])
    plt.title("q1 v.s. Q1")
    ax2 = fig.add_subplot(222)
    #ax2.scatter(t[:,1]-t[:,0],T[:,1])
    ax2.scatter(t[:,0],T[:,1])
    plt.title("q1 v.s. Q2")
    ax3 = fig.add_subplot(223)
    #ax3.scatter(t[:,1]+t[:,0],T[:,0])
    ax3.scatter(t[:,1],T[:,0])
    plt.title("q2 v.s. Q1")
    ax4 = fig.add_subplot(224)
    #ax4.scatter(t[:,1]+t[:,0],T[:,1])
    ax4.scatter(t[:,1],T[:,1])
    plt.title("q2 v.s. Q2")

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.scatter(t[:,1]-t[:,0],T[:,0])
    plt.title("q2-q1 v.s. Q1")
    #ax1.scatter(t[:,0],T[:,0])
    ax2 = fig.add_subplot(222)
    ax2.scatter(t[:,1]-t[:,0],T[:,1])
    plt.title("q2-q1 v.s. Q2")
    #ax2.scatter(t[:,0],T[:,1])
    ax3 = fig.add_subplot(223)
    ax3.scatter(t[:,1]+t[:,0],T[:,0])
    plt.title("q2+q1 v.s. Q1")
    #ax3.scatter(t[:,1],T[:,0])
    ax4 = fig.add_subplot(224)
    ax4.scatter(t[:,1]+t[:,0],T[:,1])
    plt.title("q2+q1 v.s. Q2")
    #ax4.scatter(t[:,1],T[:,1])
    '''

    '''
    mX = np.linspace(-5,13,900)
    mY = np.linspace(-5,17,1100)
    X, Y = np.meshgrid(mX, mY)
    XY =torch.cat([torch.tensor(X.reshape(-1,1)),torch.tensor(Y.reshape(-1,1))],dim=1).float()
    Z = qf.forward(XY)[0]
    Z0 = Z[:,0].reshape(1100,900).detach().numpy()
    Z1 = Z[:,1].reshape(1100,900).detach().numpy()
    plt.figure()
    plt.contour(X,Y,Z0)
    plt.title("Q1 vs q1,q2")
    plt.figure()
    plt.contour(X,Y,Z1)
    plt.title("Q2 vs q1,q2")
    '''
    import pdb
    pdb.set_trace()

LOSS = train.learn(target,f,batchSize,epochs,lr,saveSteps = lossPlotStep,savePath=rootFolder)





