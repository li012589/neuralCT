import numpy as np

import argparse

import torch
from torch import nn

import flow
import source
import hamilton
import train
import utils

import math
import h5py

#torch.manual_seed(42)
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('learning  parameters')
parser.add_argument("-folder", default=None,help = "folder to save and load")
group.add_argument("-Nepochs", type=int, default=1000, help="")
group.add_argument("-batch", type=int, default=1000, help="")
group.add_argument("-cuda", type=int, default=-1, help="use GPU")
group.add_argument("-lr", type=float, default=0.001, help="learning rate")
group.add_argument("-save", action='store_true',help="if save or not")
group.add_argument("-load", action='store_true' ,help="if load or not")
group.add_argument("-seed", type=int, default=-1 ,help="manual_seed or not")
group.add_argument("-save_period", type=int, default=10, help="")
group.add_argument("-K",type=float, default=300, help="Temperature")
group.add_argument("-double", action='store_true',help="use double or single")

group = parser.add_argument_group('network parameters')
group.add_argument("-hdim", type=int, default=128, help="")
group.add_argument("-numFlow", type=int, default=1, help="depth")
group.add_argument("-nlayers", type=int, default=8, help="depth")
group.add_argument("-nmlp", type=int, default=2, help="depth")
group.add_argument("-typeLayer", type=int, default=1, help="")
group.add_argument("-shift",action="store_false",help="shift or not")
group.add_argument("-relax",action="store_false",help="relax or not")

group = parser.add_argument_group('target parameters')
group.add_argument("-n",type=int, default=784,help="number of dimensions")
group.add_argument("-dataset", default="./mnist.npz", help="path to training data")

args = parser.parse_args()

device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

if args.folder is None:
    rootFolder = './opt/MNIST_relax_'+str(args.relax)+"_shift_"+str(args.shift) +"_T_"+str(args.K)+"_depthLevel_"+str(args.numFlow)+'_l'+str(args.nlayers)+'_M'+str(args.nmlp)+'_H'+str(args.hdim)+"/"
    print("No specified saving path, using",rootFolder)
else:
    rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'
utils.createWorkSpace(rootFolder)
if not args.load:
    n = args.n
    numFlow = args.numFlow
    lossPlotStep = args.save_period
    hidden = args.hdim
    nlayers = args.nlayers
    nmlp = args.nmlp
    lr = args.lr
    batchSize = args.batch
    Nepochs = args.Nepochs
    typeLayer = args.typeLayer
    K = args.K
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
        f.create_dataset("typeLayer",data=typeLayer)
        f.create_dataset("K",data=K)
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
        typeLayer = int(np.array(f["typeLayer"]))
        K = int(np.array(f["K"]))

from dataloader import MDSampler,load

loadrange = ["arr_0"]
dataset = load(args.dataset).to(device)

target = MDSampler(dataset)

def innerBuilder(n):
    maskList = []
    for i in range(nlayers):
        if i %2==0:
            b = torch.zeros(n)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[i] = 1
            b=b.reshape(1,n)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    fl = flow.RNVP(maskList, [utils.SimpleMLPreshape([n]+[hidden]*nmlp+[n],[nn.Softplus()]*nmlp+[None]) for _ in range(nlayers)], [utils.SimpleMLPreshape([n]+[hidden]*nmlp+[n],[nn.Softplus()]*nmlp+[utils.ScalableTanh(n)]) for _ in range(nlayers)])
    return fl

from flowBuilder import flowBuilder2

f = flowBuilder2(n,numFlow,innerBuilder,typeLayer,relax=args.relax,shift=args.shift).to(device)

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
    omega = (1/(torch.exp(d0+d1))).detach().numpy()
    print("omega",omega)

    from matplotlib import pyplot as plt

    import pdb
    pdb.set_trace()

LOSS = train.forwardLearn(target,f,batchSize,Nepochs,lr,saveSteps = lossPlotStep,savePath=rootFolder)

import pdb
pdb.set_trace()