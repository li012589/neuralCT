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

LOSS = train.forwardLearn(target,f,batchSize,Nepochs,lr,saveSteps = lossPlotStep,savePath=rootFolder)

