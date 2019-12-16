from utils import stormerVerlet as _stormerVerlet
import flow
import torch

def stormerVerlet(eta,*args):
    q,p = eta.split(eta.shape[-1]//2,dim=-1)
    _Q,_P = _stormerVerlet(q,p,*args)
    return torch.cat([_Q,_P],dim=-1)

def buildSource(f):
    return flow.FlowNet([f.layerList[0]],f.prior).double()

def timeEvolve(flow,t,steps,batchSize,method="stomerVerlect",initalPoint=None):
    H = lambda q,p: flow.energy(torch.cat([q,p],dim=-1))
    if initalPoint is None:
        initalPoint = flow.sample(batchSize)[0]
    trajs = stormerVerlet(initalPoint,H,t,steps)
    return trajs