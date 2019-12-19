import torch
import numpy as np

import flow
import source
import train
import utils

import math


def flowBuilder(n,numFlow,innerBuilder=None,typeLayer=3,relax=False,shift=False):
    nn = n*2
    op = source.Gaussian([nn]).to(torch.float64)

    if innerBuilder is None:
        raise Exception("innerBuilder is None")
    if relax:
        f3 = flow.DiagScaling(nn,initValue=0.1*np.random.randn(nn),fix=[0]*n+[0]*n,shift=shift)
    else:
        f3 = flow.DiagScaling(nn,initValue=0.1*np.random.randn(nn),fix=[0]*n+[1]*n,shift=shift)
    layers=[f3]
    if typeLayer == 0:
        layers.append(flow.Symplectic(nn))
    else:
        for d in range(numFlow):
            if typeLayer == 3:
                layers.append(flow.PointTransformation(innerBuilder(n)))
                layers.append(flow.Symplectic(nn))
            elif typeLayer ==2:
                layers.append(flow.Symplectic(nn))
            elif typeLayer ==1:
                layers.append(flow.PointTransformation(innerBuilder(n)))
            elif typeLayer!=0:
                raise Exception("No such type")
    return flow.FlowNet(layers,op).double()

def extractFlow(flowCon):
    from copy import deepcopy
    layers = []
    _op = deepcopy(flowCon.prior)
    _rnvp = deepcopy(flowCon.layerList[1].flow)
    _diag = flowCon.layerList[0]
    nn = _diag.shift.shape[0]//2
    layers.append(flow.DiagScaling(nn,initValue=_diag.elements.clone().detach()[:nn]))
    layers.append(_rnvp)
    return flow.FlowNet(layers,_op).double()