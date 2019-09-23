from .flowRelated import *

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import numpy as np
import utils
import flow
import source


def test_bijective():
    p = source.Gaussian([4])
    f1 = flow.Scaling(4,[2,3])
    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.reshape(1,4)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    f2 = flow.NICE(maskList,[utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)])
    f = flow.FlowNet([f1,f2],p)
    bijective(f)

def test_saveload():
    p = source.Gaussian([4])
    f1 = flow.Scaling(4,[2,3])
    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.reshape(1,4)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    f2 = flow.NICE(maskList,[utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)])
    f = flow.FlowNet([f1,f2],p)
    f1 = flow.Scaling(4)
    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.reshape(1,4)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    f2 = flow.NICE(maskList,[utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)])
    blankf = flow.FlowNet([f1,f2],p)
    saveload(f,blankf)

if __name__ == "__main__":
    test_bijective()
    test_saveload()
