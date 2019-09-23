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
    p = source.Gaussian([2,2])

    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.reshape(1,2,2)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    f = flow.NICE(maskList, [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)],p)
    bijective(f)
'''
def test_symplectic():
    p = source.Gaussian([8])

    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,8)
            #i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,:4] = 1
            b=b.reshape(1,8)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    f = flow.NICE(maskList, [utils.SimpleMLPreshape([8,32,32,8],[nn.ELU(),nn.ELU(),None]) for _ in range(4)],p)
    symplectic(f,decimal=4)
'''

def test_saveload():
    p = source.Gaussian([2,2])

    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.reshape(1,2,2)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    f = flow.NICE(maskList, [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)],p)
    p = source.Gaussian([2,2])

    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.reshape(1,2,2)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    blankf = flow.NICE(maskList, [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)],p)
    saveload(f,blankf)

if __name__ == "__main__":
    #test_bijective()
    test_symplectic()
