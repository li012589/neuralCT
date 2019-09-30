
import os
import sys
sys.path.insert(0,os.getcwd())

from test import *
import torch
from torch import nn
import numpy as np
import utils
import flow
import source
import utils

def test_bijective():
    p = source.Gaussian([8])
    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[i] = 1
            b=b.reshape(1,4)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)

    fl = flow.RNVP(maskList, [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)])
    f = flow.PointTransformation(fl,p)
    bijective(f)

def test_saveload():
    p = source.Gaussian([8])
    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[i] = 1
            b=b.reshape(1,4)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)

    fl = flow.RNVP(maskList, [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)])
    f = flow.PointTransformation(fl,p)

    p = source.Gaussian([8])
    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[i] = 1
            b=b.reshape(1,4)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)

    fl = flow.RNVP(maskList, [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)])
    blankf = flow.PointTransformation(fl,p)
    saveload(f,blankf)

def test_symplectic():
    p = source.Gaussian([8])
    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[i] = 1
            b=b.reshape(1,4)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)

    fl = flow.RNVP(maskList, [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)])
    f = flow.PointTransformation(fl,p)
    symplectic(f)


if __name__ == "__main__":
    test_bijective()