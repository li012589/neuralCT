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

from numpy.testing import assert_array_almost_equal,assert_almost_equal

def test_fix():
    p = source.Gaussian([4])
    f = flow.DiagScaling(4,initValue=[2,3,4,5],fix = [1,0,1,0],prior=p)
    t = torch.tensor([2,3,4,5],dtype=torch.float32)
    x,p = f.inverse(t)
    assert_array_almost_equal(x.detach().numpy(),t*torch.exp(torch.tensor([0.0,3.0,0.0,5.0])))
    assert_almost_equal(p.item(),3+5)

def test_bijective():
    p = source.Gaussian([4])
    f = flow.DiagScaling(4,prior=p)
    bijective(f)

def test_saveload():
    p = source.Gaussian([4])
    f = flow.DiagScaling(4,initValue=[0.01,0.02,0.03,0.04],prior=p)
    blankf = flow.DiagScaling(4,prior=p)
    saveload(f,blankf,decimal=5)

if __name__ == "__main__":
    test_bijective()
    test_saveload()
