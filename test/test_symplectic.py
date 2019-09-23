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
    p = source.Gaussian([6])

    f = flow.Symplectic(6,prior=p)
    bijective(f)

def test_symplectic():
    p = source.Gaussian([6])

    f = flow.Symplectic(6,prior=p)
    symplectic(f,decimal=4)

def test_saveload():
    p = source.Gaussian([6])
    f = flow.Symplectic(6,prior=p)

    p = source.Gaussian([6])
    blankf = flow.Symplectic(6,prior=p)

    saveload(f,blankf)

if __name__ == "__main__":
    #test_bijective()
    test_symplectic()