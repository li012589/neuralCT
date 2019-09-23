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
    f = flow.Scaling(4,[2,3],p)
    bijective(f)

def test_saveload():
    p = source.Gaussian([4])
    f = flow.Scaling(4,[2,3],p)
    blankf = flow.Scaling(4,prior=p)
    saveload(f,blankf)

if __name__ == "__main__":
    test_saveload()
