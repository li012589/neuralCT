import torch
import numpy as np
from numpy.testing import assert_array_almost_equal,assert_array_equal

import os
import sys
sys.path.append(os.getcwd())
from utils import assertMTJM,jacobian

def bijective(flow,batch=100,decimal=5):
    x,p = flow.sample(batch)
    z,ip = flow.forward(x)
    xz,gp = flow.inverse(z)
    op = flow.prior.logProbability(z)
    zx,ipp = flow.forward(xz)
    assert_array_almost_equal(x.detach().numpy(),xz.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(z.detach().numpy(),zx.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(ip.detach().numpy(),-gp.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(ip.detach().numpy(),ipp.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(p.detach().numpy(),(op-gp).detach().numpy(),decimal=decimal)

def saveload(flow,blankFlow,batch=100,decimal=5):
    x,p = flow.sample(batch)
    z,ip = flow.forward(x)
    d = flow.save()
    torch.save(d,"testsaving.saving")
    dd = torch.load("testsaving.saving")
    blankFlow.load(dd)
    op = blankFlow.prior.logProbability(z)
    xz,gp = blankFlow.inverse(z)
    assert_array_almost_equal(x.detach().numpy(),xz.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(ip.detach().numpy(),-gp.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(p.detach().numpy(),(op-gp).detach().numpy(),decimal=decimal)


def symplectic(flow,batch=100,decimal=5):
    tmp = flow.prior.sample(batch).requires_grad_()
    tmpi = flow.inverse(tmp)[0]
    M = jacobian(tmpi,tmp)
    assertMTJM(M,decimal)