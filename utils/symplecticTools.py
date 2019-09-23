import torch
import numpy as np
from numpy.testing import assert_array_almost_equal,assert_array_equal

def J(n,batchSize=1,dtype=torch.double):
    n = n//2
    I = lambda n: torch.eye(n).to(dtype)
    O = lambda n: torch.zeros(n,n).to(dtype)
    j = torch.cat([torch.cat([O(n),I(n)],1),torch.cat([-I(n),O(n)],1)],0)
    j = j.repeat(batchSize,1,1)
    return j

def MTJM(M):
    j = J(M.shape[-1],batchSize=M.shape[0]).to(M)
    return torch.matmul(torch.matmul(M.permute(0,2,1),j),M)

def assertMTJM(M,decimal=6):
    j = J(M.shape[-1],batchSize=M.shape[0]).to(M)
    assert_array_almost_equal(torch.matmul(torch.matmul(M.permute(0,2,1),j),M).detach().numpy(),j.detach().numpy(),decimal=decimal)
