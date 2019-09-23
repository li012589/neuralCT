import torch
from torch import nn

from .flow import Flow
import numpy as np

#from numpy.testing import assert_array_almost_equal

class Scaling(Flow):
    def __init__(self,shape,lamb=None,prior=None,name="Scaling"):
        super(Scaling,self).__init__(prior,name)
        if lamb is None:
            lamb = [1]*(shape//2)
        assert len(lamb) == shape//2
        self.lamb = torch.nn.Parameter(torch.tensor(lamb).to(torch.float32),requires_grad=True)
        #assert_almost_equal(np.prod(lamb),1.0)

    def inverse(self,y):
        tmp = torch.diag(torch.cat((self.lamb,1/self.lamb)))
        inverseLogjac = y.new_zeros(y.shape[0])
        #y = torch.matmul(tmp,y.permute(1,0)).permute(1,0)
        y = torch.matmul(y,tmp)
        return y,inverseLogjac

    def forward(self,z):
        tmp = torch.diag(torch.cat((1/self.lamb,self.lamb)))
        forwardLogjac = z.new_zeros(z.shape[0])
        #z = torch.matmul(tmp,z.permute(1,0)).permute(1,0)
        z = torch.matmul(z,tmp)
        return z,forwardLogjac

    def transMatrix(self,sign=1):
        if sign == 1:
            tmp = torch.diag(torch.cat((self.lamb,1/self.lamb)))
        else:
            tmp = torch.diag(torch.cat((1/self.lamb,self.lamb)))
        return tmp