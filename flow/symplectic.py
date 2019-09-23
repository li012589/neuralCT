import torch
import utils
from torch import nn

from .flow import Flow

class Symplectic(Flow):
    def __init__(self,dim,initScale=1e-2,rtol=1e-8,prior=None,name = "Symplectic"):
        super(Symplectic,self).__init__(prior,name)
        halfDim = dim//2
        self.A = nn.Parameter(initScale*torch.randn(halfDim,halfDim))
        self.Bmeta = nn.Parameter(initScale*torch.randn(halfDim,halfDim))
        self.Cmeta = nn.Parameter(initScale*torch.randn(halfDim,halfDim))
        '''
        self.A = nn.Parameter(0.18232159*100*torch.eye(halfDim))
        self.Bmeta = nn.Parameter(torch.zeros(halfDim,halfDim))
        self.Cmeta = nn.Parameter(torch.zeros(halfDim,halfDim))
        '''
        self.rtol = rtol

    def _metaProcess(self,y,sign=1):
        inverseLogjac = y.new_zeros(y.shape[0])
        B = self.Bmeta+self.Bmeta.t()
        C = self.Cmeta+self.Cmeta.t()
        Q = torch.cat((torch.cat((self.A,B),1),torch.cat((C,-self.A.t()),1)),0)
        return utils.expmv(sign*Q,y.permute(1,0),rtol=self.rtol).permute(1,0),inverseLogjac

    def inverse(self,y):
        return self._metaProcess(y,1)

    def forward(self,z):
        return self._metaProcess(z,-1)

    def transMatrix(self,sign=1):
        B = self.Bmeta+self.Bmeta.t()
        C = self.Cmeta+self.Cmeta.t()
        Q = torch.cat((torch.cat((self.A,B),1),torch.cat((C,-self.A.t()),1)),0)
        return utils.expm(sign*Q,rtol=self.rtol)