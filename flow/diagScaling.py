import torch
from torch import nn
import numpy as np

from .flow import Flow

class DiagScaling(Flow):
    def __init__(self,shape,initValue=None,fix=None,prior=None,shift = False,name ="DiagScaling"):
        super(DiagScaling,self).__init__(prior,name)

        if initValue is None:
            initValue = [0]*shape
        if fix is None:
            fix = [1]*shape
        else:
            fix = [1-item for item in fix]

        assert shape == len(initValue)
        assert shape == len(fix)
        self.elements = nn.Parameter(torch.tensor(initValue,dtype=torch.float32))
        self.fix = nn.Parameter(torch.tensor(fix,dtype=torch.float32),requires_grad=False)
        if shift:
            self.shift = nn.Parameter(torch.tensor(0.01*np.random.randn(shape),dtype=torch.float32),requires_grad=True)
        else:
            self.shift = nn.Parameter(torch.zeros(shape),requires_grad=False)

    def inverse(self,y):
        inverseLogjac = torch.sum(self.elements*self.fix)
        return y*torch.exp(self.elements*self.fix)+self.shift,inverseLogjac

    def forward(self,z):
        forwardLogjac = -torch.sum(self.elements*self.fix)
        return z*torch.exp(-self.elements*self.fix)-self.shift,forwardLogjac

    def transMatrix(self,sign=1):
        if sign == 1:
            ele = self.elements
        else:
            ele = 1/self.elements
        return torch.diag(ele)