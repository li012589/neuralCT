import torch
from torch import nn

from .flow import Flow

class PointTransformation(Flow):
    def __init__(self,flow,prior = None, name = "PointTransformation"):
        super(PointTransformation,self).__init__(prior,name)
        self.flow = flow

    def inverse(self,y):
        batchSize = y.shape[0]
        inverseLogjac = y.new_zeros(y.shape[0])
        q,p = y.split(y.shape[-1]//2,dim=1)
        Q = self.flow.inverse(q)[0]
        P = torch.autograd.grad((p*self.flow.forward(Q)[0]).sum(1),Q,grad_outputs=torch.ones(batchSize).to(y),create_graph=True)[0]
        return torch.cat([Q,P],dim=1),inverseLogjac

    def forward(self,x):
        batchSize = x.shape[0]
        forwardLogjac = x.new_zeros(x.shape[0])
        Q,P = x.split(x.shape[-1]//2,dim=1)
        q = self.flow.forward(Q)[0]
        p = torch.autograd.grad((P*self.flow.inverse(q)[0]).sum(1),q,grad_outputs=torch.ones(batchSize).to(x),create_graph=True)[0]
        return torch.cat([q,p],dim=1),forwardLogjac

    def transMatrix(self,sign=1):
        return None