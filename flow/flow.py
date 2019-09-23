
import numpy as np
import torch
from torch import nn

class Flow(nn.Module):

    def __init__(self, prior = None,name = "Flow"):
        super(Flow, self).__init__()
        self.name = name
        self.prior = prior
        self.callCounter = 0

    def __call__(self,*args,**kargs):
        return self.sample(*args,**kargs)

    def sample(self,batchSize, K=None,prior = None):
        if prior is None:
            prior = self.prior
        assert prior is not None
        z = prior.sample(batchSize,K)
        logp = prior.logProbability(z,K)
        x,logp_ = self.inverse(z)
        return x,logp-logp_

    def logProbability(self,x,K=None):
        self.callCounter += 1
        z,logp = self.forward(x)
        if self.prior is not None:
            return self.prior.logProbability(z,K)+logp
        return logp

    def energy(self,x,K=None):
        return -self.logProbability(x,K)

    def forward(self,x):
        raise NotImplementedError(str(type(self)))

    def inverse(self,z):
        raise NotImplementedError(str(type(self)))


    def transMatrix(self,sign):
        raise NotImplementedError(str(type(self)))

    def save(self):
        return self.state_dict()

    def load(self,saveDict):
        self.load_state_dict(saveDict)
        return saveDict