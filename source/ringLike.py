import numpy as np
import torch

from .source import Source

class Ring2d(Source):
    def __init__(self):
        super(Ring2d,self).__init__([4],name = 'Ring2D')

    def sample(self,batchSize,thermalSteps = 100, tranCore=None):
        return self._sampleWithMetropolis(batchSize,thermalSteps,tranCore)
    '''
    def sample(self, batchSize, thermalSteps = 50, interSteps=5, epsilon=0.1):
        return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)
    '''

    def _energy(self,x):
        q, p = x.split(2, dim=1)
        return  (torch.sqrt((q**2).sum(dim=1))-2.0)**2/0.32+0.5*(p**2).sum(dim=1)

class Ring2dNoMomentum(Source):
    def __init__(self):
        super(Ring2dNoMomentum,self).__init__([2],name = 'Ring2D')

    def sample(self,batchSize,thermalSteps = 100, tranCore=None):
        return self._sampleWithMetropolis(batchSize,thermalSteps,tranCore)
    '''
    def sample(self, batchSize, thermalSteps = 50, interSteps=5, epsilon=0.1):
        return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)
    '''

    def _energy(self,q):
        return  (torch.sqrt((q**2).sum(dim=1))-2.0)**2/0.32