import numpy as np
import torch

from .source import Source

def harmonic(alpha,t,eta):
    num = eta.shape[1]//2
    return 0.5*(eta[:,:num]**2).sum(-1)+alpha*0.5*(eta[:,num:]**2).sum(-1)

class HarmonicChain(Source):
    def __init__(self,nvars,alpha,K=1.0,name="HarmonicChain"):
        super(HarmonicChain,self).__init__([nvars],K,name)
        self.alpha = alpha

    def _energy(self,x):
        return harmonic(self.alpha,0,x)