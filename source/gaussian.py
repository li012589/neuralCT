import torch
import numpy as np

from .source import Source

class Gaussian(Source):
    def __init__(self, nvars, sigma = 1, K=1.0, name="gaussian"):
        super(Gaussian,self).__init__(nvars,K,name)
        self.sigma = torch.nn.Parameter(torch.tensor([sigma]).to(torch.float32),requires_grad=False)

    def sample(self, batchSize,K=None):
        if K is None:
            K = 1.0
        size = [batchSize] + self.nvars
        return (torch.randn(size,dtype=self.sigma.dtype).to(self.sigma)*self.sigma/K)

    def _energy(self, z):
        return -(-0.5 * (z/self.sigma)**2-0.5*torch.log(2.*np.pi*self.sigma**2)).reshape(z.shape[0],-1).sum(dim=1)
