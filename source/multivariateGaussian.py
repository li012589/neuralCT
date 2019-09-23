import torch
import numpy as np

from .source import Source

class MultivariateGaussian(Source):
    def __init__(self, nvars, mu=None,sigma=None  ,K=1.0,name="gaussian"):
        super(MultivariateGaussian,self).__init__(nvars,K,name)
        if mu is None:
            mu = [0]*self.nvars[-1]
        if sigma is None:
            sigma = torch.diag(torch.tensor([1]*self.nvars[-1])).to(torch.float32)
        self.mu = torch.nn.Parameter(torch.tensor(mu).to(torch.float32),requires_grad=False)
        self.sigma = torch.nn.Parameter(sigma,requires_grad=False)

    def sample(self, batchSize):
        L = torch.cholesky(self.sigma)
        size = [batchSize] + self.nvars
        return (self.mu+torch.matmul(torch.randn(size).to(self.sigma),self.sigma)).to(self.sigma)

    def _energy(self, z):
        return -(-0.5*torch.matmul(torch.matmul((z-self.mu),torch.inverse(self.sigma)).reshape(z.shape[0],1,-1),(z-self.mu).reshape(z.shape[0],-1,1))-0.5*self.nvars[-1]*torch.log(torch.tensor(2.*np.pi).to(z))-0.5*torch.log(torch.det(self.sigma))).reshape(-1)
