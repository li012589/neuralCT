import numpy as np
import torch

from .source import Source

def adjMatrixHamiltonian(Adj,t,eta):
    shape = eta.shape[-1]
    return torch.matmul(torch.matmul(eta,Adj).reshape(-1,1,shape),eta.reshape(-1,shape,1)).reshape(-1)

class HarmonicChain(Source):
    def __init__(self,n,kappa,K=1.0,name="HarmonicChain"):
        nvars = [2*n]
        super(HarmonicChain,self).__init__(nvars,K,name)
        self.kappa = kappa
        if n >=2:
            adjmeta = torch.diag(torch.tensor([2]*n,dtype=torch.float32))+torch.diag(torch.tensor([-1]*(n-1),dtype=torch.float32),diagonal=1)+torch.diag(torch.tensor([-1]*(n-1),dtype=torch.float32),diagonal=-1)
            adj = torch.cat((torch.cat((self.kappa/2*adjmeta,torch.zeros(n,n)),1),torch.cat((torch.zeros(n,n),0.5*torch.eye(n)),1)),0)
        else:
            adj = torch.tensor([[0.5,0],[0,kappa/2]]).to(torch.float32)
        self.adj=torch.nn.Parameter(adj,requires_grad=False)
    def _energy(self,x):
        return adjMatrixHamiltonian(self.adj,0,x)