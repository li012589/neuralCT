import torch
from torch import nn
from ..matrixGrad import netJacobian,netLaplacian,netHessian

class ScalarField(nn.Module):
    def __init__(self):
        super(ScalarField,self).__init__()

    def grad(self,x):
        if not x .requires_grad:
            x = x.requires_grad_()
        return netJacobian(self.forward,x).reshape(x.shape[0],-1)

    def laplacian(self,x):
        if not x .requires_grad:
            x = x.requires_grad_()
        return netLaplacian(self.forward,x)

    def hessian(self,x):
        if not x .requires_grad:
            x = x.requires_grad_()
        return netHessian(self.forward,x)
