import torch
from torch import nn

from .scalarField import ScalarField
from .mlp import SimpleMLP

class SimpleScalarMLP(ScalarField):
    def __init__(self,dimList,activation=None,name="ScalarField"):
        super(SimpleScalarMLP,self).__init__()
        self.mlp = SimpleMLP(dimList+[1],activation,name)
    def forward(self,x):
        return self.mlp(x)