import torch
import torch.nn as nn
import torch.nn.functional as F

def lncosh(x):
    return -x + F.softplus(2.*x) - math.log(2.)

def tanhPrime(x):
    return 1.-torch.tanh(x)**2

def tanhPrime2(x):
    t = torch.tanh(x)
    return 2.*t*(t*t-1.)

def sigmoidPrime(x):
    s = torch.sigmoid(x)
    return s*(1.-s)

class Lncosh(nn.Module):
    def __init__(self):
        super(Lncosh,self).__init__()

    def forward(self,x):
        return lncosh(x)

class Tanhprime(nn.Module):
    def __init__(self):
        super(Tanhprime,self).__init__()

    def forward(self,x):
        return tanhPrime(x)

class Tanhprime2(nn.Module):
    def __init__(self):
        super(Tanhprime2,self).__init__()

    def forward(self,x):
        return tanhPrime2(x)

class Sigmoidprime(nn.Module):
    def __init__(self):
        super(Sigmoidprime,self).__init__()

    def forward(self,x):
        return sigmoidPrime(x)