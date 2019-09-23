import math 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

def lncosh(x):
    return -x + F.softplus(2.*x) - math.log(2.)

def tanh_prime(x):
    return 1.-torch.tanh(x)**2

def tanh_prime2(x):
    t = torch.tanh(x)
    return 2.*t*(t*t-1.)

def sigmoid_prime(x):
    s = torch.sigmoid(x)
    return s*(1.-s)

class CNN(nn.Module):
    def __init__(self, L, channel, hidden_size, device='cpu', name=None):
        super(CNN, self).__init__()
        self.device = device
        if name is None:
            self.name = 'CNN'
        else:
            self.name = name
 
        self.L = L 
        self.dim = L**2
        self.channel = channel
        self.conv1 = nn.Conv2d(self.channel, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, 2*hidden_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2*hidden_size*(self.L//4)**2, 64)
        self.fc2 = nn.Linear(64, 1, bias=False)
    
    def forward(self, x):
        x = x.view(x.shape[0], self.channel, self.L, self.L)
        x = F.softplus(F.max_pool2d(self.conv1(x), 2))
        x = F.softplus(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.softplus(self.fc1(x))
        return self.fc2(x).sum(dim=1)

    def grad(self, x):
        batch_size = x.shape[0]
        return torch.autograd.grad(self.forward(x), x, grad_outputs=torch.ones(batch_size, device=x.device), create_graph=True)[0]

class MLP(nn.Module):
    def __init__(self, dim, hidden_size, use_z2=True, device='cpu', name=None):
        super(MLP, self).__init__()
        self.device = device
        if name is None:
            self.name = 'MLP'
        else:
            self.name = name

        self.dim = dim
        self.fc1 = nn.Linear(dim, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1, bias=False)
        if use_z2:
            self.activation = lncosh
        else:
            self.activation = F.softplus

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = F.softplus(self.fc2(out))
        out = self.fc3(out)
        return out.sum(dim=1)

    def grad(self, x):
        batch_size = x.shape[0]
        return torch.autograd.grad(self.forward(x), x, grad_outputs=torch.ones(batch_size, device=x.device), create_graph=True)[0]

 
class Simple_MLP(nn.Module):
    '''
    Single hidden layer MLP 
    with handcoded grad and laplacian function
    '''
    def __init__(self, dim, hidden_size, use_z2=True, device='cpu', name=None):
        super(Simple_MLP, self).__init__()
        self.device = device
        if name is None:
            self.name = 'Simple_MLP'
        else:
            self.name = name

        self.dim = dim
        self.fc1 = nn.Linear(dim, hidden_size, bias=not use_z2)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)

        if use_z2:
            self.activation = lncosh
            self.activation_prime = torch.tanh
            self.activation_prime2 = tanh_prime 
        else:
            self.activation = F.softplus
            self.activation_prime = torch.sigmoid
            self.activation_prime2 = sigmoid_prime

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return out.sum(dim=1)

    def grad(self, x):
        '''
        grad u(x)
        '''
        out = self.activation_prime(self.fc1(x)) 
        out = torch.mm(out, torch.diag(self.fc2.weight[0]))  
        out = torch.mm(out, self.fc1.weight)
        return out

    def laplacian(self, x):
        '''
        div \cdot grad u(x)
        '''
        out = self.activation_prime2(self.fc1(x)) 
        out = torch.mm(out, torch.diag(self.fc2.weight[0]))  
        out = torch.mm(out, self.fc1.weight**2)
        return out.sum(dim=1)

    def acceleration(self, x):
        '''
        d^x/dt^2 = grad [(grad phi)^2] = 2 (v cdot grad) v = 2 H\cdot v 
        '''
        grad = self.grad(x)
        return torch.autograd.grad((grad**2).sum(dim=1), x, grad_outputs=torch.ones(x.shape[0], device=x.device), create_graph=True)[0]

 
if __name__=='__main__':
    from hessian import compute_grad_and_hessian
    batchsize = 1
    L = 4
    dim = L**2
    x = torch.randn(batchsize, dim, requires_grad = True)
    net = Simple_MLP(dim=dim, hidden_size = 10)

    print (net.acceleration(x))
    grad, hessian = compute_grad_and_hessian(net(x), x)
    print (grad.shape)
    print (hessian.shape)
    print (2.*torch.bmm(grad.unsqueeze(1), hessian).squeeze(0))
