import torch
from torch.autograd import grad
import numpy as np
from scipy import integrate
import time

from utils import stormerVerlet

# INITAL CONDITIONS
q = torch.randn(1,1).double()
p = torch.randn(1,1).double()

h = 0.001
step = 10000

eta = torch.cat((q,p),dim=-1).double()

def harmonic(alpha,t,eta):
    num = eta.shape[1]//2
    return 0.5*(eta[:,:num]**2).sum(-1)+alpha*0.5*(eta[:,num:]**2).sum(-1)
Horigin = harmonic(1,0,eta)


# FOR SCIPY SANITY CHECK
start = time.time()
def diff(f,t,x):
    res= f(t,x)
    return grad(torch.matmul(res,torch.ones(res.shape).to(res)),x)[0]

def diffH(f,t,x):
    num = x.shape[1]//2
    part0 = torch.zeros(num,num)
    part1 = torch.diag_embed(torch.ones(num))
    J = torch.cat((torch.cat((part0,part1),1),torch.cat((-part1,part0),1))).to(x)
    res = diff(f,t,x)
    return torch.matmul(J,res.reshape(res.shape[0],res.shape[1],1)).reshape(x.shape)

H = lambda t,eta: harmonic(1,1,eta)

F = lambda t, eta:diffH(H,t,torch.tensor(eta.reshape(1,-1),requires_grad=True)).numpy().reshape(-1)

sciInt = integrate.solve_ivp(F,[0,h*step],y0 = eta.reshape(-1).numpy(),method='RK45',rtol=1e-6)

Ts = sciInt.t
Y = np.transpose(sciInt.y,(1,0))
P = Y[:,Y.shape[-1]//2]
Q = Y[:,0]
end = time.time()
timeScipy = end-start


# STORMERVERLECT
start = time.time()
def HH(q,p):
    ETA = torch.cat((q,p),dim=-1)
    return harmonic(1,1,ETA)

mQ,mP = stormerVerlet(q,p,HH,h,step)
mQ = mQ.reshape(-1)
mP = mP.reshape(-1)
end = time.time()
timeStormerVerlect = end-start


# TESTS
print("Scipy time:",timeScipy,"Pytorch time:",timeStormerVerlect)
etascipy = torch.tensor([Q[-1],P[-1]],dtype=torch.float64).reshape(1,-1)
Hscipy = harmonic(1,0,etascipy)
errorScipy = Horigin-Hscipy
Hstormerverlect = HH(mQ[-1].reshape(1,1),mP[-1].reshape(1,1))
errorstormerverlect = Horigin-Hstormerverlect
print("Error of scipy:",errorScipy.item(),"Error of stormerVerlet:",errorstormerverlect.item())


# MATPLOTLIB PLOT
from matplotlib import pyplot as plt

fig1 = plt.figure()
ax1 = fig1.add_subplot(211)
ax1.plot(Ts,Q,label="scipy.RK45",linewidth=2)
ax1.plot(np.arange(0,h*step,h),mQ.numpy(),label="stormerVerlet",linewidth=2)
plt.legend()

ax2 = fig1.add_subplot(212)
ax2.plot(Ts,P,label="scipy.RK45",linewidth=2)
ax2.plot(np.arange(0,h*step,h),mP.numpy(),label="stormerVerlet",linewidth=2)
plt.legend()

fig1 = plt.figure()
plt.plot(Q,P,'o',label="scipy.RK45",markersize=4)
plt.plot(mQ.numpy(),mP.numpy(),'+',label = "stormerVerlet",markersize=4)

plt.legend()
plt.show()