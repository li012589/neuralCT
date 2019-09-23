import torch

def jacobian(y,x):
    assert y.shape[0] == x.shape[0]
    batchsize = x.shape[0]
    dim = y.shape[1]
    res = torch.zeros(x.shape[0],y.shape[1],x.shape[1]).to(x)
    for i in range(dim):
        res[:,i,:] = torch.autograd.grad(y[:,i],x,grad_outputs=torch.ones(batchsize).to(x),create_graph=True)[0].reshape(res[:,i,:].shape)
    return res

def jacobianDiag(y,x):
    y = y.reshape(y.shape[0],-1)
    assert y.shape[0] == x.shape[0]
    batchsize = x.shape[0]
    dim = y.shape[1]
    res = torch.zeros(x.shape).to(x)
    for i in range(dim):
        res[:,i] = torch.autograd.grad(y[:,i],x,grad_outputs=torch.ones(batchsize).to(x),create_graph=True)[0][:,i]
    return res

def netJacobian(fn,x):
    return jacobian(fn(x),x)

def hessian(y,x):
    return jacobian(jacobian(y,x).reshape(y.shape[0],-1),x)

def netHessian(fn,x):
    return hessian(fn(x),x)

def laplacian(y,x):
    return jacobianDiag(jacobian(y,x),x).sum(1)

def netLaplacian(fn,x):
    return laplacian(fn(x),x)

def laplacianHutchinson(y,x):
    assert y.shape[0] == x.shape[0]
    batchsize = x.shape[0]
    z = torch.randn(batchsize, x.shape[1]).to(x.device)
    grd = (scalarFnGrad(y,x)*z).sum(1)
    grd2 = scalarFnGrad(grd,x)
    return (grd2*z).sum(1)