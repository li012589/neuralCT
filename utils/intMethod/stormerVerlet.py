import torch
from .iteration import iteration,iterationMulti

def stormerVerlet(q,p,H,t,steps,rtol=1e-05,atol=1e-08,maxSteps=10,detach=True):
    if detach:
        Q = [q.detach().unsqueeze(0)]
        P = [p.detach().unsqueeze(0)]
    else:
        Q = [q.unsqueeze(0)]
        P = [p.unsqueeze(0)]
    Hq,Hp = _force(H)
    p = p.requires_grad_()
    q = q.requires_grad_()
    """
    if q.shape[-1] == 1:
        _iteration = iteration
    else:
        _iteration = iterationMulti
    """
    _iteration = iteration
    for ss in range(steps):
        p_ = p
        p = _iteration(p,lambda p:p_-t*0.5*Hq(q,p),rtol=rtol,atol=atol,maxSteps=maxSteps,detach=detach)

        q_ = q
        q = _iteration(q,lambda q:q_+t*0.5*(Hp(q_,p)+Hp(q,p)),rtol=rtol,atol=atol,maxSteps=maxSteps,detach=detach)

        p = p-t*0.5*Hq(q,p)
        if detach:
            Q.append(q.detach().unsqueeze(0))
            P.append(p.detach().unsqueeze(0))
            p = p.detach().requires_grad_()
            q = q.detach().requires_grad_()
        else:
            Q.append(q.unsqueeze(0))
            P.append(p.unsqueeze(0))
    Q = torch.cat(Q,dim=0)
    P = torch.cat(P,dim=0)
    return Q,P

def _force(H):
    Hq = lambda q,p: torch.autograd.grad(H(q,p).sum(),q)[0]
    Hp = lambda q,p: torch.autograd.grad(H(q,p).sum(),p)[0]
    return Hq,Hp