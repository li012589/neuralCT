import torch
import numpy as np

def measure(vector):
    return (vector**2).sum()

def expm(q,rtol=1e-3,maxStep=15):
    accumulator = torch.eye(q.shape[-1]).to(q)
    tmpq = q
    i = 1
    error = rtol*measure(q)
    while measure(tmpq) >= error:
        accumulator = tmpq +accumulator
        i+=1
        tmpq = torch.matmul(tmpq,q)/i
        if i>maxStep:
            break
    return accumulator

def expmv(q,v,rtol=1e-3,maxStep=15):
    accumulator = v
    tmpq = torch.matmul(q,v)
    i = 1
    error = rtol*measure(tmpq)
    while measure(tmpq) >= error:
        accumulator = tmpq + accumulator
        i+=1
        tmpq = torch.matmul(q,tmpq)/i
        if i > maxStep:
            break
    return accumulator