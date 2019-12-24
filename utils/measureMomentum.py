import torch

def measureM(data,idx=None,rank=40):
    nn = data.shape[1]//2
    if idx is not None:
        data = data[:,idx[:rank]]
    return (data[:,nn:]**2).sum(1)