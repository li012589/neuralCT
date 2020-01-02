import torch
import numpy as np

# From https://github.com/soumith/dcgan.torch/issues/14

def np_slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


def slerp(val, low, high):
    omega = torch.acos(torch.clamp(torch.dot(low/torch.norm(low), high/torch.norm(high)), -1, 1))
    so = torch.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return torch.sin((1.0-val)*omega) / so*low + torch.sin(val*omega)/so*high