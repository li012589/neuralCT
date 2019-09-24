import numpy as np

import torch

def logit(x, alpha=1E-6):
    y = alpha + (1.-2*alpha)*x
    return np.log(y) - np.log(1. - y)

def logit_back(x, alpha=1E-6):
    y = torch.sigmoid(x)
    return (y- alpha)/(1.-2*alpha)
