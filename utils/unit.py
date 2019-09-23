import torch

ATOM_MESS = {"C":12.01,"O":15.99,"N":14.007,"AL":26.981}

mp = 1.67262192369E-27
Na = 6.02214076E23
kB = 1.380649E-23
J2cal = 5000/20929

def variance(weightNum,K=300):
    return torch.sqrt(weightNum*mp*Na*kB*Na*J2cal*K)

def smile2mass(smile):
    mess = []
    for u in smile:
        if u.isalpha():
            mess.append(ATOM_MESS[u])
    return mess