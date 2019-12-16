import numpy as np

import torch

# Adopted from https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def dihedralAngle(p0,p1,p2,p3):
    b0 = -1.0*(p1-p0)
    b1 = p2-p1
    b2 = p3-p2

    b0xb1 = torch.cross(b0,b1,dim=1)
    b1xb2 = torch.cross(b2,b1,dim=1)

    b0xb1_x_b1xb2 = torch.cross(b0xb1,b1xb2,dim=1)

    y = torch.bmm(b0xb1_x_b1xb2.reshape(-1,1,3),b1.reshape(-1,3,1)).reshape(-1,1)*(1.0/torch.norm(b1,p=2,dim=1)).reshape(-1,1)
    x = torch.bmm(b0xb1.reshape(-1,1,3),b1xb2.reshape(-1,3,1)).reshape(-1,1)

    return torch.atan2(y,x)*180/np.pi


def alanineDipeptidePhiPsi(data):
    p0 = data[:,1]
    p1 = data[:,3]
    p2 = data[:,4]
    p3 = data[:,6]

    Phi = dihedralAngle(p0,p1,p2,p3)

    p0 = data[:,3]
    p1 = data[:,4]
    p2 = data[:,6]
    p3 = data[:,8]

    Psi =  dihedralAngle(p0,p1,p2,p3)

    return Phi,Psi
