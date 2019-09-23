import torch
from torch import nn

from .flow import Flow

class NICE(Flow):
    def __init__(self, maskList, tList, prior = None, name = "RNVP"):
        super(NICE,self).__init__(prior,name)

        assert len(tList) == len(maskList)

        self.maskList = nn.Parameter(maskList,requires_grad=False)

        self.tList = torch.nn.ModuleList(tList)

    def inverse(self,y):
        inverseLogjac = y.new_zeros(y.shape[0])
        for i in range(len(self.tList)):
            y_ = y*self.maskList[i]
            t = self.tList[i](y_)*(1-self.maskList[i])
            #y = y_ + self.maskListR[i] * (y + t)
            y = y + t
        return y,inverseLogjac

    def forward(self,z):
        forwardLogjac = z.new_zeros(z.shape[0])
        for i in reversed(range(len(self.tList))):
            z_ = self.maskList[i]*z
            t = self.tList[i](z_)*(1-self.maskList[i])
            #z = self.maskListR[i] * (z - t) + z_
            z = z - t
        return z,forwardLogjac
    def transMatrix(self,sign = 1):
        return None