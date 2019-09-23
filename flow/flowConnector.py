from .flow import Flow
import torch

from utils import inverseThoughList

class FlowNet(Flow):
    def __init__(self,layerList,prior=None,name="FlowNet"):
        super(FlowNet,self).__init__(prior,name)
        self.layerList = torch.nn.ModuleList(layerList)

    def inverse(self,y):
        inverseLogjac = y.new_zeros(y.shape[0])
        y,inverseLogjacTMP = inverseThoughList(self.layerList,y,1)
        inverseLogjac = inverseLogjac + inverseLogjacTMP
        return y,inverseLogjac
    def forward(self,z):
        forwardLogjac = z.new_zeros(z.shape[0])
        z,forwardLogjacTMP = inverseThoughList(self.layerList,z,-1)
        forwardLogjac = forwardLogjac + forwardLogjacTMP
        return z,forwardLogjac
    def transMatrix(self,sign=1):
        lst = []
        if sign == 1:
            for layer in reversed(self.layerList):
                lst.append(layer.transMatrix())
            return lst
        elif sign == -1:
            for layer in self.layerList:
                lst.append(layer.transMatrix())
            return lst