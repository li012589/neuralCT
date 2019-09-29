import torch
import source
import numpy as np
from .img_trans import logit

def loadmd(filename,dataname=["arr_0","arr_1","arr_2"],scale=10.0,fix=np.array([0,2.3222,0])):
    data = []
    with np.load(filename) as f:
        for name in dataname:
            data.append(f[name])
    data = np.concatenate(data,0)
    batchSize = data.shape[0]
    data = data.reshape(-1,3) - fix
    return torch.from_numpy(data.reshape(batchSize,-1)).double()*scale

def load(filename):
    data = torch.from_numpy(np.load(filename)["arr_0"]).float()
    data = logit((data + torch.rand(data.shape))/256.0)
    batchSize = data.shape[0]
    return data.reshape(batchSize,-1)

class DataSampler(object):
    def __init__(self,data,ratio,shuffle=True):
        if shuffle:
            ind = torch.randperm(data.shape[0])
            data = data[ind]
        self.data = data
        batchSize = data.shape[0]
        self.Tsize = int(batchSize*ratio)
        self.Vsize = int(batchSize-self.Tsize)
    def sample(self,size):
        if size >self.Tsize:
            raise Exception("Size exceeding dataset")
        index = torch.randint(0,self.Tsize,[size])
        return self.data[index]
    def sampleVaildation(self,size):
        if size >self.Vsize:
            raise Exception("Size exceeding dataset")
        index = torch.randint(0,self.Vsize,[size])
        return self.data[-index]


class MDSampler(DataSampler):
    def __init__(self,data,ratio=0.9,pMean=None,pVariance=None):
        super(MDSampler,self).__init__(data,ratio)
        if pMean is None:
            pMean = torch.tensor([0.0]*self.data.shape[-1]).to(self.data)
        if pVariance is None:
            pVariance = torch.tensor([1.0]*self.data.shape[-1]).to(self.data)
        self.pMean = pMean.to(self.data)
        self.pVariance = pVariance.to(self.data)

    def sample(self,size,momentum=True):
        if momentum:
            p = torch.randn([size,self.data.shape[-1]]).to(self.data)*(self.pVariance)+self.pMean
            return torch.cat((super(MDSampler,self).sample(size),p),dim=-1)
        else:
            return super(MDSampler,self).sample(size)

    def sampleVaildation(self,size,momentum=True):
        if momentum:
            p = torch.randn([size,self.data.shape[-1]]).to(self.data)*(self.pVariance)+self.pMean
            return torch.cat((super(MDSampler,self).sampleVaildation(size),p),dim=-1)
        else:
            return super(MDSampler,self).sampleVaildation(size)


