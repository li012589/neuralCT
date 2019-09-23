import torch
import source
import numpy as np

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
    data = (data + torch.rand(data.shape))/256.0
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


class TwoGaussianSampler(object):
    def __init__(self,mean1,mean2,cov1,cov2,size, ratio=0.9,momentum=True):
        dims = len(mean1)
        self.dims = dims
        self.mean1 = mean1
        self.mean2 = mean2
        self.A1 = torch.cholesky(cov1)
        self.A2 = torch.cholesky(cov2)
        self.Tsize = int(size*ratio)
        self.Vsize = int(size-self.Tsize)
        self.momentum = momentum

    def sample(self, batchSize):
        p = torch.randn([batchSize,self.dims]).to(self.A1)
        meta1 = (torch.matmul(torch.randn([batchSize,self.dims]).to(self.A1),self.A1)+self.mean1).permute([1,0])
        meta2 = (torch.matmul(torch.randn([batchSize,self.dims]).to(self.A1),self.A2)+self.mean2).permute([1,0])
        mask = (torch.randn(batchSize) > 0).to(self.A1).reshape(1,-1)
        if self.momentum:
            return torch.cat(((mask*meta1+(1-mask)*meta2).permute([1,0]),p),dim=-1)
        else:
            return (mask*meta1+(1-mask)*meta2).permute([1,0])

    def sampleVaildation(self,batchSize):
        return self.sample(batchSize)

class Ring2DSampler(object):
    def __init__(self,rCenter,r,rp,phip,size,ratio=0.9,momentum=True):
        self.r = r
        self.rCenter = rCenter
        self.Tsize = int(size*ratio)
        self.Vsize = int(size-self.Tsize)
        self.momentum = momentum
        self.rp = rp
        self.phip = phip

    def sample(self,batchSize):
        r = torch.randn(batchSize,1).to(self.r)*(self.r)**0.5+self.rCenter
        phi = 2*np.pi*torch.rand(batchSize,1).to(self.r)
        x = (r*torch.cos(phi)).detach()
        y = (r*torch.sin(phi)).detach()
        q = torch.cat([x,y],dim=1)
        if self.momentum:
            rp = torch.randn(batchSize,1).to(self.r)*(self.rp)**0.5
            phip = torch.randn(batchSize,1).to(self.r)*(self.phip)**0.5
            xp = (rp*torch.cos(phi)+phip*r*torch.sin(phi)).detach()
            yp = (rp*torch.sin(phi)+phip*r*torch.cos(phi)).detach()
            p = torch.cat([xp,yp],dim=1)
            return torch.cat([q,p],dim = 1)
        else:
            return q
    def sampleVaildation(self,batchSize):
        return self.sample(batchSize)

class HMCRing2DSampler(object):
    def __init__(self,device,size,ratio=0.9,momentum=True):
        self.Tsize = int(size*ratio)
        self.Vsize = int(size-self.Tsize)
        self.momentum = momentum
        self.ring = source.Ring2dNoMomentum().to(device)

    def sample(self,batchSize):
        q = self.ring.sample(batchSize)
        if self.momentum:
            p = torch.randn(batchSize,2).to(q)
            return torch.cat([q,p],dim = 1)
        else:
            return q
    def sampleVaildation(self,batchSize):
        return self.sample(batchSize)


class DataBaseRing2DSampler(DataSampler):
    def __init__(self,device,size,ratio=0.9,momentum=True):
        ring = source.Ring2dNoMomentum().to(device)
        data = ring.sample(size).to(device)
        if momentum:
            p = torch.randn(data.shape).to(device)
            data = torch.cat([data,p],dim = -1)
        super(DataBaseRing2DSampler,self).__init__(data,ratio)

class MMDSampler(object):
    def __init__(self,device=torch.device("cpu"),means=None,widths=None):
        if means is None:
            self.means = [np.array([-1,1]), np.array([1,-1])]
        else:
            self.means = means
        if widths is None:
            self.widths = [np.array([0.3,2]),np.array([0.3,2])]
        else:
            self.widths = widths
        self.device = device
        self.Tsize = 100000
        self.Vsize = 40000

    def sample(self,batchSize):
        T = batchSize
        X = np.zeros((T, 2))
        # hidden trajectory
        D = (np.random.randn(T)>0).astype(int)

        for t in range(T):
            s = D[t]
            X[t,0] = self.widths[s][0] * np.random.randn() + self.means[s][0]
            X[t,1] = self.widths[s][1] * np.random.randn() + self.means[s][1]
        X = torch.from_numpy(X).to(torch.float32).to(self.device)
        PX = torch.randn(X.shape).to(self.device)
        return torch.cat([X,PX],dim=1)
    def sampleVaildation(self,batchSize):
        return self.sample(batchSize)


