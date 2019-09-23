import torch
import numpy as np
import h5py

import matplotlib.pyplot as plt

import utils

def forwardLearn(dataloader,flow,batchSize,epochs,lr = 1e-3, save =True, saveSteps = 1, savePath = None, weight_decay=0.001, adaptivelr = True,ifPlot=True):
    if savePath is None:
        savePath = "./opt/tmp/"
    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    if ifPlot:
        print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if adaptivelr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    LOSS = []
    LOSSVAL = []

    nSteps = int(dataloader.Tsize/batchSize)+1
    nStepsVal = int(dataloader.Vsize/batchSize)+1

    for epoch in range(epochs):
        for step in range(nSteps):
            loss,lossstd = forwardKLD(flow,dataloader,batchSize)

            flow.zero_grad()
            loss.backward()
            optimizer.step()

            if ifPlot:
                print("epoch:",epoch,"step:",str(step),"/",str(nSteps) ,"L:",loss.item(),"+/-",lossstd.item())

        if save and epoch%saveSteps == 0:
            totalloss = 0
            totallossVal = 0
            for _ in range(nStepsVal):
                loss,_ = forwardKLD(flow,dataloader,batchSize)
                lossVal,_ = forwardKLD(flow,dataloader,batchSize,validation=True)
                totalloss = totalloss + loss.mean().item()
                totallossVal = totallossVal + lossVal.mean().item()
            totalloss = totalloss/nStepsVal
            totallossVal = totallossVal/nStepsVal
            LOSS.append(totalloss)
            LOSSVAL.append(totallossVal)
            d = flow.save()
            torch.save(d,savePath+"savings/"+flow.name+"Saving_epoch"+str(epoch)+".saving")
            with h5py.File(savePath+"records/"+flow.name+"Record_epoch"+str(epoch)+".hdf5", "w") as f:
                f.create_dataset("LOSS",data=np.array(LOSS))
                f.create_dataset("LOSSVAL",data=np.array(LOSSVAL))
            utils.cleanSaving(savePath,epoch,6*saveSteps,flow.name)

        if adaptivelr:
            scheduler.step(lossVal)
            #scheduler.step()

    return LOSS,LOSSVAL

def forwardKLD(model,dataloader,batchSize,validation=False):
    if not validation:
        x = dataloader.sample(batchSize)
        lossorigin = -model.logProbability(x)
        loss = lossorigin.mean()
        lossstd = lossorigin.std()
        return loss,lossstd
    else:
        xVal = dataloader.sampleVaildation(batchSize)
        lossoriginVal = -model.logProbability(xVal)
        lossVal = lossoriginVal.mean()
        lossstdVal = lossoriginVal.std()
        return lossVal,lossstdVal
