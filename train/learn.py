import torch
import numpy as np
import utils
import h5py
import matplotlib.pyplot as plt

def learn(source,flow,batchSize,epochs,lr = 1e-3, save =True, saveSteps = 10, savePath = None, weight_decay =0.001, adaptivelr = False,ifPlot=True):
    if savePath is None:
        savePath = "./opt/tmp/"
    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    if ifPlot:
        print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if adaptivelr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    LOSS = []

    for epoch in range(epochs):
        loss,lossstd=reversedKLD(flow,source,batchSize)
        flow.zero_grad()
        loss.backward()
        optimizer.step()

        if ifPlot:
            print("epoch:",epoch, "L:",loss.item(),"+/-",lossstd.item())

        LOSS.append(loss.item())
        if adaptivelr:
            scheduler.step()
        if save and epoch%saveSteps == 0:
            d = flow.save()
            torch.save(d,savePath+"savings/"+flow.name+"Saving_epoch"+str(epoch)+".saving")
            with h5py.File(savePath+"records/"+flow.name+"Record_epoch"+str(epoch)+".hdf5", "w") as f:
                f.create_dataset("LOSS",data=np.array(LOSS))
            utils.cleanSaving(savePath,epoch,6*saveSteps,flow.name)

    return LOSS

def reversedKLD(model,target,batchSize):
    x ,sampleLogProbability = model.sample(batchSize)
    lossorigin = (sampleLogProbability - target.logProbability(x))
    loss = lossorigin.mean()
    lossstd = lossorigin.std()
    return loss,lossstd