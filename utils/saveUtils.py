import subprocess

def createWorkSpace(path):
    savingPath = path+"savings/"
    recordPath = path+"records/"
    picPatch = path+"pic/"
    cmd = ['mkdir', '-p', savingPath]
    subprocess.check_call(cmd)
    cmd = ['mkdir', '-p', recordPath]
    subprocess.check_call(cmd)
    cmd = ['mkdir', '-p', picPatch]
    subprocess.check_call(cmd)

def cleanSaving(path,epoch,keptEpoch,name):
    if epoch >= keptEpoch:
        cmd =["rm","-rf",path+"savings/"+name+"Saving_epoch"+str(epoch-keptEpoch)+".saving"]
        subprocess.check_call(cmd)
        cmd =["rm","-rf",path+"records/"+name+"Record_epoch"+str(epoch-keptEpoch)+".hdf5"]
        subprocess.check_call(cmd)