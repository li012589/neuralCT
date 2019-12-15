import torch

def iteration(outputs,fn,rtol=1e-05,atol=1e-08,maxSteps=10,detach=True):
    last = outputs
    if detach:
        last = last.detach().requires_grad_()
    for _ in range(maxSteps):
        outputs = fn(last)
        if (outputs-last).mean()<=atol+rtol*(last.mean()):
            return outputs
        if detach:
            outputs = outputs.detach()
            last = outputs.detach().requires_grad_()
        else:
            last = outputs
    return outputs.requires_grad_()

def iterationMulti(outputs,fn,rtol=None,atol=None,maxSteps=10,detach=True):
    last = outputs
    if detach:
        last = [term.detach().requires_grad_() for term in last]
    if rtol is None:
        rtol = [1e-05]*len(outputs)
    if atol is None:
        atol = [1e-08]*len(outputs)
    for _ in range(maxSteps):
        outputs = fn(*last)
        if _output_test(outputs,last):
            return outputs
        if detach:
            outputs_ = []
            for term in outputs:
                outputs_.append(term.detach())
            outputs = outputs_
            last = [item.detach().requies_grad_() for item in outputs]
        else:
            last = outputs
    return [term.requires_grad_() for term in outputs]


def _output_test(outputs,lastOutputs,rtol,atol):
    for idx, term in enumerate(outputs):
        if (term-lastOutputs[idx]) > atol+rtol*lastOutputs[idx]:
            return False
    return True