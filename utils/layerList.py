import torch

def inverseThoughList(layers,x,sign=1):
    inverseLogjac = x.new_zeros(x.shape[0])
    if sign == -1:
        layers = reversed(layers)
        inverse = lambda layer, x: layer.forward(x)
    elif sign == 1:
        inverse = lambda layer, x: layer.inverse(x)
    for layer in layers:
        x,inverseLogjacTMP = inverse(layer,x)
        inverseLogjac = inverseLogjac + inverseLogjacTMP
    return x,inverseLogjac