import torch
import torch.nn as nn


def smoothabs_activ(x):
    return torch.sqrt(x**2 + 0.001)

def quadratic_activ(x):
    return torch.mul(x,x)


