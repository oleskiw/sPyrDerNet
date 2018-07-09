import torch
import torch.nn as nn


def quad_activ(x):
    return torch.sqrt(x**2 + 0.001)


