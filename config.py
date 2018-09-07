import torch

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float32
