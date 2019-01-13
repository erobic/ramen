import torch.nn.functional as F
import torch.nn as nn
import torch


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class NormalizedSwish(nn.Module):
    def __init__(self):
        super(NormalizedSwish, self).__init__()

    def forward(self, x):
        return 1.78718727865 * (x * torch.sigmoid(x) - 0.20662096414)
