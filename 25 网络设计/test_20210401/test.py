import torch
from torch import nn

net = nn.Conv2d(2,4,3,1,1,groups=2)
print(net.weight.shape)
print(net.weight)