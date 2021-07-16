import torch
from torch import nn


conv_1 = nn.Conv2d(4,20,3,1)
conv_2 = nn.Conv2d(4,20,3,1,groups=2)
conv_3 = nn.Conv2d(4,20,3,1,groups=4)

x = torch.randn(1,4,112,112)

# y1 = conv_1(x)
# print(y1.shape)
# y2 = conv_2(x)
# print(y2.shape)
# y3 = conv_3(x)
# print(y3.shape)

