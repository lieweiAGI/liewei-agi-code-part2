import torch
from torch import nn

a = torch.randn(1,6,3,3)
conv1 = nn.Conv2d(6,6,1,1,groups=2)

y = conv1(a)
y = y.reshape(1,2,3,3,3)
print(y)
y = y.permute(0,2,1,3,4)#通道混洗
y = y.reshape(1,6,3,3)
print("=======")
print(y)