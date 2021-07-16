from torch import nn

conv = nn.Conv2d(6,6,3,1,groups=6)
print(conv.weight.shape)