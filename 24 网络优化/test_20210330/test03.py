import torch
from torch import nn

if __name__ == '__main__':
    conv = nn.Conv2d(3,3,3,1)
    print(conv.weight.shape)
    print(conv.weight)

    # nn.init.kaiming_normal_(conv.weight)
    # print(conv.weight)

    # nn.init.normal_(conv.weight,0,0.1)
    # print(conv.weight)

    nn.init.zeros_(conv.bias)
    print(conv.bias)