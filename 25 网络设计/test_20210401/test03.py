import torch
from torch import nn
import thop

class Net_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(4, 20, 3, 1,bias=False)
    def forward(self,x):
        return self.layer(x)
class Net_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(4, 20, 3, 1,groups=2,bias=False)
    def forward(self,x):
        return self.layer(x)
class Net_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(4, 20, 3, 1,groups=4,bias=False)
    def forward(self,x):
        return self.layer(x)
if __name__ == '__main__':
    x = torch.randn(1, 4, 112, 112)
    net_v1 = Net_v1()
    net_v2 = Net_v2()
    net_v3 = Net_v3()

    print(thop.clever_format(thop.profile(net_v1,(x,))))
    print(thop.clever_format(thop.profile(net_v2, (x,))))
    print(thop.clever_format(thop.profile(net_v3, (x,))))