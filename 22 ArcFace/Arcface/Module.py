import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,3,stride=2,padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(32),

        )
        self.liner=nn.Sequential(
            nn.Linear(32*14*14, 512),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256 , 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2,bias=False)
        )

    def forward(self, x):
        conv=self.conv(x).view(-1,32*14*14)
        liner=self.liner(conv)
        return liner
if __name__ == '__main__':
    Data=torch.Tensor(3,1,28,28)
    net=Net()
    print(net(Data))
#