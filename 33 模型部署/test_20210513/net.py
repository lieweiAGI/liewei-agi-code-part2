import torch
from torch import nn

class Netv1(nn.Module):

    #对网络的结构进行构造（设计网络）
    def __init__(self):
        super().__init__()

        self.W = nn.Parameter(torch.randn(784,10))

    #网络前向过程的逻辑
    def forward(self,x):
        h = x @ self.W

        #Softmax
        h = torch.exp(h)
        z = torch.sum(h,dim=1,keepdim=True)
        return h/z

class Netv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100,52)
        self.fc3 = nn.Linear(52,10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        h = self.fc1(x)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.fc3(h)
        out = self.softmax(h)
        return out

class Netv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(784,100),
            nn.ReLU(),
            nn.Linear(100,52),
            nn.ReLU(),
            nn.Linear(52,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        return self.layer(x)

if __name__ == '__main__':
    net = Netv3()
    x = torch.randn(6,784)
    y = net(x)
    print(y.shape)