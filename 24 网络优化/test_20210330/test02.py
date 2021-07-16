import torch
from torch import nn
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,16,3,1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.3),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, 1),
            nn.Dropout2d(0.1),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(784,512),
            nn.BatchNorm1d(512)
        )
    def forward(self,x):
        out = self.layers(x)
        return out + x

net = models.resnet50()
net.train()
net.eval()
print(net)