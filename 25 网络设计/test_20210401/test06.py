from torchvision import models
from torch import nn

net = models.densenet121(pretrained=True)
print(net)
net.classifier = nn.Linear(1024,10)
print(net)