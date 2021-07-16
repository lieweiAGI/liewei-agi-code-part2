from test01 import Net
from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == '__main__':
    net = Net()
    net.load_state_dict(torch.load("models/net.pth"))

    print(net)

    layer1_weight = net.fc1[0].weight
    layer2_weight = net.fc1[3].weight
    layer3_weight = net.fc1[6].weight
    layer4_weight = net.fc1[9].weight

    summaryWriter = SummaryWriter("logs")

    summaryWriter.add_histogram("layer1",layer1_weight)
    summaryWriter.add_histogram("layer2",layer2_weight)
    summaryWriter.add_histogram("layer3",layer3_weight)
    summaryWriter.add_histogram("layer4",layer4_weight)
