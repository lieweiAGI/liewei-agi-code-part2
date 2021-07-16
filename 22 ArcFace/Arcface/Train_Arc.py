import torch
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
from ArcLoss import *
from Module import *

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
train_data = torchvision.datasets.MNIST(
    root="D:\data\MNIST_data",
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.MNIST(
    root="D:\data\MNIST_data",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

train = data.DataLoader(dataset=train_data,batch_size=2048,shuffle=True)

'''Train'''
if __name__ == '__main__':

    net = Net().to(device)
    arc = Arc().to(device)
    # net = torch.load("parms/net_10.pt")
    # arc = torch.load("parms/arc_10.pt")
    optimizer_net = torch.optim.Adam(net.parameters())
    optimizer_arc = torch.optim.Adam(arc.parameters())
    loss_nll = nn.NLLLoss()

    plt.ion()
    for epoch in range(1000):
        print('↓'*10,'Epoch=',epoch,'↓'*10)
        for i, (data, target) in enumerate(train):
            data, target = data.to(device), target.to(device)
            layer = net(data)
            out = arc(layer)

            plt.clf()
            for j in range(10):
                plt.plot(layer[target == j, 0].cpu().detach().numpy(), layer[target == j, 1].cpu().detach().numpy(),'.')
            plt.pause(0.01)

            plt.show()

            loss_arc=loss_nll(out, target)

            optimizer_net.zero_grad()
            optimizer_arc.zero_grad()
            loss_arc.backward()
            optimizer_net.step()
            optimizer_arc.step()
        print(loss_arc.item())
        plt.savefig('.\images_arc\{0}.jpg'.format(epoch))
        # torch.save(net,r'.\parms\arc_{}.pt'.format(epoch))
        # torch.save(arc, r'.\parms\net_{}.pt'.format(epoch))
        print('save susccesfully')
    plt.ioff()

