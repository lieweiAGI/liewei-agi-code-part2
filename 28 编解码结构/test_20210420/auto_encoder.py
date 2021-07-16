import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import save_image

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.liner_layers = nn.Sequential(
            nn.Linear(784,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )

    def forward(self,x):
        x = x.reshape(-1,784)
        out = self.liner_layers(x)
        return out

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.liner_layers = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 784),
            nn.BatchNorm1d(784),
            nn.LeakyReLU()
        )

    def forward(self,x):
        out = self.liner_layers(x)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self,x):
        encoder_ = self.encoder(x)
        out = self.decoder(encoder_)
        out = out.reshape(-1,1,28,28)
        return out
if __name__ == '__main__':
    net = Net().cuda()
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()
    mnist_data = datasets.MNIST("D:\data\MNIST_data",train=True,transform=transforms.ToTensor(),download=False)
    train_loader = DataLoader(mnist_data,100,shuffle=True)
    k = 0
    for epoch in range(1000):
        for i,(img,label) in enumerate(train_loader):
            img = img.cuda()
            out_img = net(img)
            loss = loss_func(out_img,img)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i%10 ==0:
                print(loss.item())
                fake_img = out_img.detach()
                save_image(fake_img,"img/{}-fack_img.png".format(k),nrow=10)
                save_image(img,"img/{}-real_img.png".format(k),nrow=10)
                k+=1