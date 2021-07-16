import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import save_image

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,128,3,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 512,3,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*512,128),
            nn.Tanh()
        )
    def forward(self,x):
        conv1_out = self.conv1(x)
        conv1_out = conv1_out.reshape(-1,7*7*512)
        out = self.fc(conv1_out)
        return out

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128,7*7*512,bias=False),
            nn.BatchNorm1d(7*7*512),
            nn.LeakyReLU()
        )
        self.conv1_tr = nn.Sequential(
            nn.ConvTranspose2d(512,128,3,2,1,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128,1,3,2,1,1),
            nn.LeakyReLU(),
        )

    def forward(self,x):
        fc_out = self.fc(x)
        fc_out = fc_out.reshape(-1,512,7,7)
        out = self.conv1_tr(fc_out)
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