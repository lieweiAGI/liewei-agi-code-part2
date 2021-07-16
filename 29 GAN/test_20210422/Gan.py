import torch,os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import save_image

class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        out = self.layers(x)
        return out

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(128,256),
            nn.LeakyReLU(),
            nn.Linear(256,512),
            nn.LeakyReLU(),
            nn.Linear(512,784)
        )

    def forward(self,x):
        out = self.layer(x)
        return out

if __name__ == '__main__':
    d_net = D_Net().cuda()
    g_net = G_Net().cuda()

    loss_fun = nn.BCELoss()

    d_opt = torch.optim.Adam(d_net.parameters(),lr=0.0002,betas=(0.5,0.999))
    g_opt = torch.optim.Adam(g_net.parameters(),lr=0.0002,betas=(0.5,0.999))

    mnist_data = datasets.MNIST("D:\data\MNIST_data",train=True,transform=transforms.ToTensor(),download=False)
    train_loader = DataLoader(mnist_data,100,shuffle=True)

    for epoch in range(10000):
        for i,(img,_) in enumerate(train_loader):
            real_img = img.reshape(-1,784).cuda()
            real_label = torch.ones(img.size(0),1).cuda()
            fake_label = torch.zeros(img.size(0),1).cuda()

            real_out = d_net(real_img)
            d_real_loss = loss_fun(real_out,real_label)

            z = torch.randn(img.size(0),128).cuda()
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_fake_loss = loss_fun(fake_out,fake_label)

            d_loss = d_real_loss + d_fake_loss

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            #训练生成器
            z = torch.randn(img.size(0),128).cuda()
            fake_img = g_net(z)
            g_fake_out = d_net(fake_img)
            g_loss = loss_fun(g_fake_out,real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i%10==0:
                print("epoch==>",epoch,";d_loss:",d_loss.item(),";g_loss:",g_loss.item())
        real_img = real_img.reshape(-1,1,28,28)
        fake_img = fake_img.reshape(-1,1,28,28)
        save_image(real_img,"img/{}-real_img.jpg".format(epoch+1),nrow=10)
        save_image(fake_img,"img/{}-fake_img.jpg".format(epoch+1),nrow=10)