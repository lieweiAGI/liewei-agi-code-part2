from torch import nn
import torch

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,64,5,3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,4,2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,256,4,2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256,512,4,2,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512,1,4,1,bias=False)
        )
    def forward(self,x):
        h = self.layers(x)
        return h.reshape(-1)

class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128,512,4,2,0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )
    def forward(self,x):
        return self.layers(x)

class DCGAN(nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.gnet = GNet().cuda()
        self.dnet = DNet().cuda()

        self.loss_func = nn.BCEWithLogitsLoss()
    def forward(self,noise):
        noise = noise.cuda()
        return self.gnet(noise)
    def get_D_loss(self,noise_d,real_img):
        noise_d = noise_d.cuda()
        real_img = real_img.cuda()
        real_y = self.dnet(real_img)
        g_img = self.gnet(noise_d)
        fake_y = self.dnet(g_img)

        real_tag = torch.ones(real_img.size(0)).cuda()
        fake_tag = torch.zeros(noise_d.size(0)).cuda()

        loss_real = self.loss_func(real_y,real_tag)
        loss_fake = self.loss_func(fake_y,fake_tag)
        loss_d = loss_real + loss_fake
        return loss_d

    def get_G_loss(self,noise_g):
        noise_g = noise_g.cuda()
        _g_img = self.gnet(noise_g)
        _real_y = self.dnet(_g_img)
        _real_tag = torch.ones(noise_g.size(0)).cuda()

        loss_g = self.loss_func(_real_y,_real_tag)
        return loss_g

if __name__ == '__main__':
    dnet = DNet()
    input = torch.randn(1,3,96,96)
    print(dnet(input).shape)

    gnet = GNet()
    x = torch.randn(1,128,1,1)
    print(gnet(x).shape)
    #
    gan = DCGAN()
    x = torch.randn(1,128,1,1)
    #
    d_loss = gan.get_D_loss(x,input)
    g_loss = gan.get_G_loss(x)
    print(d_loss,g_loss)