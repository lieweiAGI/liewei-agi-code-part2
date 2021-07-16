import torch
from torch import nn
from torchvision.utils import save_image
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from DC_GAN import DCGAN

class Trainer():
    def __init__(self):
        self.mnist_data = datasets.MNIST("D:\data\MNIST_data", train=True, transform=transforms.ToTensor(), download=False)
        self.train_loader = DataLoader(self.mnist_data, 100, shuffle=True)

        self.net = DCGAN().cuda()

        self.d_opt = torch.optim.Adam(self.net.dnet.parameters(),0.0002,betas=(0.5,0.999))
        self.g_opt = torch.optim.Adam(self.net.gnet.parameters(), 0.0002, betas=(0.5, 0.999))

    def __call__(self):
        for epoch in range(100000):
            for i,(img,_) in enumerate(self.train_loader):
                real_img = img.cuda()
                noise_d = torch.normal(0,0.02,(100,128,1,1)).cuda()

                loss_d = self.net.get_D_loss(noise_d,real_img)

                self.d_opt.zero_grad()
                loss_d.backward()
                self.d_opt.step()

                noise_g = torch.normal(0,0.02,(100,128,1,1)).cuda()
                loss_g = self.net.get_G_loss(noise_g)

                fake_img = self.net(noise_g)

                self.g_opt.zero_grad()
                loss_g.backward()
                self.g_opt.step()

                if i % 10 == 0:
                    print("epoch==>", epoch, ";d_loss:", loss_d.item(), ";g_loss:", loss_g.item())
            real_img = real_img.reshape(-1, 1, 28, 28)
            fake_img = fake_img.reshape(-1, 1, 28, 28)
            save_image(real_img, "img_dc/{}-real_img.jpg".format(epoch + 1), nrow=10)
            save_image(fake_img, "img_dc/{}-fake_img.jpg".format(epoch + 1), nrow=10)

if __name__ == '__main__':
    train = Trainer()
    train()