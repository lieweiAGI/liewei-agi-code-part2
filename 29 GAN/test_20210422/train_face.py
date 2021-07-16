import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from DC_GAN_face import DCGAN
from face_data import FaceMyData

class Trainer():
    def __init__(self,root):
        faceMyData = FaceMyData(root)
        self.train_loader = DataLoader(faceMyData, 100, shuffle=True)

        self.net = DCGAN().cuda()

        self.d_opt = torch.optim.Adam(self.net.dnet.parameters(),0.0002,betas=(0.5,0.999))
        self.g_opt = torch.optim.Adam(self.net.gnet.parameters(), 0.0002, betas=(0.5, 0.999))

    def __call__(self):
        k = 0
        for epoch in range(100000):
            for i,(img) in enumerate(self.train_loader):
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
                    real_img = real_img.reshape(-1, 3, 96, 96)
                    fake_img = fake_img.reshape(-1, 3, 96, 96)
                    save_image(real_img, "img_face/{}-real_img.jpg".format(k + 1), nrow=10)
                    save_image(fake_img, "img_face/{}-fake_img.jpg".format(k + 1), nrow=10)
                    k+=1

if __name__ == '__main__':
    train = Trainer(r"E:\MyData\Cartoon_faces")
    train()