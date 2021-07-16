import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import UNet
import MKDataset
from torchvision.utils import save_image

path = r'E:\MyData\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
module = r'./module.pth'
img_save_path = r'./train_img'
epoch = 1

net = UNet.MainNet().cuda()
optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.BCELoss()

dataloader = DataLoader(MKDataset.MKDataset(path), batch_size=4, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
else:
    print('No Params!')
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

while True:
    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.cuda()
        ys = ys.cuda()
        xs_ = net(xs)

        loss = loss_func(xs_, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print('epoch: {},  count: {},  loss: {}'.format(epoch, i, loss))

        # torch.save(net.state_dict(), module)
        print('module is saved !')

        x = xs[0]
        x_ = xs_[0]
        y = ys[0]
        # print(y.shape)
        img = torch.stack([x,x_,y],0)
        print(img.shape)

        save_image(img.cpu(), os.path.join(img_save_path,'{}.png'.format(i)))
        print("saved successfully !")
    epoch += 1