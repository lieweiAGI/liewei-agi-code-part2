import torch
from torch import nn
import os
import torch.utils.data as data
from data import MyDataset
import numpy as np

img_path = "data"
BATCH_SIZE = 64
EPOCH = 1000
save_path = "params/cnn2seq.pkl"

#编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,24,3,1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(24, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(512*4*14,128)
    def forward(self,x):
        layer_out = self.layers(x)
        layer_out = layer_out.reshape(-1,512*4*14)
        out = self.out(layer_out)
        return out
#解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(128,128,2,batch_first=True)
        self.out = nn.Linear(128,10)
    def forward(self,x):
        x = x.reshape(-1,1,128)
        x = x.expand(-1,4,128)
        lstm_out,(h_n,c_n) = self.lstm(x)
        lstm_out = lstm_out.reshape(-1,128)
        out = self.out(lstm_out)
        out = torch.softmax(out,dim=1)
        out = out.reshape(-1,4,10)
        return out

#主网络(使用编解码器)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x):
        encoder_ = self.encoder(x)
        out = self.decoder(encoder_)
        return out
if __name__ == '__main__':
    net = Net().cuda()
    opt = torch.optim.Adam(net.parameters())
    loss_fun = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))

    train_data = MyDataset(root="data")
    train_loader = data.DataLoader(train_data,BATCH_SIZE,shuffle=True)
    for epoch in range(EPOCH):
        for i,(x,y) in enumerate(train_loader):
            batch_x = x.cuda()
            batch_y = y.float().cuda()

            out = net(batch_x)

            loss = loss_fun(out,batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 5 == 0:
                test_y = torch.argmax(y, 2).numpy()
                perd_y = torch.argmax(out, 2).cpu().numpy()
                acc = np.mean(np.all(perd_y == test_y, axis=1))
                print("epoch:", epoch, "loss:", loss.item(), "acc:", acc)
                print("test_y:", test_y[0])
                print("perd_y:", perd_y[0])
        torch.save(net.state_dict(), save_path)
    #
    # x = torch.randn(1,3,60,120)
    # encoder = Encoder()
    # y = encoder(x)
    # print(y.shape)


