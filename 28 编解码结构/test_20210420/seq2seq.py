import torch
from torch import nn
import os
import torch.utils.data as data
from data import MyDataset
import numpy as np

img_path = "data"
BATCH_SIZE = 64
EPOCH = 100
save_path = "params/seq2seq.pkl"

#编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.lstm = nn.LSTM(128,128,2,batch_first=True)
    def forward(self,x):
        x = x.reshape(-1,180,120).permute(0,2,1)
        x = x.reshape(-1,180)
        fc1_out = self.fc1(x)
        fc1_out = fc1_out.reshape(-1,120,128)
        out,(h_n,c_n) = self.lstm(fc1_out)
        out = out[:,-1,:]
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


