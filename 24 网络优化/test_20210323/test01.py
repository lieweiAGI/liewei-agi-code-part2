import torch
from torch import nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

save_path = "models/net.pth"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784,512,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512,256,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256,128,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2,10),
            nn.Softmax()
        )
    def forward(self,x):
        fc1_out = self.fc1(x)
        out = self.fc2(fc1_out)
        return out,fc1_out

def show_feature(feat,label,epoch):
    plt.ion()
    c = ["#ff0000","#ffff00","#00ff00","#00ffff","#0000ff",
         "#ff00ff","#990000","#999900","#009900","#009999"]
    plt.clf()
    for i in range(10):
        plt.plot(feat[label == i,0],feat[label == i,1],".",c=c[i])
    plt.legend(["0",'1','2','3','4','5','6','7','8','9'],loc="upper right")
    plt.title("epoch=%d" % epoch)
    plt.savefig("images/epoch=%d.jpg"%epoch)
    plt.pause(0.001)
    plt.ioff()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torchvision.datasets.MNIST(root="D:\data\MNIST_data",download=True,train=True,transform=transforms.ToTensor())
train_loader = data.DataLoader(dataset=train_data,shuffle=True,batch_size=512)

if __name__ == '__main__':
    net = Net().to(device)
    # if os.path.exists(save_path):
    #     net = torch.load(save_path)
    loss_func = nn.MSELoss()
    # loss_func = nn.CrossEntropyLoss()
    # opt = torch.optim.SGD(net.parameters(),lr=0.0001)
    opt = torch.optim.Adam(net.parameters())
    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        for i,(x,y) in enumerate(train_loader):
            x = x.reshape(-1,784).to(device)
            target = torch.zeros(y.size(0),10).scatter_(1,y.reshape(-1,1),1).to(device)
            # target = y.to(device)

            out_put,feat = net(x)
            loss = loss_func(out_put,target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            feat_loader.append(feat)
            label_loader.append(y)

            if i%10 ==0:
                print(loss.item())
        epoch+=1
        feat = torch.cat(feat_loader,0)
        labels = torch.cat(label_loader,0)
        show_feature(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)
        torch.save(net.state_dict(),save_path)

