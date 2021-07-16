import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader
from torch.nn import functional as F

def normalize(tensor):
    if __name__ == '__main__':
        tensor_minsmean = tensor - tensor.mean()
        return tensor_minsmean/tensor_minsmean.max()


tf = torchaudio.transforms.MFCC(sample_rate=8000)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1,4,(1,3),(1,2),(0,1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4,4, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3,2,1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,1,(8,1))
        )
    def forward(self,x):
        h = self.seq(x)
        return h.reshape(-1,8)
if __name__ == '__main__':

    data_laoder = DataLoader(torchaudio.datasets.YESNO(".",download=True),batch_size=1,shuffle=True)
    net = Net()
    opt = torch.optim.Adam(net.parameters())

    loss_fn = torch.nn.MSELoss()

    for epoch in range(10000000):
        datas = []
        tags = []
        for data,_,tag in data_laoder:
            tag = torch.stack(tag,dim=1).float()
            specgram = normalize(tf(data))
            datas.append(F.adaptive_avg_pool2d(specgram,(32,256)))
            tags.append(tag)
        specgrams = torch.cat(datas,dim=0)
        tags = torch.cat(tags,dim=0)
        y = net(specgrams)
        loss = loss_fn(y,tags)

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
