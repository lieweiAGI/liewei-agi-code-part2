import torch,torchaudio
from torch import nn
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(1,8,32,16),
            nn.ReLU(),
            nn.Conv1d(8, 16,16,8),
            nn.ReLU(),
            nn.Conv1d(16, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2),
            nn.ReLU()
        )
        self.output_layer = nn.GRUCell(128,1)

    def forward(self,x):
        y = self.seq(x)

        output = []
        hx = torch.randn(1,1)
        for i in range(y.shape[2]):
            self.output_layer(y[:,:,i],hx)
            output.append(hx)
        output = torch.cat(output,dim=0)
        return output[-8:].reshape(-1,8)

# if __name__ == '__main__':
#     x = torch.randn(1,1,53075)
#     net = Net()
#     y = net(x)
#     print(y.shape)
if __name__ == '__main__':
    data_laoder = DataLoader(torchaudio.datasets.YESNO(".", download=True), batch_size=1, shuffle=True)
    net = Net()
    opt = torch.optim.Adam(net.parameters())

    loss_fn = torch.nn.MSELoss()

    for epoch in range(10000):
        for waveform,_,tag in data_laoder:
            y = net(waveform)
            loss = loss_fn(y,tag)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(loss)