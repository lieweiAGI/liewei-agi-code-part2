import torch
from torch import nn

class RNN_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(28,64,1,batch_first=True)
        self.fc = nn.Linear(64,10)
    def forward(self,x):
        x = x.reshape(-1,28,28)#NCHW==>NSV

        out_put,(h_n,c_n) = self.rnn(x)
        #取出最后一个S的V
        out = out_put[:,-1,:]
        out = self.fc(out)
        return torch.softmax(out,dim=1)