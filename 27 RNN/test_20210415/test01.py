from torch import nn
import torch

rnn = nn.RNN(10,5,6,batch_first=True)
input = torch.randn(50,30,10)
# h0 = torch.zeros(6,50,5)

output,hn = rnn(input)
print(output.shape)
print(hn.shape)

