import torch
from torch import nn

gru = nn.GRU(10,5,6,batch_first=True)

input = torch.randn(50,30,10)
# h_0 = torch.zeros(6,50,5)

out_put,h_n = gru(input)

print(out_put.shape)
print(h_n.shape)