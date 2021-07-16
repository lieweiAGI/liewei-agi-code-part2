from torch import nn
import torch

lstm = nn.LSTM(10,5,6,batch_first=True)
input = torch.randn(50,30,10)

# h_0 = torch.zeros(6,50,5)
# c_0 = torch.zeros(6,50,5)
#
# out_put,(h_n,c_n) = lstm(input,(h_0,c_0))
out_put,(h_n,c_n) = lstm(input)

print(out_put.shape)
# print(h_n.shape)
# print(c_n.shape)