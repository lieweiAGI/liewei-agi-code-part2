import torch

input = torch.rand(2,2)
print(input.dtype)
print(input)
Q = torch.quantize_per_tensor(input,scale=0.025,zero_point=0,dtype=torch.qint8)
print(Q.dtype)
print(Q)
Q = Q.dequantize()
print(Q)
print(Q.dtype)