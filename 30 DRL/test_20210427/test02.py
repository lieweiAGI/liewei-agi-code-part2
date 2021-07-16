import torch

a = torch.randn(3,2)
b = torch.tensor([[1],[0],[1]])

y = torch.gather(a,1,b)
print(a)
print(y)