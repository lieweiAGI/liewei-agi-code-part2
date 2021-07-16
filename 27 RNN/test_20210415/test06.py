import torch

a = torch.tensor([[1,2,3]])

print(a.shape)
print(a)

a = a.expand(3,3)
print(a.shape)
print(a)