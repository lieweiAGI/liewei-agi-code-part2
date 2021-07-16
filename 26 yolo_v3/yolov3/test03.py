import torch

x = torch.tensor([[[1,2],[3,4]]]).float()
print(x)
x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
print(x)