import torch

input = torch.arange(180).reshape(1,2,2,3,15)
mask = input[..., 0] > 28
print(mask)
print("mask:===>",mask.shape)
idxs = mask.nonzero()

print(idxs.shape)
print(idxs)
print(idxs[:, 0].shape)
print(input[mask].shape)