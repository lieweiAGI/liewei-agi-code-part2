import torch
from torch.nn import functional as F

x = torch.randn(1,54088)
y = F.adaptive_avg_pool1d(x[None,...],(50000))
print(y.shape)