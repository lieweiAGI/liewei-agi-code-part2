from torch import nn
import torch

m = nn.AdaptiveMaxPool2d((5,7))
input = torch.randn(1, 64, 18, 29)
output = m(input)
print(output.shape)
nn.NLLLoss()
nn.CrossEntropyLoss()
nn.BCELoss()
nn.BCEWithLogitsLoss()

torch.optim.Adam(weight_decay=0.2)