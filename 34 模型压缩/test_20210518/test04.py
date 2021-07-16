import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# model = LeNet().to(device=device)

# module = model.conv1
# print(list(module.named_parameters()))
#
# print(list(module.named_buffers()))
# #随机非结构化剪枝
# prune.random_unstructured(module, name="weight", amount=0.3)
# print(list(module.named_parameters()))
# print(list(module.named_buffers()))
# print(module.weight)
# print(module._forward_pre_hooks)
# #L1非结构化剪枝
# prune.l1_unstructured(module, name="bias", amount=3)
# print(list(module.named_parameters()))
# print(list(module.named_buffers()))
# print(module.bias)
# print(module._forward_pre_hooks)
#
# #L2结构化剪枝
# prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
# print(module.weight)
#
# for hook in module._forward_pre_hooks.values():
#     if hook._tensor_name == "weight":  # select out the correct hook
#         break
#
# print(list(hook))  # pruning history in the container
#
# print(list(module.named_parameters()))
# print(list(module.named_buffers()))
#
#
# prune.remove(module, 'weight')
# print(list(module.named_parameters()))
# print("=============")
# print(list(module.named_buffers()))

# new_model = LeNet()
# for name, module in new_model.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         prune.ln_structured(module, n = 2,name='weight', amount=0.2,dim=0)
#     # prune 40% of connections in all linear layers
#     elif isinstance(module, torch.nn.Linear):
#         prune.l1_unstructured(module, name='weight', amount=0.4)
#
# print(dict(new_model.named_buffers()).keys())  # to verify that all masks exist
# print(new_model.conv1.weight)

model = LeNet()

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
