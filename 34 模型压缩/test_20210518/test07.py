import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset,DataLoader,SequentialSampler

class model(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(model, self).__init__()
        self.layer1 = nn.LSTM(input_dim,hidden_dim,output_dim)
        self.layer2 = nn.Linear(hidden_dim,output_dim)
    def forward(self,inputs):
        layer1_output,layer1_hidden = self.layer1(inputs)
        layer2_output = self.layer2(layer1_output)
        layer2_output = layer2_output[:,-1,:]
        return layer2_output
#创建小模型
model_student = model(input_dim=2,hidden_dim=8,output_dim=4)
#创建大模型
model_teacher = model(input_dim=2,hidden_dim=16,output_dim=4)

#设置输入数据
inputs = torch.randn(4,6,2)
true_label = torch.tensor([0,1,0,0])

dataset = TensorDataset(inputs,true_label)
sampler = SequentialSampler(inputs)
datalaoder = DataLoader(dataset=dataset,sampler=sampler,batch_size=2)

loss_func = CrossEntropyLoss()
#KL散度
kl_loss = nn.KLDivLoss()
optmizier = torch.optim.SGD(model_student.parameters(),lr=0.1,momentum=0.9)

for step,(x,y) in enumerate(datalaoder):
    out_student = model_student(x)
    out_teacher = model_teacher(x)

    loss_hard = loss_func(out_student,y)
    loss_kl = kl_loss(out_student,out_teacher)

    loss = 0.9*loss_kl + 0.1*loss_hard
    print(loss)

    optmizier.zero_grad()
    loss.backward()
    optmizier.step()
