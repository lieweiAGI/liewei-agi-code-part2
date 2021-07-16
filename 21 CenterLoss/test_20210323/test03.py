import torch
from torch import nn
import numpy as np

class CenterLoss(nn.Module):
    def __init__(self,cls_num,feature_num):
        super(CenterLoss, self).__init__()

        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num,feature_num))

    def forward(self,xs,ys):
        center_exp = self.center.index_select(dim=0, index=ys.long())
        count = torch.histc(ys, bins=self.cls_num, min=0, max=self.cls_num-1)
        count_exp = count.index_select(dim=0, index=ys.long())
        return torch.sum(torch.div(torch.sqrt(torch.sum(torch.pow(xs-center_exp,torch.tensor(2)),dim=1).float()),count_exp))

class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(784,120),
            nn.ReLU(),
            nn.Linear(120,2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(2,10)
        )
        self.center_loss_layer = CenterLoss(10,2)
        self.crossEntropy = nn.CrossEntropyLoss()
    def forward(self,xs):
        features = self.hidden_layer(xs)
        outputs = self.output_layer(features)
        return features,outputs
    def getloss(self,features,outputs,labels):
        loss_cls = self.crossEntropy(outputs,labels)
        loss_center = self.center_loss_layer(features,labels)
        loss = loss_cls+loss_center
        return loss

if __name__ == '__main__':
    CenterLoss(10,2)
