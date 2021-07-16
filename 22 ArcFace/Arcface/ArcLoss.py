# Coding:utf-8
# Author:Naturino
import torch
import torch.nn as nn
import torch.nn.functional as F

class Arc(nn.Module):
    def __init__(self,feature_dim=2,cls_dim=10):
        super().__init__()
        self.W=nn.Parameter(torch.randn(feature_dim,cls_dim))

    def forward(self, feature,m=1,s=10):
        x=F.normalize(feature,dim=1)#x/||x||
        w = F.normalize(self.W, dim=0)#w/||w||
        cos = torch.matmul(x, w)/10
        print(cos)
        print(x)
        print(w)
        a=torch.acos(cos)
        top=torch.exp(s*torch.cos(a+m))
        down2=torch.sum(torch.exp(s*torch.cos(a)),dim=1,keepdim=True)-torch.exp(s*torch.cos(a))
        print(a)
        print(down2)
        out=torch.log(top/(top+down2))
        return out

if __name__ == '__main__':

    arc=Arc(2,10)
    data=torch.randn(3,2)
    out=arc(data)
    print("data===>",data)
    print('out===>',out)
    print(torch.sum(out,dim=1))

