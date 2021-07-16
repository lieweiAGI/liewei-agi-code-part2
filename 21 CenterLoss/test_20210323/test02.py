import torch

data = torch.tensor([[3,4],[5,6],[7,8],[9,8],[6,5]])
label = torch.Tensor([0,0,1,0,1])
center = torch.tensor([[1,1],[2,2]])

center_exp = center.index_select(dim=0,index=label.long())
print(center_exp)

count = torch.histc(label,bins=2,min=0,max=1)
print(count)

count_exp = count.index_select(dim=0,index=label.long())
print(count_exp)

center_loss = torch.sum(torch.div(torch.sqrt(torch.sum(torch.pow(data-center_exp,torch.tensor(2)),dim=1).float()),count_exp))
print(center_loss)
