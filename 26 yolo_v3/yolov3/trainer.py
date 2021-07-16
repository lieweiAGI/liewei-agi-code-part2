import dataset
from module import *


def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

    mask_obj = target[..., 0] > 0
    mask_noobj = target[..., 0] == 0

    loss_obj = torch.mean((output[mask_obj] - target[mask_obj]) ** 2)
    loss_noobj = torch.mean((output[mask_noobj][0] - target[mask_noobj][0]) ** 2)
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj
    return loss


if __name__ == '__main__':

    myDataset = dataset.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=2, shuffle=True)

    net = Darknet53()
    net.train()

    opt = torch.optim.Adam(net.parameters())

    for target_13, target_26, target_52, img_data in train_loader:
        output_13, output_26, output_52 = net(img_data)
        loss_13 = loss_fn(output_13, target_13, 0.9)
        loss_26 = loss_fn(output_26, target_26, 0.9)
        loss_52 = loss_fn(output_52, target_52, 0.9)

        loss = loss_13 + loss_26 + loss_52

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(loss.item())
