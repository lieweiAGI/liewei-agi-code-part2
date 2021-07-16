from module import *
import cfg


class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = Darknet53()
        self.net.eval()

    def forward(self, input, thresh, anchors):
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        #mask:N,H,W,3,15
        mask = output[..., 0] > thresh

        idxs = mask.nonzero()
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors)

        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x

        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])

        return torch.stack([n.float(), cx, cy, w, h], dim=1)


if __name__ == '__main__':
    detector = Detector()
    y = detector(torch.randn(3, 3, 416, 416), 0.3, cfg.ANCHORS_GROUP)
    print(y.shape)
