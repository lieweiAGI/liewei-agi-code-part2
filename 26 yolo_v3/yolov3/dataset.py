import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import cfg
import os

from PIL import Image
import math

LABEL_FILE_PATH = "data/person_label.txt"
IMG_BASE_DIR = "data"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b


class MyDataset(Dataset):

    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}

        line = self.dataset[index]
        strs = line.split()
        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))
        img_data = transforms(_img_data)

        _boxes = np.array([float(x) for x in strs[1:]])
        # _boxes = np.array(list(map(float, strs[1:])))
        boxes = np.split(_boxes, len(_boxes) // 5)
        index = 0
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))

            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)

                for i, anchor in enumerate(anchors):

                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    p_area = w * h
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    index+=1
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array([iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))])
                    print(feature_size,cx_index,cy_index,i)
        print(index)
        return labels[13], labels[26], labels[52], img_data

if __name__ == '__main__':
    data = MyDataset()
    print(data[0][0][...,0])
    print("============")
    print(data[0][1][...,0])
    print("============")
    print(data[0][2][...,0])

