from torch.utils.data import Dataset
import cv2,os
import numpy as np

class FaceMyData(Dataset):
    def __init__(self,root):
        self.root = root
        self.dataset = os.listdir(root)
    def __len__(self):

        return len(self.dataset)
    def __getitem__(self, index):
        pic_name = self.dataset[index]
        img_data = cv2.imread(f"{self.root}/{pic_name}")
        img_data = img_data[...,::-1]
        img_data = img_data.transpose([2,0,1])
        img_data = ((img_data/255.-0.5)*2).astype(np.float32)
        return img_data
if __name__ == '__main__':
    data = FaceMyData(r"E:\MyData\Cartoon_faces")
    print(data[0].shape)