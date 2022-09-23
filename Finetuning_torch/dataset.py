import torch
from torchvision import transforms
import os, glob
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image

class train_data(nn.Module):
    def __init__(self, data_path, transform):
        self.datapath = []
        self.datapath.extend(glob.glob(os.path.join(data_path, "*.jpg")))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.datapath[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if "cat" in img_path:
            label = 0
        else:
            label = 1

        return img, label

    def __len__(self):
        return len(self.datapath)

if __name__=="__main__":
    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),])
    path = "E:/deeplearning/tensorflow/Tensorflow Input Pipeline/dogs-cats/train"
    datatrain = train_data(path, data_transform)
    dataloader = DataLoader(datatrain, batch_size=8)

    for x, y in dataloader:
        print("x.shape is:", x.shape)