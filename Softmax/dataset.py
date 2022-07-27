import torch
import torchvision
import os

class train_data():
    def __init__(self, batch_size, data_path):
        self.datapath = data_path
        if os.path.exists(data_path):
            self.download = False
        else:
            self.download = True
        self.batch_size = batch_size

    def dataload(self):
        data = torchvision.datasets.MNIST(self.datapath, train=True, download=self.download,transform=torchvision.transforms.ToTensor())
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size)
        return data_loader

if __name__=="__main__":
    traindata = train_data(32, "data")
    dataloader = traindata.dataload()
    for x, y in dataloader:
        print("x.shape is:", x.shape)
        print("y.shape is:", y.shape)