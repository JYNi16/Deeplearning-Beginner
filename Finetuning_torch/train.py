import torch
from tqdm import tqdm
from model import *
from optim import get_optim
from dataset import train_data
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([transforms.Resize([224, 224]),
    transforms.ToTensor(), ])
path = "E:/deeplearning/tensorflow/Tensorflow Input Pipeline/dogs-cats/train"
datatrain = train_data(path, data_transform)
dataloader = DataLoader(datatrain, batch_size=32)
feature_extract = True
model_name = "alexnet"
num_classes = 10
model_ft, input_size = initialize_model(model_name, num_classes,feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)
optimizer_ft = get_optim(model_ft, feature_extract)

def Acc(y_pred, y_truth):
    pred = y_pred.argmax(dim=1)
    return (pred.data.cpu() == y_truth.data).sum()

def train():
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(10):
        loss_all = 0
        acc = 0
        step = 0
        for data in tqdm(dataloader):
            x, y = data
            step += 1
            optimizer_ft.zero_grad()
            y_pred = model_ft(x.to(device, torch.float))
            loss = loss_function(y_pred, y.to(device, torch.long))
            loss.backward()
            acc += Acc(y_pred, y)
            optimizer_ft.step()
        print("epoch is:", epoch, "loss is %.4f" % (loss_all / step), "accuracy is %.4f" % (acc / (step * 32)))

if __name__=="__main__":
    train()