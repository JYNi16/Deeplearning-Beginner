import torch
from torch import nn
import torchvision
from dataset import train_data

def net():
    network = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))
    #initinalize the weights of network
    for m in network:
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    return network

def Acc(y_pred, y_truth):
    pred = y_pred.argmax(dim=1)
    return (pred.data.cpu() == y_truth.data).sum()

def train():
    traindata = train_data(32, "data")
    dataloader = traindata.dataload()
    model = net()
    Updater = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    model.train()
    model.to(device)
    for e in range(10):
        loss_all = 0
        acc = 0.0
        for step, data in enumerate(dataloader):
            x, y = data
            Updater.zero_grad()
            y_pred = model(x.to(device, torch.float))
            loss = loss_function(y_pred, y.to(device, torch.long))
            loss.backward()
            loss_all += float(loss.data.cpu())
            acc += Acc(y_pred, y)
            Updater.step()
        print("epoch is:", e, "loss is %.4f"%(loss_all/step), "accuracy is %.4f"%(acc / (step * 32)))

if __name__=="__main__":
    train()
