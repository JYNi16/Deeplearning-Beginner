import torch
from torch import nn
from d2l import torch as d2l

class softmax():
    def __int__(self):
        self.batch_size = 16
        self.epoch = 10

    def data(self):
        train_iter, test_iter = d2l.load_data_fashion_mnist(16)
        return train_iter, test_iter

    def net(self):
        network = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
        for m in network:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        return network

    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = self.net()
        net.to(device)
        torch.set_grad_enabled(True)
        net.train()
        train_iter, test_iter = self.data()
        Updater = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_function = nn.CrossEntropyLoss()
        loss_all = 0
        iter = 0

        for e in range(10):
            for x, y in train_iter:
                y_pred = self.net()(x.to(x.to(device, torch.float)))
                loss = loss_function(y_pred, y.to(device, torch.long))
                loss_all += float(loss.data.cpu())
                Updater.zero_grad()
                loss.backward()
                Updater.step()
                iter += 1

            print("epoch is",e, "loss is %.4f"%(loss_all/iter))

if __name__=="__main__":
    a = torch.zeros(1, 1, 28, 28)
    model = softmax()
    model.train()


