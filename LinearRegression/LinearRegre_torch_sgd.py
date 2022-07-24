import random
import torch

def synthetic_data(w, b, num_examples):  # @save
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.2, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def net(X, w, b):  # @save
    """线性回归模型。"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  # @save
    """均方损失。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  # @save
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_batch(features, labels, batch_size, lr, num_epochs, w, b):
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = squared_loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_loss = squared_loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')
    print("w is:", w[0].item(), w[1].item(), "b is:", b.item())

def main():
    lr = 0.03
    num_epochs = 100
    batch_size = 20
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    train_batch(features, labels, batch_size, lr, num_epochs, w, b)

if __name__ == "__main__":
    main()