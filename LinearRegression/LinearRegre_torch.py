import torch
import random
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device('cpu')


# Create random input and output data
def synthetic_data(w, b, num_examples):
    # 生成 y = Xw + b + 噪声
    x = torch.normal(0, 1, (num_examples, 1))
    y = x * w + b
    y += torch.normal(0, 0.6, y.shape)
    return x, y.reshape((-1, 1))

def initial():
    w = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    return w, b

# net
def linreg(w, b, x):
    return b + w * x


def train_1(x, y, w, b, lr):
    # 训练步数
    for epoch in range(1000):
        y_pred = linreg(w, b, x)
        # print("y_pred.shape:", y_pred.shape)
        loss = (y_pred - y).pow(2).sum()
        if epoch % 10 == 0:
            print(epoch, loss.item())
        # 反向传播求梯度
        loss.backward()
        with torch.no_grad():
            b -= lr * b.grad
            w -= lr * w.grad
            # 梯度清零
            w.grad = None
            b.grad = None

    return (w.item(), b.item())

def main():
    true_w = -2.0
    true_b = 1.0
    x, y = synthetic_data(true_w, true_b, 1000)
    w, b = initial()
    w_pre, b_pre = train_1(x, y, w, b, 0.00001)
    print(w_pre, b_pre)

if __name__ == '__main__':
    main()