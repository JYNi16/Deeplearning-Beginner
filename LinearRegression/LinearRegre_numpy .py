import numpy as np
import math
import matplotlib.pyplot as plt


# Create random input and output data
def synthetic_data(w, b, num_examples):
    # 生成 y = Xw + b + 噪声
    x = np.random.normal(0, 1, (num_examples, 1))
    print("x.shape:", x.shape)
    y = x * w + b
    y += np.random.normal(0, 0.6, y.shape)
    print("y.shape:", y.shape)
    return x, y


def initial():
    w = np.random.randn()
    b = np.random.randn()
    return w, b

# net
def linreg(w, b, x):
    return b + w * x

def train_1(x, y, w, b, lr):
    # 训练步数
    for t in range(1000):
        y_pred = linreg(w, b, x)
        loss = np.square(y_pred - y).sum()
        if t % 10 == 0:
            print(t, loss / 1000)
        # 求梯度
        grad_y_pred = 2.0 * (y_pred - y)
        grad_b = grad_y_pred.sum()
        grad_w = (grad_y_pred * x).sum()
        # 更新参数
        b -= lr * grad_b
        w -= lr * grad_w
    return (round(w, 4), round(b, 4))


def plot_fig(w_pre, b_pre, x, y):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=15)
    plt.plot(x, w_pre * x + b_pre, color='r', linewidth=3)
    plt.xlabel('x', fontproperties='Times New Roman', size=2)
    plt.ylabel('y', fontproperties='Times New Roman', size=2)
    plt.savefig('1.jpg', dpi=300)
    plt.show()


def main():
    true_w = -2.0
    true_b = 1.0
    x, y = synthetic_data(true_w, true_b, 2000)
    w, b = initial()
    w_pre, b_pre = train_1(x, y, w, b, 0.00001)
    #plot_fig(w_pre, b_pre, x, y)
    print(f'Result: y = {b_pre}{w_pre} x')

if __name__ == '__main__':
    main()