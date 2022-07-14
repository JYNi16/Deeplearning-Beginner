import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential,metrics, losses

class LeNet(keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = layers.Conv2D(6, kernel_size=3, strides=1, activation="relu")
        self.s1 = layers.MaxPool2D(pool_size=2, strides=2)
        self.c2 = layers.Conv2D(16, kernel_size=3, padding="same", activation="relu")
        self.s2 = layers.MaxPool2D(pool_size=2, strides=2)
        self.f1 = layers.Flatten()
        self.d1 = layers.Dense(100)
        self.r1 = layers.Activation('relu')
        self.d3 = layers.Dense(10)

    def call(self, x):
        x = self.c1(x)
        x = self.s1(x)
        x = self.c2(x)
        x = self.s2(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.r1(x)
        x = self.d3(x)
        return x

if __name__=="__main__":
    model = LeNet()
    a = tf.ones((32, 28, 28, 1)) # 这部可以用来测试搭建的网络输入输出是否正确
    b = model(a)
    print(b.shape)