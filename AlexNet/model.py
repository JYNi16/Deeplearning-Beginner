import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential,metrics, losses

class AlexNet(layers.Layer):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = layers.Conv2D(96, kernel_size=11, strides=4, activation="relu")
        self.s1 = layers.MaxPool2D(pool_size=3, strides=2)
        self.c2 = layers.Conv2D(256, kernel_size=5, padding="same", activation="relu")
        self.s2 = layers.MaxPool2D(pool_size=3, strides=2)
        self.c3 = layers.Conv2D(384, kernel_size=3, padding="same", activation="relu")
        self.c4 = layers.Conv2D(384, kernel_size=3, padding="same", activation="relu")
        self.c5 = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")
        self.s3 = layers.MaxPool2D(pool_size=3, strides=2)
        self.f1 = layers.Flatten()
        self.d1 = layers.Dense(4096)
        self.r6 = layers.Activation('relu')
        self.dro1 = layers.Dropout(0.5)
        self.d2 = layers.Dense(4096)
        self.r7 = layers.Activation('relu')
        self.dro2 = layers.Dropout(0.5)
        self.d3 = layers.Dense(10)

    def call(self, x):
        x = self.c1(x)
        x = self.s1(x)
        x = self.c2(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.s3(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.r6(x)
        x = self.dro1(x)
        x = self.d2(x)
        x = self.r7(x)
        x = self.dro2(x)
        x = self.d3(x)
        return x

if __name__=="__main__":
    model = AlexNet()
    a = tf.ones((1, 224, 224, 1)) # 这部可以用来测试搭建的网络输入输出是否正确
    b = model(a)
    print(b.shape)