import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Reshape

class ResBlockUp(layers.Layer):
    def __init__(self, out_channel):
        super(ResBlockUp, self).__init__()
        self.c1 = layers.Conv2D(out_channel, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.r1 = layers.Activation("relu")
        self.c2 = layers.Conv2D(out_channel, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.r2 = layers.Activation("relu")
    def call(self, x):
        res = x
        x = self.c1(x)
        x = self.bn1(x)
        x = self.r1(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = x + res
        x = self.r2(x)
        return x

class ResBlockDown(layers.Layer):
    def __init__(self, out_channel):
        super(ResBlockDown, self).__init__()
        self.p1 = layers.Conv2D(out_channel, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.c1 = layers.Conv2D(out_channel, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.r1 = layers.Activation("relu")
        self.c2 = layers.Conv2D(out_channel, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.r2 = layers.Activation("relu")

    def call(self, x):
        res = x
        res = self.p1(x)
        x = self.c1(x)
        x = self.bn1(x)
        x = self.r1(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = x + res
        x = self.r2(x)
        return x

class ResNetModel(layers.Layer):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.c1 = layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.r1 = layers.Activation("relu")
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        self.c2 = Sequential([
            ResBlockDown(64),
            ResBlockUp(64),
            ResBlockUp(64)]
        )
        self.c3 = Sequential([
            ResBlockDown(128),
            ResBlockUp(128),
            ResBlockUp(128)]
        )
        self.c4 = Sequential([
            ResBlockDown(256),
            ResBlockUp(256),
            ResBlockUp(256),
            ResBlockUp(256),
            ResBlockUp(256),
            ResBlockUp(256)]
        )
        self.c5 = Sequential([
            ResBlockDown(512),
            ResBlockUp(512),
            ResBlockUp(512)]
        )
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(10)
    def call(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = self.r1(x)
        x = self.pool1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.avgpool(x)
        #x = tf.reshape(x, [1,-1])
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = ResNetModel()
    a = tf.ones((1, 32, 32, 3))
    b = model(a)
    print(b.shape)