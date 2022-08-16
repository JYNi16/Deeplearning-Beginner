import tensorflow as tf
from tensorflow import keras
from keras import datasets

class vgg(keras.Model):
    def __init__(self):
        super(vgg, self).__init__()
        self.model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
        self.model.trainable = False
        self.Flatten = keras.layers.Flatten()
        self.f1 = keras.layers.Dense(5)

    def call(self, x):
        x = self.model(x)
        x = self.Flatten(x)
        x = self.f1(x)
        return tf.nn.softmax(x)

if __name__=="__main__":
    a = tf.ones((1, 224, 224, 3))
    model = vgg()
    out = model(a)
    print("out.shape is:", out.shape)
