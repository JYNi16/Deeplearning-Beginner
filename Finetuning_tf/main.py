import tensorflow as tf
from tensorflow import keras
from keras import datasets

class vgg(keras.Model):
    def __init__(self):
        super(vgg, self).__init__()
        self.model = keras.applications.vgg16.VGG16()
        self.model.trainable = False

    def call(self, x):
        return self.model(x)

if __name__=="__main__":
    a = tf.ones((1, 224, 224, 3))
    model = vgg()
    out = model(a)
    print("a.shape is:", a.shape)
