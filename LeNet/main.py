#main file 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, metrics
from model import LeNet
from dataset import mnist
import config

def train(train_data):
    optimizer = optimizers.SGD(lr=0.005)
    acc_meter = metrics.Accuracy()

    model = LeNet()
    loss_all = 0
    for epoch in range(25):
        for data in train_data:
            with tf.GradientTape() as tape:
                x, y = data
                x = tf.reshape(x, [-1, 28, 28, 1])
                out = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.square(out - y_onehot)
                loss = tf.reduce_sum(loss) / config.batch_size
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                acc_meter.update_state(tf.argmax(out, axis=1), y)
        print("epoch:", epoch, "| Loss is:%.4f" % (loss), "|Accuracy is:%.4f" % (acc_meter.result().numpy()))
        acc_meter.reset_states()

if __name__ == '__main__':
    data = mnist()
    train_data = data.train_data()
    train(train_data)
