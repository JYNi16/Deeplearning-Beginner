import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, metrics
from funetuing import vgg
from dataset import DC_data

def train(train_data):
    optimizer = optimizers.SGD(lr=0.005)
    acc_meter = metrics.Accuracy()

    model = vgg()
    loss_all = 0
    for epoch in range(25):
        for data in train_data:
            with tf.GradientTape() as tape:
                x, y = data
                x = tf.reshape(x, [-1, 224, 224, 3])
                out = model(x)
                #y_onehot = tf.one_hot(y, depth=2)
                #loss = tf.square(out - y_onehot)
                #loss = tf.reduce_sum(loss) / config.batch_size
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred=out)
                loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                acc_meter.update_state(tf.argmax(out, axis=1), y)
        print("epoch:", epoch, "| Loss is:%.4f" % (loss), "|Accuracy is:%.4f" % (acc_meter.result().numpy()))
        acc_meter.reset_states()
    tf.saved_model(model, "saved/1")

if __name__ == '__main__':
    root = "E:/deeplearning/tensorflow/Tensorflow Input Pipeline/dogs-cats/train"
    batchsz = 8
    db_train = DC_data(root, 8)
    data = db_train.create_data()
    train(data)