import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics
from model import ResNetModel
from dataset import data
from time import time

def train(network, batch_size, optimizer, acc_meter):
    data_load = data()
    train_iter = data_load.train_data(batch_size)
    for epoch in range(20):
        loss_all = 0
        iter = 0
        for step, (x, y) in enumerate(train_iter):
            iter += 1
            with tf.GradientTape() as tape:
                x = tf.reshape(x, (-1, 32, 32, 3))
                out = network(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.square(out - y_onehot)
                loss = tf.reduce_sum(loss) / batch_size
                loss_all += loss
            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
            acc_meter.update_state(tf.argmax(out, axis=1), y)

        print("epoch:", epoch, "| Loss is:%.4f"%(loss_all/iter), "|Accuracy is:%.4f"%(acc_meter.result().numpy()))
        acc_meter.reset_states()

if __name__=="__main__":
    network = ResNetModel()
    batch_size = 128
    optimizer = optimizers.Adam(learning_rate=0.0015)
    acc_meter = metrics.Accuracy()
    t1 = time()
    train(network, batch_size, optimizer, acc_meter)
    t2 = time()
    print("training time is%6f:"%(t2-t1))