import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics
from model import ResNetmodel
from d2l import tensorflow as d2l


def train(network, batch_size, train_iter, optimizer, acc_meter):
    for step, (x, y) in enumerate(train_iter):
        with tf.GradientTape() as tape:
            out = network(x)
            y_onehot = tf.one_hot(y, depth=10)
            # print("y_onehot_shape:", y_onehot.shape)
            loss = tf.square(out - y_onehot)
            # print(loss)
            loss = tf.reduce_sum(loss) / batch_size
            # print(loss)
            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
            acc_meter.update_state(tf.argmax(out, axis=1), y)

            if step % 200 == 0:
                print("epoch:", step, "| Loss is:%.4f" % (loss), "|Accuracy is:%.4f" % (acc_meter.result().numpy()))
                acc_meter.reset_states()


if __name__ == "__main__":
    network = ResNetmodel()
    batch_size = 32
    network.build(input_shape=(batch_size, 32, 32, 3))
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=32)
    optimizer = optimizers.Adam(learning_rate=0.001)
    acc_meter = metrics.Accuracy()
    train(network, batch_size, train_iter, optimizer, acc_meter)