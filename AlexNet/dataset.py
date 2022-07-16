import tensorflow as tf
#from tensorflow.keras import layers, Model

class cifar():
    def __int__(self, batch_size):
        self.batch_size = batch_size

    def train_data(self):
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)/255
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.batch(32)

        return train_data

    def val_data(self):
        _, (x_val, y_val) = tf.keras.datasets.mnist.load_data()
        val_data = tf.data.Dataset.from_tensor_slices(x_val, y_val)
        val_data = val_data.batch(self.batch_size)

        return val_data

if __name__ == "__main__":
    data_mnist = cifar()
    train_data = data_mnist.train_data()

    for x, y in train_data:
        print("x.shaoe:", x.shape)
        print("y.shape:", y.shape)