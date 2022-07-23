import tensorflow as tf
from tensorflow.keras import datasets

class data():
    def __init__(self):
        self.data_cifar10 = datasets.cifar10.load_data()

    def data_re(self):
        (x, y), (x_val, y_val) = self.data_cifar10
        return (x, y), (x_val, y_val)


    def train_data(self, batchsz_t):
        (x, y), _ = self.data_re()
        y = tf.squeeze(y, axis=1)
        x = tf.cast(x, dtype=tf.float32) / 255.
        y = tf.cast(y, dtype=tf.int32)
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.batch(batchsz_t)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset

    def val_data(self, batchsz_v):
        _, (x_val, y_val) = self.data_re()
        y_val = tf.squeeze(y_val, axis=1)
        x_val = tf.cast(x_val, dtype=tf.float32) / 255.
        y_val = tf.cast(y_val, dtype=tf.int32)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(batchsz_v)

        return val_dataset

if __name__=="__main__":
    data_load = data()
    data_it =  data_load.train_data(64)
    for x, y in data_it:
        print("x.shape:", x.shape)
        print("y.shape:", y.shape)