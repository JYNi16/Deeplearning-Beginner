# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 22:57:26 2022
@author: 26526
"""
import os, glob 
import random, csv
import tensorflow as tf

class DC_data():
    def __init__(self, data_path, batch_size=32):
        self.images_path = []
        self.data_path = data_path
        self.images_path = os.listdir(data_path)
        self.batch_size = batch_size

    def load_path(self):
        # random.shuffle(self.images_path)
        images, labels = [], []
        for image in self.images_path:
            # print("image is:", image)
            images.append(os.path.join(self.data_path, image))
            if "cat" in image:
                labels.append(1)
            else:
                labels.append(0)
        assert len(self.images_path) == len(labels)

        return images, labels

    def normalize(self, x):
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        x = (x - mean) / std
        return x

    def load_data(self, x, y):
        # x: 图片的路径，y：图片标签
        x = tf.io.read_file(x)
        x = tf.image.decode_jpeg(x, channels=3)  # RGBA
        x = tf.image.resize(x, [244, 244])
        # x = tf.image.resize(x, [112, 112])
        x = tf.image.random_crop(x, [224, 224, 3])
        x = tf.cast(x, dtype=tf.float32) / 255.
        x = self.normalize(x)
        y = tf.convert_to_tensor(y)
        return x, y

    def create_data(self):
        images, labels = self.load_path()
        db_train = tf.data.Dataset.from_tensor_slices((images, labels))
        # db_train = db_train.map(self.load_data).batch(self.batch_size)
        train_data = db_train.map(lambda images, label :
                                  tf.py_function(self.load_data,inp=[images, label],
                                                 Tout=[tf.float32, tf.int32]),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data = train_data.batch(self.batch_size)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
        return train_data

if __name__=="__main__":
    root = "E:/deeplearning/tensorflow/Tensorflow Input Pipeline/dogs-cats/train"
    batchsz = 32
    db_train = DC_data(root, 32)
    data = db_train.create_data()

    for x, y in data:
        print("x.shape is:", x.shape)
