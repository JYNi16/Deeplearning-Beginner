# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 22:57:26 2022

@author: 26526
"""

import os, glob 
import random, csv
import tensorflow as tf

def load_path(root, name2label):
    #root: 数据集根目录
    #name2label: 

    images, labels = [], []
    for name in name2label.keys():
        images += glob.glob(os.path.join(root, "*.jpg"))

    random.shuffle(images)
    for image in images:
        if "cat" in image:
            labels.append(1)
        elif "dog" in image:
            labels.append(0)
    for i in range(100):
        print("image is:", images[i])
        print("label is:", labels[i])
    
    assert len(images) == len(labels)

    return images, labels

def normalize(x):
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    x = (x - mean)/std
    return x

def load_data(x,y):
    # x: 图片的路径，y：图片标签
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [244, 244])
    #x = tf.image.resize(x, [112, 112])
    x = tf.image.random_crop(x, [224,224,3])
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)

    return x, y

def create_data(root, name2label, batchsz):
    images, labels = load_path(root, name2label)
    db_train = tf.data.Dataset.from_tensor_slices((images, labels))
    db_train = db_train.map(load_data).batch(batchsz)
    
    print(len(db_train))
    return db_train

if __name__=="__main__":
    root = "ResNet_selfdata/dataset/train"
    name2label = {"cat":0, "dog":1}
    batchsz = 32
    db_train = load_data(root, name2label)