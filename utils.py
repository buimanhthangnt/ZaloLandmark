import tensorflow as tf
import numpy as np
import config
from collections import defaultdict


def top_layers(X, mode):
    if mode == 'res':
        pred = tf.layers.average_pooling2d(X, (7, 7), strides=(7,7))
        pred = tf.layers.flatten(pred)
    elif mode == 'inc':
        pred = tf.reduce_mean(X, axis=[1,2])
    elif mode == 'v3':
        pred = tf.reduce_mean(X, axis=[1,2])
    elif mode == 'xce':
        pred = tf.reduce_mean(X, axis=[1,2])
    return pred


def get_input_shape(mode):
    feature_shape = None
    if mode == 'res':
        feature_shape = [None, 7, 7, 2048]
    elif mode == 'inc':
        feature_shape = [None, 8, 8, 1536]
    elif mode == 'v3':
        feature_shape = [None, 8, 8, 2048]
    elif mode == 'xce':
        feature_shape = [None, 10, 10, 2048]
    return feature_shape


def get_image_size(mode):
    image_size = None
    if mode == 'res':
        image_size = config.image_size_res
    elif mode == 'inc':
        image_size = config.image_size_inc
    elif mode == 'v3':
        image_size = config.image_size_v3
    elif mode == 'xce':
        image_size = config.image_size_xce
    return image_size


def cal_weights(data, num_classes=103):
    weights = [0 for i in range(num_classes)]
    for y in data:
        idx = np.argmax(y)
        weights[idx] += 1
    weights = np.array(weights)
    mysum = np.sum(weights)
    weights = weights / mysum
    return np.array(weights)


def upsampling(data_x, data_y, num_classes):
    weights = cal_weights(data_y, num_classes)
    weights = weights / np.max(weights)
    weights = np.sqrt(weights) - weights
    dict_samples = defaultdict(list)
    idx_y = np.argmax(data_y, axis=-1)
    for idx, y in enumerate(data_y):
        y = np.argmax(y)
        dict_samples[y].append(idx)
    max_length = np.max(np.array([len(dict_samples[key]) for key in dict_samples]))
    for idx, weight in enumerate(weights):
        num_extra = int(np.ceil(weight * max_length))
        if num_extra == 0: continue
        rand_idx = np.random.randint(len(dict_samples[idx]), size=num_extra)
        extras_x = data_x[np.array(dict_samples[idx])[rand_idx]]
        data_x = np.concatenate((data_x, extras_x), axis=0)
        extras_y = np.zeros((num_extra, num_classes))
        extras_y[:,idx] = 1
        data_y = np.concatenate((data_y, extras_y))
    return data_x, data_y
