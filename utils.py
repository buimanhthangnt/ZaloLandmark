import tensorflow as tf
import numpy as np
import config


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
