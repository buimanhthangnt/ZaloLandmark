from __future__ import division
import os
import numpy as np
import pickle
import csv
import tensorflow as tf
import config


mode = 'res'
feature_shape = None

if mode == 'res':
    feature_shape = [None, 7, 7, 2048]
elif mode == 'inc':
    feature_shape = [None, 8, 8, 1536]
elif mode == 'v3':
    feature_shape = [None, 8, 8, 2048]
elif mode == 'xce':
    feature_shape = [None, 10, 10, 2048]


def next_batch_test():
    path = 'pickle_test'
    for fil in os.listdir(path):
        x_batch, paths = pickle.load(open(os.path.join(path, fil), 'rb'))
        yield x_batch, paths


_, _, x_test = pickle.load(open('data.pickle', 'rb'))

X = tf.placeholder(dtype=tf.float32, shape=feature_shape)

pred = None
if mode == 'res':
    pred = tf.layers.average_pooling2d(X, (7, 7), strides=(7,7))
    pred = tf.layers.flatten(pred)
elif mode == 'inc':
    pred = tf.reduce_mean(X, axis=[1,2])
elif mode == 'v3':
    pred = tf.reduce_mean(X, axis=[1,2])
elif mode == 'xce':
    pred = tf.reduce_mean(X, axis=[1,2])

pred = tf.layers.dense(pred, config.num_classes)

sess =  tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess, './model_' + mode + '/' + mode + '.ckpt')
results = []
ids = []
for x_batch, paths in next_batch_test():
    prediction = sess.run(pred, feed_dict={X: x_batch})
    prediction = np.argsort(prediction, axis=1)
    prediction = prediction[:,::-1]
    prediction = prediction[:,:3].tolist()
    ids.extend(paths)
    results.extend(prediction)

title = ["id", "predicted"]
content = []
content.append(title)
for col1, col2 in zip(ids, results):
    col2 = ' '.join(str(ele) for ele in col2)
    content.append([col1, col2])

save_path = 'results/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
result_file = open(os.path.join(save_path, mode + '.csv'), 'w')
with result_file:
    writer = csv.writer(result_file)
    writer.writerows(content)
