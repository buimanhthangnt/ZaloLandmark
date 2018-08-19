from __future__ import division
import os
import numpy as np
import pickle
import csv
import tensorflow as tf
import config
import utils


mode1 = 'res'
feature_shape1 = utils.get_input_shape(mode1)
mode2 = 'inc'
feature_shape2 = utils.get_input_shape(mode2)


def next_batch_test():
    path = 'x_test_'
    for fil in sorted(os.listdir(path + mode1)):
        x_batch1, paths = pickle.load(open(os.path.join(path + mode1, fil), 'rb'))
        x_batch2, paths = pickle.load(open(os.path.join(path + mode2, fil), 'rb'))
        yield x_batch1, x_batch2, paths


_, _, x_test = pickle.load(open('data.pickle', 'rb'))

X1 = tf.placeholder(dtype=tf.float32, shape=feature_shape1)
X2 = tf.placeholder(dtype=tf.float32, shape=feature_shape2)

pred1 = utils.top_layers(X1, mode1)
pred2 = utils.top_layers(X2, mode2)

pred = tf.concat([pred1, pred2], axis=-1)
pred = tf.layers.dense(pred, config.num_classes)

sess =  tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess, './model_' + mode1 + '_' + mode2 + '/' + mode1 + mode2 + '.ckpt')
results = []
ids = []
for x_batch1, x_batch2, paths in next_batch_test():
    prediction = sess.run(pred, feed_dict={X1: x_batch1, X2: x_batch2})
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
result_file = open(os.path.join(save_path, mode1 + '_' + mode2 + '.csv'), 'w')
with result_file:
    writer = csv.writer(result_file)
    writer.writerows(content)
