from __future__ import division
import os
import numpy as np
import pickle
import csv
import tensorflow as tf
import config
import utils


mode = 'xce'
feature_shape = utils.get_input_shape(mode)


def next_batch_test():
    path = 'x_test_'
    for fil in sorted(os.listdir(path + mode)):
        x_batch, paths = pickle.load(open(os.path.join(path + mode, fil), 'rb'))
        yield x_batch, paths


_, _, x_test = pickle.load(open('data.pickle', 'rb'))

X = tf.placeholder(dtype=tf.float32, shape=feature_shape)

pred = tf.layers.batch_normalization(X)
pred = tf.nn.relu(pred)

pred = tf.layers.separable_conv2d(pred, 2048, (3,3), padding='same', use_bias=False)
pred = tf.layers.batch_normalization(pred)
pred = tf.nn.relu(pred)

pred = utils.top_layers(pred, mode)
pred = tf.layers.dense(pred, config.num_classes)

sess =  tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess, './best_model/' + mode + '.ckpt')
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
