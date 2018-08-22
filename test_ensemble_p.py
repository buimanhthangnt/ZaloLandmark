from __future__ import division
import os
import numpy as np
import pickle
import csv
import tensorflow as tf
import config
import utils


mode1 = 'res'
mode2 = 'v3'
# feature_shape = utils.get_input_shape(mode)


def next_batch_test():
    path = 'x_test_'
    for fil in sorted(os.listdir(path + mode1)):
        x_batch, paths = pickle.load(open(os.path.join(path + mode1, fil), 'rb'))
        yield x_batch, paths


# _, _, x_test = pickle.load(open('data.pickle', 'rb'))

# X = tf.placeholder(dtype=tf.float32, shape=feature_shape)

# pred = utils.top_layers(X, mode)
# pred = tf.layers.dense(pred, config.num_classes, activation='softmax')

# sess =  tf.Session()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()

# saver.restore(sess, './model_' + mode + '/' + mode + '.ckpt')
results = []
ids = []
# predictions = []
prediction1 = pickle.load(open('tmp/' + mode1 + '.pickle', 'rb'))
prediction2 = pickle.load(open('tmp/' + mode2 + '.pickle', 'rb'))
for i, x in enumerate(next_batch_test()):
    prediction = 0.6 * prediction1[i] + 0.4 * prediction2[i]
    prediction = np.argsort(prediction, axis=1)
    prediction = prediction[:,::-1]
    prediction = prediction[:,:3].tolist()
    ids.extend(x[1])
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
