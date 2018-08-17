from __future__ import division
import os
import numpy as np
import glob
import cv2
import pickle
import cv2
from sklearn.utils import shuffle
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
import csv
import tensorflow as tf
import config


model = InceptionResNetV2(weights='imagenet', include_top=False)


def get_features(images):
    global model
    features = model.predict(images)
    return features


def get_test(path):
    x = []
    for image in os.listdir(path):
        x.append(os.path.join(path, image))
    return np.array(x)


def next_batch_test(XXX, batch_size=config.batch_size):
    num_batch = int(np.ceil(len(XXX) / config.batch_size))
    for i in range(num_batch):
        x_batch, paths = [], []
        for path in XXX[i*config.batch_size:(i+1)*config.batch_size]:
            try:
                img = image.load_img(path, target_size=(config.image_size, config.image_size))
            except:
                print(path)
                continue

            img = image.img_to_array(img)
            x_batch.append(img)

            path = path.split('/')[1].split('.')[0]
            paths.append(path)

        x_batch = np.array(x_batch)
        x_batch = preprocess_input(x_batch)
        x_batch = get_features(x_batch)

        paths = np.array(paths)

        yield x_batch, paths


_, _, x_test = pickle.load(open('data.pickle', 'rb'))


X = tf.placeholder(dtype=tf.float32, shape=config.feature_shape)

pred = tf.reduce_mean(X, axis=[1,2])
# pred = tf.layers.average_pooling2d(X, (7, 7), strides=(7,7))
# pred = tf.layers.flatten(pred)
pred = tf.layers.dense(pred, config.num_classes)

sess =  tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess, config.inception_resnet_model)
results = []
ids = []
for x_batch, paths in next_batch_test(x_test):
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
result_file = open('results_inc.csv', 'w')
with result_file:
    writer = csv.writer(result_file)
    writer.writerows(content)
