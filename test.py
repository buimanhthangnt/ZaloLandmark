import os
import numpy as np
import glob
import cv2
import pickle
from keras.applications.resnet50 import ResNet50
import cv2
from sklearn.utils import shuffle
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import csv
import tensorflow as tf


model = ResNet50(weights='imagenet', include_top=False)
num_classes = 103
batch_size = 128


def get_features(images):
    global model
    features = model.predict(images)
    return features


def get_test(path):
    x = []
    for image in os.listdir(path):
        x.append(os.path.join(path, image))
    return np.array(x)


def next_batch_test(XXX, batch_size=batch_size):
    num_batch = len(XXX) // batch_size
    for i in range(num_batch):
        x_batch, paths = [], []
        for path in XXX[i*batch_size:(i+1)*batch_size]:
            try:
                img = image.load_img(path, target_size=(224, 224))
            except:
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


X = tf.placeholder(dtype=tf.float32, shape=[None, 7, 7, 2048])

pred = tf.layers.average_pooling2d(X, (7, 7), strides=(7,7))
pred = tf.layers.flatten(pred)
pred = tf.layers.dense(pred, num_classes)

sess =  tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess, './model/resnet.ckpt')
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
result_file = open('results.csv', 'w')
with result_file:
    writer = csv.writer(result_file)
    writer.writerows(content)
