from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pickle
from sklearn.utils import shuffle
import tensorflow as tf
import cv2


num_classes = 103
num_epochs = 3
batch_size = 128

model = ResNet50(weights='imagenet', include_top=False)


def get_features(images):
    global model
    features = model.predict(images)
    return features


def get_rand_test():
    _, _, x_test = pickle.load(open('data.pickle', 'rb'))
    x_test = shuffle(x_test)
    rand_choice = []
    for path in x_test[:10]:
        try:
            img = image.load_img(path, target_size=(224, 224))
        except:
            continue
        img = image.img_to_array(img)
        rand_choice.append(img)
        print(path)

    rand_choice = np.array(rand_choice)
    rand_choice = preprocess_input(rand_choice)
    rand_choice = get_features(rand_choice)
    return rand_choice


x_train = tf.placeholder(dtype=tf.float32, shape=[None, 7, 7, 2048])
y_train = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

pred = tf.layers.average_pooling2d(x_train, (7, 7), strides=(7,7))
pred = tf.layers.flatten(pred)
pred = tf.layers.dense(pred, num_classes)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=pred))
train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, './model/resnet.ckpt')
    x_test = get_rand_test()
    y_pred = sess.run(pred, feed_dict={x_train: x_test})
    print(np.argmax(y_pred, axis=1))
