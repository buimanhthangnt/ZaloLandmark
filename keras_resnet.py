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


def next_batch(batch_size=batch_size):
    x_train, y_train, _ = pickle.load(open('data.pickle', 'rb'))
    x_train, y_train = shuffle(x_train, y_train)
    num_batch = len(y_train) // batch_size
    for i in range(num_batch):
        x_batch, y_batch = [], []
        for path, label in zip(x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]):
            try:
                img = image.load_img(path, target_size=(224, 224))
            except:
                continue

            img = image.img_to_array(img)
            x_batch.append(img)

            y_tmp = np.zeros((num_classes, ))
            y_tmp[label] = 1
            y_batch.append(y_tmp)

        x_batch = np.array(x_batch)
        x_batch = preprocess_input(x_batch)
        x_batch = get_features(x_batch)

        y_batch = np.array(y_batch)

        yield x_batch, y_batch


def get_features(images):
    global model
    features = model.predict(images)
    return features


x_train = tf.placeholder(dtype=tf.float32, shape=[None, 7, 7, 2048])
y_train = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

pred = tf.layers.average_pooling2d(x_train, (7, 7), strides=(7,7))
pred = tf.layers.flatten(pred)
pred = tf.layers.dense(pred, num_classes)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=pred))
train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_epochs):
        j = 0
        for x_feed, y_feed in next_batch():
            loss, _ = sess.run([loss_op, train_op], feed_dict={x_train: x_feed, y_train: y_feed})
            if j % 50 == 0:
                print("Epoch %d, batch %d: loss = %f" % (i, j, loss))
            j += 1
    
    saver = tf.train.Saver()
    saver.save(sess, './model/resnet.ckpt')
    print("Saved")


# img = image.load_img('/media/thangbm/6CF2096DF2093CB8/Images/Wallpapers/1.jpg', target_size=(224, 224))
# img = image.img_to_array(img)
# img = np.expand_dims(img, 0)
# features = get_features(img)
# print(features.shape)
# print('Predicted:', decode_predictions(preds, top=3)[0])
