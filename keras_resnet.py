import numpy as np
import pickle
from sklearn.utils import shuffle
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import config

mode = 'res'
feature_shape = None

if mode == 'res':
    feature_shape = [None, 7, 7, 2048]
elif mode == 'inc':
    feature_shape = [None, 8, 8, 1536]
elif mode == 'v3':
    feature_shape = [None, 8, 8, 2048]

x_train, y_train = pickle.load(open(mode + '.pickle', 'rb'))
x_train, y_train = shuffle(x_train, y_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)


def next_batch(_X, _Y, batch_size=config.batch_size):
    _X, _Y = shuffle(_X, _Y)
    num_batch = len(_Y) // config.batch_size
    for i in range(num_batch):
        x_batch, y_batch = [], []
        for path, label in zip(_X[i*config.batch_size:(i+1)*config.batch_size], _Y[i*config.batch_size:(i+1)*config.batch_size]):
            feature_vec = pickle.load(open(path, 'rb'))
            x_batch.append(feature_vec)
            y_batch.append(label)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        yield x_batch, y_batch


X = tf.placeholder(dtype=tf.float32, shape=feature_shape)
y = tf.placeholder(dtype=tf.float32, shape=[None, config.num_classes])

pred = None
if mode == 'res':
    pred = tf.layers.average_pooling2d(X, (7, 7), strides=(7,7))
    pred = tf.layers.flatten(pred)
elif mode == 'inc':
    pred = tf.reduce_mean(X, axis=[1,2])
elif mode == 'v3':
    pred = tf.reduce_mean(X, axis=[1,2])

pred = tf.layers.dense(pred, config.num_classes)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss_op)

sess =  tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
n_steps_no_improvement = 0
best_acc = 0
saver = tf.train.Saver()
for i in range(config.num_epochs):
    j = 0
    for x_feed, y_feed in next_batch(x_train, y_train):
        loss, _ = sess.run([loss_op, train_op], feed_dict={X: x_feed, y: y_feed})
        if j % 40 == 0:
            print("Epoch %d, batch %d: loss = %f" % (i, j, loss))
        if j % 200 == 0:
            accs = []
            for x_val_feed, y_val_feed in next_batch(x_val, y_val):
                y_pred = sess.run(pred, feed_dict={X: x_val_feed, y: y_val_feed})
                y_pred = np.argsort(y_pred, axis=1)
                y_pred = y_pred[:,::-1]
                y_pred = y_pred[:,:3]
                y_val_label = np.argmax(y_val_feed, axis=1)
                acc = np.mean(np.array([y_val_label[k] in y_pred[k] for k in range(y_pred.shape[0])]))
                accs.append(acc)
            final_acc = np.mean(np.array(accs))
            print("Validation accuracy: %f" % final_acc)
            if final_acc > best_acc:
                best_acc = final_acc
                n_steps_no_improvement = 0
                saver.save(sess, './model_' + mode + '/' + mode + '.ckpt')
                print("Saved model")
            else:
                n_steps_no_improvement += 1
                print(str(n_steps_no_improvement) + " steps with no improvement")
                if n_steps_no_improvement > config.patience:
                    print("Top 3 acc: %f" % (best_acc))
                    break
        j += 1
    if n_steps_no_improvement > config.patience:
        accs = []
        for x_val_feed, y_val_feed in next_batch(x_val, y_val):
            y_pred = sess.run(pred, feed_dict={X: x_val_feed, y: y_val_feed})
            y_pred = np.argmax(y_pred, axis=1)
            y_val_label = np.argmax(y_val_feed, axis=1)
            acc = np.mean(y_pred == y_val_label)
            accs.append(acc)
        final_acc = np.mean(np.array(accs))
        print("Top 1 acc: %f" % (final_acc))
        break
