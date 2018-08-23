from __future__ import division
import numpy as np
import pickle
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import config
import utils


mode = 'xce'
feature_shape = utils.get_input_shape(mode)

x_train, y_train = pickle.load(open(mode + '.pickle', 'rb'))
x_train, y_train = shuffle(x_train, y_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.08, shuffle=True)

#x_train, y_train = utils.upsampling(x_train, y_train, config.num_classes)


def next_batch(_X, _Y, batch_size=config.batch_size):
    _X, _Y = shuffle(_X, _Y)
    num_batch = int(np.ceil(len(_Y) / config.batch_size))
    for i in range(num_batch):
        x_batch, y_batch = [], []
        for path, label in zip(_X[i*config.batch_size:(i+1)*config.batch_size], \
                               _Y[i*config.batch_size:(i+1)*config.batch_size]):
            feature_vec = pickle.load(open(path, 'rb'))
            x_batch.append(feature_vec)
            y_batch.append(label)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        yield x_batch, y_batch


X = tf.placeholder(dtype=tf.float32, shape=feature_shape)
y = tf.placeholder(dtype=tf.float32, shape=[None, config.num_classes])

pred = tf.layers.batch_normalization(X)
pred = tf.nn.relu(pred)

pred = tf.layers.separable_conv2d(pred, 2048, (3,3), padding='same', use_bias=False, \
                                  depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                                  pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                                  depthwise_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                  pointwise_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
pred = tf.layers.batch_normalization(pred)
pred = tf.nn.relu(pred)

pred = utils.top_layers(pred, mode)
pred = tf.layers.dense(pred, config.num_classes, \
                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                       bias_initializer=tf.contrib.layers.xavier_initializer())

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss_op)

sess =  tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
n_steps_no_improvement = 0
best_acc = 0
saver = tf.train.Saver()
# saver.restore(sess, './best_model/' + mode + '.ckpt')
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
