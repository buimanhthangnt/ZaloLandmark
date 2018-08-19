from __future__ import division
import numpy as np
import pickle
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import config
import utils


mode1 = 'res'
feature_shape1 = utils.get_input_shape(mode1)
mode2 = 'inc'
feature_shape2 = utils.get_input_shape(mode2)


x_train1, y_train1 = pickle.load(open(mode1 + '.pickle', 'rb'))
x_train2, y_train2 = pickle.load(open(mode2 + '.pickle', 'rb'))
x_train1 = np.expand_dims(x_train1, axis=1)
x_train2 = np.expand_dims(x_train2, axis=1)
x_train = np.concatenate((x_train1, x_train2), axis=-1)
y_train = y_train1

x_train, y_train = shuffle(x_train, y_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)


def next_batch(_X, _Y, batch_size=config.batch_size):
    _X, _Y = shuffle(_X, _Y)
    num_batch = int(np.ceil(len(_Y) / config.batch_size))
    for i in range(num_batch):
        x_batch1, x_batch2, y_batch = [], [], []
        for path, label in zip(_X[i*config.batch_size:(i+1)*config.batch_size], _Y[i*config.batch_size:(i+1)*config.batch_size]):
            path1, path2 = path
            feature_vec1 = pickle.load(open(path1, 'rb'))
            feature_vec2 = pickle.load(open(path2, 'rb'))
            x_batch1.append(feature_vec1)
            x_batch2.append(feature_vec2)
            y_batch.append(label)

        x_batch1 = np.array(x_batch1)
        x_batch2 = np.array(x_batch2)
        y_batch = np.array(y_batch)

        yield x_batch1, x_batch2, y_batch


X1 = tf.placeholder(dtype=tf.float32, shape=feature_shape1)
X2 = tf.placeholder(dtype=tf.float32, shape=feature_shape2)
y = tf.placeholder(dtype=tf.float32, shape=[None, config.num_classes])

pred1 = utils.top_layers(X1, mode1)
pred2 = utils.top_layers(X2, mode2)

pred = tf.concat([pred1, pred2], axis=-1)
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
    for x1_feed, x2_feed, y_feed in next_batch(x_train, y_train):
        loss, _ = sess.run([loss_op, train_op], feed_dict={X1: x1_feed, X2: x2_feed, y: y_feed})
        if j % 40 == 0:
            print("Epoch %d, batch %d: loss = %f" % (i, j, loss))
        if j % 200 == 0:
            accs = []
            for x1_val_feed, x2_val_feed, y_val_feed in next_batch(x_val, y_val):
                y_pred = sess.run(pred, feed_dict={X1: x1_val_feed, X2: x2_val_feed, y: y_val_feed})
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
                saver.save(sess, './model_' + mode1 + '_' + mode2 + '/' + mode1 + mode2 + '.ckpt')
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
        for x1_val_feed, x2_val_feed, y_val_feed in next_batch(x_val, y_val):
            y_pred = sess.run(pred, feed_dict={X1: x1_val_feed, X2: x2_val_feed, y: y_val_feed})
            y_pred = np.argmax(y_pred, axis=1)
            y_val_label = np.argmax(y_val_feed, axis=1)
            acc = np.mean(y_pred == y_val_label)
            accs.append(acc)
        final_acc = np.mean(np.array(accs))
        print("Top 1 acc: %f" % (final_acc))
        break
