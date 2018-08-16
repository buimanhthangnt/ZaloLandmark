from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pickle
from sklearn.utils import shuffle
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split


num_classes = 103
num_epochs = 32
batch_size = 32
patience = 20
learning_rate = 0.03
decay = 0.9

model = ResNet50(weights='imagenet', include_top=False)
x_train, y_train, _ = pickle.load(open('data.pickle', 'rb'))
x_train, y_train = shuffle(x_train, y_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.08)


def next_batch(XXX, YYY, batch_size=batch_size):
    num_batch = len(YYY) // batch_size
    for i in range(num_batch):
        x_batch, y_batch = [], []
        for path, label in zip(XXX[i*batch_size:(i+1)*batch_size], YYY[i*batch_size:(i+1)*batch_size]):
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
        x_batch = preprocess_input(x_batch, mode='tf')
        x_batch = get_features(x_batch)

        y_batch = np.array(y_batch)

        yield x_batch, y_batch


def get_features(images):
    global model
    features = model.predict(images)
    return features


X = tf.placeholder(dtype=tf.float32, shape=[None, 7, 7, 2048])
y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

pred = tf.layers.average_pooling2d(X, (7, 7), strides=(7,7))
pred = tf.layers.flatten(pred)
pred = tf.layers.dense(pred, num_classes)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

sess =  tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
n_steps_no_improvement = 0
best_acc = 0
saver = tf.train.Saver()
for i in range(num_epochs):
    j = 0
    for x_feed, y_feed in next_batch(x_train, y_train):
        loss, _ = sess.run([loss_op, train_op], feed_dict={X: x_feed, y: y_feed})
        if j % 100 == 0:
            print("Epoch %d, batch %d: loss = %f" % (i, j, loss))
        if j % 600 == 0:
            accs = []
            for x_val_feed, y_val_feed in next_batch(x_val, y_val):
                y_pred = sess.run(pred, feed_dict={X: x_val_feed, y: y_val_feed})
                y_pred = np.argmax(y_pred, axis=1)
                y_val_label = np.argmax(y_val_feed, axis=1)
                acc = np.mean(y_pred == y_val_label)
                accs.append(acc)
            final_acc = np.mean(np.array(accs))
            print("Validation accuracy: %f" % final_acc)
            if final_acc > best_acc:
                best_acc = final_acc
                n_steps_no_improvement = 0
                saver.save(sess, './model/resnet.ckpt')
                print("Saved model")
            else:
                n_steps_no_improvement += 1
                print(str(n_steps_no_improvement) + " steps with no improvement")
                if n_steps_no_improvement > patience:
                    print("Best acc: %f" % (best_acc))
                    break
        j += 1
    if n_steps_no_improvement > patience:
        break
