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
import config


model = InceptionResNetV2(weights='imagenet', include_top=False)


def get_features(images):
    global model
    features = model.predict(images)
    return features


def get(path):
    x = []
    y = []
    for folder in os.listdir(path):
        print('Processing ' + folder)
        for image in os.listdir(os.path.join(path, folder)):
            x.append(os.path.join(path, folder, image))
            y.append(int(folder))
    return np.array(x), np.array(y)


def get_test(path):
    x = []
    for image in os.listdir(path):
        x.append(os.path.join(path, image))
    return np.array(x)


def dump(data, name):
    pickle.dump(data, open(name, 'wb'), pickle.HIGHEST_PROTOCOL)


# x_train, y_train = get('TrainVal')
# x_test = get_test('Public')
x_train, y_train, _ = pickle.load(open('data.pickle', 'rb'))
x_train, y_train = shuffle(x_train, y_train)


def next_batch(XXX, YYY, batch_size=128):
    num_batch = len(YYY) // batch_size
    for i in range(num_batch):
        x_batch, y_batch = [], []
        for path, label in zip(XXX[i*batch_size:(i+1)*batch_size], YYY[i*batch_size:(i+1)*batch_size]):
            try:
                img = image.load_img(path, target_size=(config.image_size, config.image_size))
            except:
                continue

            img = image.img_to_array(img)
            x_batch.append(img)

            y_tmp = np.zeros((config.num_classes, ))
            y_tmp[label] = 1
            y_batch.append(y_tmp)

        x_batch = np.array(x_batch)
        x_batch = preprocess_input(x_batch)
        x_batch = get_features(x_batch)

        y_batch = np.array(y_batch)

        yield x_batch, y_batch


def dump_data():
    count = 0
    new_x_train, new_y_train = [], []
    for x, y in next_batch(x_train, y_train, batch_size=128):
        for e_x, e_y in zip(x, y):
            path = 'pickle_inc/' + str(count) + '.pickle'
            pickle.dump(e_x, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)
            new_x_train.append(path)
            new_y_train.append(e_y)
            count += 1
    new_x_train = np.array(new_x_train)
    new_y_train = np.array(new_y_train)
    dump((new_x_train, new_y_train), config.new_data_inception)

dump_data()

