from __future__ import division
import numpy as np
import pickle
from keras.preprocessing import image

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inc_preprocess

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as res_preprocess

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as v3_preprocess

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xce_preprocess

import config
import utils


# model_inc = InceptionResNetV2(weights='imagenet', include_top=False)
# model_res = ResNet50(weights='imagenet', include_top=False)
# model_v3 = InceptionV3(weights='imagenet', include_top=False)
model_xce = Xception(weights='imagenet', include_top=False)
model_xce = Model(inputs=model_xce.input, outputs=model_xce.get_layer('block14_sepconv1').output)


def get_features_inc(images):
    global model_inc
    features = model_inc.predict(images)
    return features


def get_features_res(images):
    global model_res
    features = model_res.predict(images)
    return features


def get_features_v3(images):
    global model_v3
    features = model_v3.predict(images)
    return features


def get_features_xce(images):
    global model_xce
    features = model_xce.predict(images)
    return features


def get_features(x_batch, mode):
    if mode == 'res':
        x_batch = res_preprocess(x_batch)
        x_batch = get_features_res(x_batch)
    elif mode == 'inc':
        x_batch = inc_preprocess(x_batch)
        x_batch = get_features_inc(x_batch)
    elif mode == 'v3':
        x_batch = v3_preprocess(x_batch)
        x_batch = get_features_v3(x_batch)
    elif mode == 'xce':
        x_batch = xce_preprocess(x_batch)
        x_batch = get_features_xce(x_batch)
    return x_batch


def dump(data, name):
    pickle.dump(data, open(name, 'wb'), pickle.HIGHEST_PROTOCOL)


x_train, y_train, x_test = pickle.load(open('data.pickle', 'rb'))


def next_batch(_X, _Y, batch_size=128, mode='res'):
    image_size = utils.get_image_size(mode)

    num_batch = int(np.ceil(len(_Y) / config.batch_size))
    for i in range(num_batch):
        if i % 30 == 0:
            print("Batch: %d" % (i))
        x_batch, y_batch = [], []
        for path, label in zip(_X[i*config.batch_size:(i+1)*config.batch_size], _Y[i*config.batch_size:(i+1)*config.batch_size]):
            try:
                img = image.load_img(path, target_size=(image_size, image_size))
            except:
                continue

            img = image.img_to_array(img)
            x_batch.append(img)

            y_tmp = np.zeros((config.num_classes, ))
            y_tmp[label] = 1
            y_batch.append(y_tmp)

        x_batch = np.array(x_batch)
        x_batch = get_features(x_batch, mode)
        print(x_batch.shape)
        exit(0)

        y_batch = np.array(y_batch)

        yield x_batch, y_batch


def dump_data(mode):
    count = 0
    new_x_train, new_y_train = [], []
    for x, y in next_batch(x_train.copy(), y_train.copy(), config.batch_size, mode):
        for e_x, e_y in zip(x, y):
            path = 'x_train_' + mode + '/' + str(count) + '.pickle'
            pickle.dump(e_x, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)
            new_x_train.append(path)
            new_y_train.append(e_y)
            count += 1
    new_x_train = np.array(new_x_train)
    new_y_train = np.array(new_y_train)
    dump((new_x_train, new_y_train), mode + '.pickle')
    

# print("Dump res")
# dump_data(mode='res')
# print("Dump inc")
# dump_data(mode='inc')
# print("Dump v3")
# dump_data(mode='v3')
print("Dump xce")
dump_data(mode='xce')


def dump_test(_X, batch_size=config.batch_size, mode='res'):
    image_size = utils.get_image_size(mode)

    num_batch = int(np.ceil(len(_X) / config.batch_size))
    for i in range(num_batch):
        if i % 30 == 0:
            print("Batch %d" % (i))
        x_batch, paths = [], []
        for path in _X[i*config.batch_size:(i+1)*config.batch_size]:
            try:
                img = image.load_img(path, target_size=(image_size, image_size))
            except:
                print(path)
                continue

            img = image.img_to_array(img)
            x_batch.append(img)

            path = path.split('/')[1].split('.')[0]
            paths.append(path)

        x_batch = np.array(x_batch)
        x_batch = get_features(x_batch, mode)

        paths = np.array(paths)
        pickle.dump((x_batch, paths), open('x_test_' + mode + '/' + str(i) + '.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)


# print("Dump test res")
# dump_test(x_test, mode='res')
# print("Dump test inc")
# dump_test(x_test, mode='inc')
# print("Dump test v3")
# dump_test(x_test, mode='v3')
print("Dump test xce")
dump_test(x_test, mode='xce')
